from tcg.controller import Controller
from tcg.config import fortress_cool, fortress_limit
from collections import deque


class Gemini7Player(Controller):
    """
    改良点:
    1. 【物流革命】BFSで「敵までの距離」を計算し、後方から前線へ一直線に兵を送る
    2. 【終盤ラッシュ】残り時間が少なくなると内政を捨てて全軍突撃する
    3. 【集中攻撃】倒せると判断した敵拠点に対し、隣接する全部隊で波状攻撃を仕掛ける
    """

    def __init__(self) -> None:
        super().__init__()
        self.step = 0
        self.MAX_STEP = 50000


    def team_name(self) -> str:
        return "Imamu"


    def get_distance_map(self, state, team):
        """
        各要塞から「最も近い敵（または中立）」までの距離を計算する (幅優先探索)
        Returns: {fort_id: distance, ...}
        """
        distances = {i: 999 for i in range(12)}
        queue = deque()


        # ターゲット（敵または中立）を距離0として初期化
        targets = [i for i, s in enumerate(state) if s[0] != team]
        if not targets:
            # 敵がいない（完全勝利目前）なら距離計算不要
            return distances


        for t in targets:
            distances[t] = 0
            queue.append(t)


        while queue:
            current = queue.popleft()
            current_dist = distances[current]
           
            # 隣接ノードをチェック
            neighbors = state[current][5]
            for n in neighbors:
                if distances[n] > current_dist + 1:
                    distances[n] = current_dist + 1
                    queue.append(n)
       
        return distances


    def update(self, info) -> tuple[int, int, int]:
        team, state, moving_pawns, spawning_pawns, done = info
        self.step += 1


        # --------------------------------------------------
        # 状況認識
        # --------------------------------------------------
        my_forts = [i for i, s in enumerate(state) if s[0] == team]
       
        # 行動リスト (priority, command, subject, to)
        actions = []


        # 終盤フラグ（残り時間が80%を切ったらラッシュ）
        is_endgame = self.step > (self.MAX_STEP * 0.8)


        # 敵までの距離マップを作成（物流制御用）
        dist_map = self.get_distance_map(state, team)


        # どこが攻撃されているか把握
        under_attack_map = {}
        for pawn in moving_pawns:
            p_team, _, _, to_id, _ = pawn[:5]
            if p_team != team and state[to_id][0] == team:
                under_attack_map[to_id] = under_attack_map.get(to_id, 0) + 1


        # ====================================================================
        # 戦略1: 緊急防衛 (Emergency Defense) [最優先]
        # ====================================================================
        for target_id, count in under_attack_map.items():
            if state[target_id][3] < 15: # 防衛が危ない場合
                neighbors = state[target_id][5]
                for n_id in neighbors:
                    if state[n_id][0] == team and state[n_id][3] > 10:
                        # 救援を送る
                        actions.append((1000, 1, n_id, target_id))


        # ====================================================================
        # 戦略2: 攻撃と拡大 (Attack & Expand)
        # ====================================================================
        for i in my_forts:
            s = state[i]
            my_pawns = s[3]
            neighbors = s[5]
            attacking_power = my_pawns // 2


            # 攻撃可能な隣接拠点を探す
            for target_id in neighbors:
                ts = state[target_id]
                target_team = ts[0]
                target_pawns = ts[3]
               
                # --- 中立への攻撃 ---
                if target_team == 0:
                    # 確実に勝てるなら取る（中盤以降は少し余裕を持つ）
                    margin = 2
                    if attacking_power > target_pawns + margin:
                        prio = 200 - target_pawns # 弱いところから取る
                        actions.append((prio, 1, i, target_id))


                # --- 敵への攻撃 ---
                elif target_team != team:
                    # 敵の増援予測（移動中に増える分）
                    # 簡易計算: 距離を考慮せず一定のバッファを見る
                    production_buffer = 5
                   
                    # 終盤なら特攻、普段は勝てる時だけ
                    win_ratio = 1.0 if is_endgame else 1.1


                    if attacking_power > (target_pawns + production_buffer) * win_ratio:
                        # 敵の重要拠点(4,7)は優先度激高
                        bonus = 100 if target_id in [4, 7] else 0
                        # 敵が弱っているところを叩く
                        prio = 300 + bonus + (100 - target_pawns)
                        actions.append((prio, 1, i, target_id))


        # ====================================================================
        # 戦略3: 物流ライン (Supply Chain)
        # 後方(敵から遠い) -> 前方(敵に近い) へ兵を送る
        # ====================================================================
        for i in my_forts:
            # 既に攻撃や防衛でアクションが決まっているならスキップしたいが、
            # ここでは簡易的に「兵士がある程度残っている」場合のみ物流を考える
            if state[i][3] < 10: continue


            my_dist = dist_map[i]
            neighbors = state[i][5]
           
            best_target = -1
            min_dist = my_dist # 自分より敵に近い場所を探す


            # 隣接する「味方」の中で、最も敵に近い（距離が小さい）拠点を探す
            for n_id in neighbors:
                if state[n_id][0] == team:
                    if dist_map[n_id] < min_dist:
                        min_dist = dist_map[n_id]
                        best_target = n_id
           
            # 送り先が見つかり、かつ自分の兵士が溢れそう、または後方なら送る
            if best_target != -1:
                # 自分が安全地帯(距離2以上)なら積極的に送る
                if my_dist >= 2 or state[i][3] > 30:
                    # 優先度は攻撃より低い(50)
                    actions.append((50, 1, i, best_target))


        # ====================================================================
        # 戦略4: 内政 (Economy)
        # ※終盤(is_endgame)は一切アップグレードしない！
        # ====================================================================
        if not is_endgame:
            for i in my_forts:
                s = state[i]
                level = s[2]
                pawns = s[3]
                is_upgrading = s[4] > 0
               
                # エラー回避のための上限チェック
                limit = fortress_limit[level] if level < len(fortress_limit) else 999999
               
                # レベル上限(50)未満で、アップグレード中でない
                if level < 50 and not is_upgrading:
                    # 敵に近い(距離1)なら、兵士を溜めたいので基準を厳しく(80%)
                    # 安全地帯(距離2以上)なら、すぐ投資する(40%)
                    threshold = 0.8 if dist_map[i] <= 1 else 0.4
                   
                    if pawns > limit * threshold:
                        # 優先度 100 (移動より優先度は高いが攻撃よりは低い)
                        actions.append((100, 2, i, 0))


        # ====================================================================
        # 意思決定
        # ====================================================================
        if actions:
            # 優先度順にソートして一番高いものを実行
            actions.sort(key=lambda x: x[0], reverse=True)
            return actions[0][1], actions[0][2], actions[0][3]


        return 0, 0, 0