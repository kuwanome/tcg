import math
from collections import deque
from tcg.controller import Controller
from tcg.config import fortress_cool, fortress_limit


class Gemini5Player(Controller):
    def __init__(self):
        super().__init__()
        self.step_count = 0

        # 戦略的価値マップ
        # 4, 7: 中央 (最重要)
        # 1, 10: 四角い砦 (攻撃力が高いので重要)
        self.FORT_IMPORTANCE = {
            0: 1.0,
            1: 1.5,
            2: 1.0,
            3: 1.1,
            4: 2.5,
            5: 1.1,
            6: 1.1,
            7: 2.5,
            8: 1.1,
            9: 1.0,
            10: 1.5,
            11: 1.0,
        }

    def team_name(self) -> str:
        return "Optimus"

    def get_distance_to_enemy(self, state, my_id, enemy_id):
        """BFSで敵までの距離マップを作成"""
        distances = {i: 999 for i in range(12)}
        queue = deque()
        # 序盤は中立もターゲットに含めて前線を広げる
        targets = [enemy_id, 0] if self.step_count < 2000 else [enemy_id]

        for i in range(12):
            if state[i][0] in targets:
                distances[i] = 0
                queue.append(i)

        while queue:
            curr = queue.popleft()
            for n in state[curr][5]:
                if distances[n] > distances[curr] + 1:
                    distances[n] = distances[curr] + 1
                    queue.append(n)
        return distances

    def predict_future_balance(self, state, moving_pawns, my_id, en_id):
        """
        未来予測: 全兵士が着弾した後の「実質兵士数」を計算する
        戻り値: {fort_id: スコア} (プラスなら味方優勢、マイナスなら敵優勢)
        """
        future_balance = {}

        for i in range(12):
            owner, kind, _, troops, _, _ = state[i]

            # 1. 現在の兵士数をセット
            if owner == my_id:
                future_balance[i] = troops
            elif owner == en_id:
                future_balance[i] = -troops
            else:
                # 中立は「倒すべき壁」としてマイナス計上
                # (少し余分に見積もって確実に落とす)
                future_balance[i] = -(troops + 2)

        # 2. 移動中の兵士を加算
        for p in moving_pawns:
            # p構造想定: [team, kind, from, to, ...]
            if len(p) < 4:
                continue
            p_team, p_kind, _, p_to = p[0], p[1], p[2], p[3]

            # 攻撃力補正 (四角い砦の兵種1は強い)
            power = 1.2 if p_kind == 1 else 0.8

            if p_to in future_balance:
                if p_team == my_id:
                    future_balance[p_to] += 1.0  # 援軍は数そのもの
                elif p_team == en_id:
                    future_balance[p_to] -= power  # 敵攻撃は補正あり

        return future_balance

    def update(self, info):
        team, state, moving, spawning, done = info
        self.step_count += 1
        my_id = team
        en_id = 2 if my_id == 1 else 1

        # 各種データ計算
        future_map = self.predict_future_balance(state, moving, my_id, en_id)
        dist_map = self.get_distance_to_enemy(state, my_id, en_id)
        my_forts = [i for i in range(12) if state[i][0] == my_id]

        actions = []  # (priority_score, cmd, src, dst)

        for src in my_forts:
            # 情報の展開
            kind = state[src][1]
            level = state[src][2]
            troops = state[src][3]
            is_upgrading = state[src][4] != -1
            capacity = fortress_limit[level]

            # 攻撃時の威力 (自分の種類による)
            my_atk_power_rate = 1.2 if kind == 1 else 0.8
            sent_amount = troops / 2
            damage = sent_amount * my_atk_power_rate

            # === 1. 緊急回避 (Overflow Check) ===
            # 兵士があふれる寸前なら、とにかく敵へ投げつける
            if troops >= capacity * 0.9:
                neighbors = state[src][5]
                # 敵を優先、いなければ中立
                targets = sorted(
                    neighbors,
                    key=lambda x: (0 if state[x][0] == en_id else 1, state[x][3]),
                )
                target = targets[0]
                # 優先度Max(9999)で登録
                actions.append((9999, 1, src, target))
                continue

            # === 2. アップグレード (Upgrade) ===
            if not is_upgrading and level < 5:
                cost = capacity / 2
                # 条件: コストがあり、未来予測で安全(プラス)であり、敵が隣接していない
                if troops > cost * 1.2 and future_map[src] > cost + 5:
                    if dist_map[src] >= 1:  # 接敵していない
                        # 重要拠点やレベルが低いところを優先
                        score = 100 + self.FORT_IMPORTANCE[src] * 20 - (level * 10)
                        actions.append((score, 2, src, 0))

            # === 3. 移動・攻撃 (Move / Attack) ===
            # 兵士がある程度いないと動かない
            if troops < 5:
                continue

            neighbors = state[src][5]
            for dst in neighbors:
                target_owner = state[dst][0]
                future_val = future_map[dst]  # 正なら味方優勢、負なら敵優勢

                # A. 敵・中立への攻撃
                if target_owner != my_id:
                    # 既に味方や自分が攻撃していて、オーバーキルになるなら控える
                    if future_val > 5:
                        continue

                    # 攻撃で制圧できるか？ (future_valはマイナスなので、damageを足してプラスになれば勝ち)
                    if future_val + damage > 2:
                        # 確実に落とせる
                        importance = self.FORT_IMPORTANCE[dst]
                        # 優先度計算: 重要度 + 敵なら高配点
                        prio = 200 + importance * 50
                        if target_owner == en_id:
                            prio += 50

                        actions.append((prio, 1, src, dst))

                    # 制圧できないが、敵(ID=2)なら削る価値あり (四角い砦からなら特に)
                    elif target_owner == en_id and kind == 1:
                        actions.append((50, 1, src, dst))

                # B. 味方への輸送・救援
                else:
                    # 救援: ピンチ(マイナス)の味方を救えるか
                    if future_val < 0 and future_val + sent_amount > 0:
                        prio = 500 + self.FORT_IMPORTANCE[dst] * 20  # 超優先
                        actions.append((prio, 1, src, dst))

                    # 輸送: 前線(敵に近い方)へ送る
                    elif dist_map[src] > dist_map[dst]:
                        # 送り先があふれていないか確認
                        target_cap = fortress_limit[state[dst][2]]
                        if state[dst][3] < target_cap * 0.7:
                            prio = 50 + (troops / capacity) * 30
                            actions.append((prio, 1, src, dst))

        # 最もスコアが高い行動を1つ選ぶ
        if actions:
            actions.sort(key=lambda x: x[0], reverse=True)
            best_action = actions[0]
            return best_action[1], best_action[2], best_action[3]

        return 0, 0, 0
