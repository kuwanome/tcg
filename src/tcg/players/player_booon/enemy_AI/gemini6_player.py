"""
Imperial Strategist V28 (Cross Counter)

「肉を切らせて骨を断つ」カウンター特化型AI。
敵が片側から攻めてきた際、その方面は撤退して遅延させつつ、
手薄になった逆サイドへ総攻撃を仕掛けて戦線崩壊を狙う。

改良点:
1. サイド認識: 砦を左右・中央に分類し、戦場のバランスを把握する。
2. クロスカウンター: 味方が撤退（敗走）を余儀なくされた際、逆サイドの攻撃優先度を劇的に引き上げ、同時多発的な反撃を行う。
3. 閾値緩和: カウンター時は、通常なら攻めない「五分五分」の状況でも攻撃を許可し、相手の計算を狂わせる。
"""

from collections import deque
from tcg.controller import Controller
from tcg.config import fortress_limit

# 定数
TEAM_NEUTRAL = 0
TEAM_MY = 1
TEAM_ENEMY = 2

# ユニットの戦闘力
POWER_FAST = 0.65   # 丸い砦 (Kind 0)
POWER_STRONG = 0.95 # 四角い砦 (Kind 1)

# キャッチボールを行うペア
STORAGE_LINKS = [
    {11, 10},
    {10, 9}
]

SQUARE_FORTS = [4, 7]

# 砦のサイド分類 (AI視点)
# 11(左下), 9(右下), 10(中下)
# 0(右上), 2(左上), 1(中上) ... FlipによりIDが変わるため、配置ベースで定義
# AI視点(Team 1)では:
# 左: 11, 8, 6, 2
# 右: 9, 7, 5, 0  (注: 盤面配置による。Gameクラスのpos_fortress参照推奨だが、簡易的に定義)
# ユーザー視点0(左) -> AI視点11
# ユーザー視点2(右) -> AI視点9
SIDE_MAP = {
    11: 'LEFT',  8: 'LEFT',  6: 'LEFT',  2: 'LEFT',
    9: 'RIGHT', 7: 'RIGHT', 5: 'RIGHT', 0: 'RIGHT',
    10: 'CENTER', 1: 'CENTER', 4: 'CENTER', 3: 'CENTER' # 3,4は中央付近
}

class test(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.step = 0

    def team_name(self) -> str:
        return "Imperial_V28"

    def is_square(self, fort_id: int) -> bool:
        return fort_id in SQUARE_FORTS

    def calculate_distance_to_target(self, state) -> list[int]:
        """全要塞から「敵または中立」までの最短距離を計算"""
        distances = [999] * 12
        queue = deque()

        targets = [i for i in range(12) if state[i][0] != TEAM_MY]
        if not targets: return [0] * 12

        for t in targets:
            distances[t] = 0
            queue.append(t)

        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            for n in state[curr][5]:
                if distances[n] > curr_dist + 1:
                    distances[n] = curr_dist + 1
                    queue.append(n)
        return distances

    def check_storage_link(self, u, v):
        pair = {u, v}
        return pair in STORAGE_LINKS

    def update(self, info) -> tuple[int, int, int]:
        team_id, state, moving_pawns, spawning_pawns, done = info
        self.step += 1

        # =================================================================
        # 0. 開幕アップグレード (初動ブースト)
        # =================================================================
        if self.step < 300:
            for i in range(12):
                if state[i][0] == TEAM_MY:
                    if state[i][2] < 3:
                        limit = fortress_limit[state[i][2]]
                        if state[i][3] >= limit and state[i][4] == -1:
                            return 2, i, 0

        # =================================================================
        # 1. 戦況分析
        # =================================================================
        
        incoming_damage = [0.0] * 12
        for p in moving_pawns:
            if len(p) < 4: continue
            p_team, p_kind, _, p_to = p[0], p[1], p[2], p[3]
            if not isinstance(p_to, int) or not (0 <= p_to < 12): continue

            dmg = POWER_STRONG if p_kind == 1 else POWER_FAST
            if p_team == TEAM_ENEMY:
                incoming_damage[p_to] += dmg

        dist_map = self.calculate_distance_to_target(state)
        aggression = max(0.6, 1.0 - (self.step / 10000))

        my_forts = [i for i in range(12) if state[i][0] == TEAM_MY]
        
        # --- 撤退・ピンチ判定 ---
        retreating_side = None
        for i in my_forts:
            # 自分の兵数 < 来るダメージ -> 撤退必至
            if incoming_damage[i] > state[i][3] * 1.1:
                side = SIDE_MAP.get(i, 'CENTER')
                if side != 'CENTER':
                    retreating_side = side
                    break # 片側でも崩れそうならフラグを立てる

        # =================================================================
        # 2. 行動決定
        # =================================================================
        
        best_score = 0
        best_cmd = (0, 0, 0)

        if not my_forts: return 0, 0, 0

        for i in my_forts:
            my_state = state[i]
            my_kind = my_state[1]
            my_level = my_state[2]
            curr_troops = my_state[3]
            upgrade_timer = my_state[4]
            neighbors = my_state[5]
            
            cap = fortress_limit[my_level] if my_level < len(fortress_limit) else 50
            upgrade_cost = cap // 2
            
            is_overflowing = curr_troops >= cap
            is_near_overflow = curr_troops >= cap * 0.85
            is_high_density = curr_troops >= cap * 0.90
            
            my_side = SIDE_MAP.get(i, 'CENTER')

            # --- カウンターモード判定 ---
            # 自分が撤退中でない、かつ、逆サイドが撤退中なら「カウンターチャンス」
            is_counter_opportunity = False
            if retreating_side and my_side != retreating_side and incoming_damage[i] == 0:
                is_counter_opportunity = True

            # --- [行動 B] アップグレード (最優先判定) ---
            if upgrade_timer == -1 and curr_troops >= upgrade_cost and my_level < 5:
                if incoming_damage[i] == 0:
                    # カウンターチャンス時は、LvUPよりも攻撃を優先したいのでスコアを下げる
                    if is_counter_opportunity:
                        score = 5000 # 攻撃(15000)に負ける
                    else:
                        score = 20000 # 絶対優先
                    
                    if score > best_score:
                        best_score = score
                        best_cmd = (2, i, 0)
            
            # --- [行動 A] 攻撃・移動 ---
            sending_count = int(curr_troops // 2)
            
            if sending_count >= 1:
                unit_power = POWER_STRONG if my_kind == 1 else POWER_FAST
                my_attack_power = sending_count * unit_power

                for target in neighbors:
                    t_state = state[target]
                    t_team = t_state[0]
                    t_troops = t_state[3]
                    
                    score = -float('inf')

                    # (1) 中立要塞
                    if t_team == TEAM_NEUTRAL:
                        if self.is_square(target):
                            if is_high_density: score = 1000
                            else: score = -9999
                        else:
                            is_safe = True
                            for tn in t_state[5]: 
                                if state[tn][0] == TEAM_ENEMY: is_safe = False; break
                            
                            margin = 2.0
                            score_base = 3000
                            
                            # カウンターチャンスなら、中立確保もアグレッシブに
                            if is_counter_opportunity:
                                margin = -5.0 # 強引に
                                score_base = 12000 # UP(5000)に勝つ

                            required = t_troops + margin

                            if my_attack_power > required:
                                score = score_base
                                if is_safe: score += 500
                            elif is_high_density:
                                score = 2000

                    # (2) 敵要塞
                    elif t_team == TEAM_ENEMY:
                        predicted_def = t_troops + incoming_damage[target]
                        
                        # カウンターアタック！
                        if is_counter_opportunity:
                            # 敵が攻めてきている隙に、逆サイドの敵拠点を強襲
                            # 多少負けていても削りに行く
                            if my_attack_power > predicted_def * 0.8:
                                score = 15000 # 超優先
                            else:
                                score = 4000 # 特攻
                        
                        else:
                            # 通常時
                            if t_troops < 10 and my_attack_power > predicted_def + 2.0:
                                score = 8000
                            elif is_high_density:
                                score = 2500 
                            elif my_attack_power > (predicted_def + 5.0) * aggression:
                                score = 4000
                            else:
                                score = -5000

                    # (3) 味方への輸送
                    elif t_team == TEAM_MY:
                        if my_level < 5:
                            if not is_near_overflow: continue

                        dist_diff = dist_map[i] - dist_map[target]
                        
                        # 道路在庫
                        is_storage_pair = self.check_storage_link(i, target)
                        
                        if is_storage_pair:
                            if is_high_density:
                                score = 3000
                            else:
                                score = -2000
                        else:
                            if dist_diff > 0:
                                score = 1500
                                if my_kind == 1: score += 500
                            else:
                                score = -1000

                    if score > best_score:
                        best_score = score
                        best_cmd = (1, i, target)
            
            # --- [行動 C] 緊急避難 (生存最優先) ---
            if incoming_damage[i] > curr_troops * 1.1:
                best_escape = -1
                max_dist = -1
                for target in neighbors:
                    if state[target][0] == TEAM_MY:
                        if dist_map[target] > max_dist:
                            max_dist = dist_map[target]
                            best_escape = target
                
                if best_escape != -1:
                    score = 30000 
                    if score > best_score:
                        best_score = score
                        best_cmd = (1, i, best_escape)

        return best_cmd