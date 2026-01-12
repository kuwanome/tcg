import random
import math
from tcg.controller import Controller

class SniperPlayer(Controller):
    def __init__(self, team=None):
        super().__init__()
        self.team = team
        self.neighbors = {
            0: [1, 3, 4], 1: [0, 2, 4], 2: [1, 4, 5], 3: [0, 4, 6, 7],
            4: [0, 1, 2, 3, 5, 6, 7, 8], 5: [2, 4, 7, 8], 6: [3, 4, 7, 9],
            7: [3, 4, 5, 6, 8, 9, 10, 11], 8: [4, 5, 7, 11], 9: [6, 7, 10],
            10: [7, 9, 11], 11: [7, 8, 10]
        }
        self.high_value_targets = [4, 7, 3, 5, 6, 8]

    def team_name(self):
        return "Sniper_Ares_v5"

    def update(self, info):
        self.team = info[0]
        state = info[1]

        my_forts = [i for i, s in enumerate(state) if s[0] == self.team]
        if not my_forts: return (0, 0, 0)

        best_action = (0, 0, 0)
        max_score = -999999 

        for f_idx in my_forts:
            f_data = state[f_idx]
            my_pawn_count = f_data[3]
            level = f_data[4]

            # --- 1. 移動・攻撃の評価 (勝てる時だけ高得点) ---
            for target_idx in self.neighbors.get(f_idx, []):
                t_data = state[target_idx]
                t_owner = t_data[0]
                t_pawns = t_data[3]
                
                m_score = -5000 # デフォルトは「行かないほうがマシ」な低得点
                
                if t_owner == 0: # 中立拠点
                    # 【重要】敵の数 + 5以上の余剰がある時だけ出陣
                    if my_pawn_count > (t_pawns + 5):
                        m_score = 3000 + (my_pawn_count - t_pawns)
                
                elif t_owner != self.team: # 敵拠点
                    # 敵拠点へは、さらに慎重に (+10の余裕)
                    if my_pawn_count > (t_pawns + 10):
                        m_score = 2000 + (my_pawn_count - t_pawns)
                
                else: # 味方拠点への援護
                    if my_pawn_count > 40 and t_pawns < 15:
                        m_score = 500

                # 重要拠点ボーナス (勝てる見込みがある時のみ加算)
                if target_idx in self.high_value_targets and m_score > 0:
                    m_score += 1500

                if m_score > max_score:
                    max_score = m_score
                    best_action = (1, f_idx, target_idx)

            # --- 2. アップグレードの評価 (攻めるべき場所がない時の選択肢) ---
            if level < 3:
                # 攻めるのに十分な兵がいないなら、レベル上げを優先する
                u_score = 1000 if my_pawn_count > 15 else 500
                
                # 拠点レベルが低いほど優先
                u_score += (3 - level) * 500

                if u_score > max_score:
                    max_score = u_score
                    best_action = (2, f_idx, 0)

        return best_action