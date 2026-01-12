import random
from tcg.controller import Controller

class NemesisPlayer:
    def __init__(self):
        self.team = None
        # neighbors の定義
        self.neighbors = {
            0: [1, 3, 4],
            1: [0, 2, 4],
            2: [1, 4, 5],
            3: [0, 4, 6, 7],
            4: [0, 1, 2, 3, 5, 6, 7, 8],
            5: [2, 4, 7, 8],
            6: [3, 4, 7, 9],
            7: [3, 4, 5, 6, 8, 9, 10, 11],
            8: [4, 5, 7, 11],
            9: [6, 7, 10],
            10: [7, 9, 11],
            11: [7, 8, 10]
        }

    def team_name(self):
        return "Nemesis_v4"

    def update(self, info):
        self.team = info[0]
        state = info[1]

        my_forts = [i for i, s in enumerate(state) if s[0] == self.team]
        if not my_forts: return (0, 0, 0)

        # --- 1. 最優先：レベルアップ ---
        for f_idx in my_forts:
            if state[f_idx][4] < 3 and state[f_idx][3] > 20:
                return (2, f_idx, 0)

        # --- 2. 攻撃：敵の中で「最も兵が少ない拠点」を特定する ---
        enemy_forts = [i for i, s in enumerate(state) if s[0] != self.team and s[0] != 0]
        
        if enemy_forts:
            # 最も兵が少ない敵拠点を探す
            weakest_enemy = min(enemy_forts, key=lambda x: state[x][3])
            
            # 自分の拠点をシャッフルして多角的に攻める
            random.shuffle(my_forts)
            for f_idx in my_forts:
                # もし「最弱の敵」が隣接していたら、全兵力で叩き潰す
                if weakest_enemy in self.neighbors.get(f_idx, []):
                    if state[f_idx][3] > state[weakest_enemy][3] + 2:
                        return (1, f_idx, weakest_enemy)

        # --- 3. バックアップ：近くの中立拠点を奪う ---
        for f_idx in my_forts:
            if state[f_idx][3] > 12:
                targets = self.neighbors.get(f_idx, [])
                for t in targets:
                    if state[t][0] == 0: # 中立
                        return (1, f_idx, t)

        return (0, 0, 0)