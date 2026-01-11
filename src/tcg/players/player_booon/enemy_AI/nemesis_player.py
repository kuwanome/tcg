import random

class NemesisPlayer:
    def __init__(self):
        self.team = None
        # nemesis_player.py の neighbors を strategy.py と同じものに修正
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
        return "Nemesis_v3"

    def update(self, info):
        self.team = info[0]
        state = info[1]

        my_forts = [i for i, s in enumerate(state) if s[0] == self.team]
        if not my_forts: return (0, 0, 0)

        # --- 1. 最優先：レベルアップ ---
        for f_idx in my_forts:
            if state[f_idx][4] < 3 and state[f_idx][3] > 20:
                return (2, f_idx, 0)

        # --- 2. 攻撃：隣接する「中立」または「敵」を狙う ---
        # 拠点をシャッフルして、特定の場所だけに固執しないようにする
        random.shuffle(my_forts)
        for f_idx in my_forts:
            if state[f_idx][3] > 12:
                targets = self.neighbors.get(f_idx, [])
                random.shuffle(targets)
                
                # まず中立を狙う
                for t in targets:
                    if state[t][0] == 0:
                        return (1, f_idx, t)
                
                # 次に敵を狙う
                for t in targets:
                    if state[t][0] != self.team and state[t][0] != 0:
                        return (1, f_idx, t)

        # --- 3. 防衛・供給：隣接する「弱っている味方」を助ける ---
        for f_idx in my_forts:
            if state[f_idx][3] > 25:
                for t in self.neighbors.get(f_idx, []):
                    if state[t][0] == self.team and state[t][3] < 10:
                        return (1, f_idx, t)

        return (0, 0, 0)