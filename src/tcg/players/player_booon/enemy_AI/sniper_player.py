import random

class SniperPlayer:
    def __init__(self):
        self.neighbors = {
            0: [1, 3, 4], 1: [0, 2, 4], 2: [1, 4, 5], 3: [0, 4, 6, 7],
            4: [0, 1, 2, 3, 5, 6, 7, 8], 5: [2, 4, 7, 8], 6: [3, 4, 7, 9],
            7: [3, 4, 5, 6, 8, 9, 10, 11], 8: [4, 5, 7, 11], 9: [6, 7, 10],
            10: [7, 9, 11], 11: [7, 8, 10]
        }
        # 重要拠点のリスト（接続数が多い順）
        self.high_value_targets = [4, 7, 3, 5, 6, 8]

    def team_name(self): return "Hub_Sniper"

    def update(self, info):
        team, state = info[0], info[1]
        my_forts = [i for i, s in enumerate(state) if s[0] == team]
        
        # 1. 重要拠点が敵のものなら、隣接する自陣から全力投入
        for target in self.high_value_targets:
            if state[target][0] != team:
                for f in my_forts:
                    if target in self.neighbors[f] and state[f][3] > 15:
                        return (1, f, target)
        
        # 2. 重要拠点を守る（レベルアップ）
        for target in self.high_value_targets:
            if state[target][0] == team and state[target][4] < 3 and state[target][3] > 20:
                return (2, target, 0)

        # 3. 暇なら近くの敵へ
        for f in my_forts:
            if state[f][3] > 30:
                targets = [t for t in self.neighbors[f] if state[t][0] != team]
                if targets: return (1, f, random.choice(targets))
        return (0, 0, 0)