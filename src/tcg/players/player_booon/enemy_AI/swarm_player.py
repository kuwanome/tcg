import random

class SwarmPlayer:
    def __init__(self):
        self.neighbors = {
            0: [1, 3, 4], 1: [0, 2, 4], 2: [1, 4, 5], 3: [0, 4, 6, 7],
            4: [0, 1, 2, 3, 5, 6, 7, 8], 5: [2, 4, 7, 8], 6: [3, 4, 7, 9],
            7: [3, 4, 5, 6, 8, 9, 10, 11], 8: [4, 5, 7, 11], 9: [6, 7, 10],
            10: [7, 9, 11], 11: [7, 8, 10]
        }

    def team_name(self): return "Zerg_Swarm"

    def update(self, info):
        team, state = info[0], info[1]
        my_forts = [i for i, s in enumerate(state) if s[0] == team]
        
        # 攻撃の敷居を極端に低く設定（12人いたら突撃）
        random.shuffle(my_forts)
        for f in my_forts:
            if state[f][3] > 12:
                targets = self.neighbors[f]
                # 中立があれば最優先、なければ敵へ
                neutrals = [t for t in targets if state[t][0] == 0]
                enemies = [t for t in targets if state[t][0] != team and state[t][0] != 0]
                
                if neutrals: return (1, f, random.choice(neutrals))
                if enemies: return (1, f, random.choice(enemies))
        
        return (0, 0, 0)