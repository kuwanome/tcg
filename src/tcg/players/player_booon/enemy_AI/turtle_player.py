import random

class TurtlePlayer:
    def __init__(self):
        self.neighbors = {
            0: [1, 3, 4], 1: [0, 2, 4], 2: [1, 4, 5], 3: [0, 4, 6, 7],
            4: [0, 1, 2, 3, 5, 6, 7, 8], 5: [2, 4, 7, 8], 6: [3, 4, 7, 9],
            7: [3, 4, 5, 6, 8, 9, 10, 11], 8: [4, 5, 7, 11], 9: [6, 7, 10],
            10: [7, 9, 11], 11: [7, 8, 10]
        }

    def team_name(self): return "Turtle_Wall"

    def update(self, info):
        team, state = info[0], info[1]
        my_forts = [i for i, s in enumerate(state) if s[0] == team]
        if not my_forts: return (0, 0, 0)

        # 1. 要塞化：レベル3未満の拠点を徹底的にアップグレード
        random.shuffle(my_forts)
        for f in my_forts:
            if state[f][4] < 3 and state[f][3] > 18:
                return (2, f, 0)

        # 2. 満潮攻撃：兵数が限界に近い(42以上)拠点のみ攻撃
        for f in my_forts:
            if state[f][3] > 42:
                targets = [t for t in self.neighbors[f] if state[t][0] != team]
                if targets:
                    return (1, f, random.choice(targets))
        return (0, 0, 0)