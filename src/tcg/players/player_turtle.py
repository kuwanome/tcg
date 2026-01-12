import random
from tcg.controller import Controller

class TurtlePlayer(Controller):
    def __init__(self):
        # 拠点間のつながり
        self.neighbors = {
            0: [1, 3, 4], 1: [0, 2, 4], 2: [1, 4, 5], 3: [0, 4, 6, 7],
            4: [0, 1, 2, 3, 5, 6, 7, 8], 5: [2, 4, 7, 8], 6: [3, 4, 7, 9],
            7: [3, 4, 5, 6, 8, 9, 10, 11], 8: [4, 5, 7, 11], 9: [6, 7, 10],
            10: [7, 9, 11], 11: [7, 8, 10]
        }

    def team_name(self):
        return "Turtle_Steel"

    def update(self, info):
        # 変数の定義（これが抜けていました）
        team = info[0]
        state = info[1]
        my_forts = [i for i, s in enumerate(state) if s[0] == team]
        
        if not my_forts:
            return (0, 0, 0)

        # 1. 全拠点のレベルを3にするまで徹底的に内政
        low_level_forts = [f for f in my_forts if state[f][4] < 3]
        if low_level_forts:
            # 兵力が一番多い低レベル拠点を選んでアップグレード
            target = max(low_level_forts, key=lambda x: state[x][3])
            if state[target][3] > 20:
                return (2, target, 0)
        
        # 2. 兵力が上限に近い拠点から、前線（敵と隣接）へ自動送金
        for f in my_forts:
            if state[f][3] > 40: # 兵が溜まったら
                # 隣接する味方拠点に送る
                for n in self.neighbors[f]:
                    if state[n][0] == team and state[n][3] < state[f][3]:
                        return (1, f, n) # 味方への補給

        # 3. 隙があれば中立拠点を一つだけ確保
        for f in my_forts:
            if state[f][3] > 15:
                for n in self.neighbors[f]:
                    if state[n][0] == 0:
                        return (1, f, n)

        return (0, 0, 0)