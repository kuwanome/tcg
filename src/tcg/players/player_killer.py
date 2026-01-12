import random
from tcg.controller import Controller

class KillerPlayer(Controller):
    def __init__(self):
        # これを全てのプレイヤーの __init__ 内にある self.neighbors に上書きしてください
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

    def team_name(self): return "Killer_Elite"

    def update(self, info):
        team, state = info[0], info[1]
        my_forts = [i for i, s in enumerate(state) if s[0] == team]
        if not my_forts: return (0, 0, 0)

        # 1. 前線の定義（敵に隣接している自陣）
        frontline = []
        for f in my_forts:
            if any(state[t][0] != team for t in self.neighbors.get(f, [])):
                frontline.append(f)

        # 2. 攻め：敵の「兵数が極端に少ない」隣接拠点を即座に奪う
        for f in frontline:
            targets = self.neighbors.get(f, [])
            for t in targets:
                if state[t][0] != team and state[f][3] > state[t][3] + 5:
                    return (1, f, t)

        # 3. 補給：後方の拠点の兵を、最も兵が少ない前線拠点へ送る
        backline = [f for f in my_forts if f not in frontline]
        if backline and frontline:
            source = max(backline, key=lambda x: state[x][3])
            if state[source][3] > 10:
                target = min(frontline, key=lambda x: state[x][3])
                return (1, source, target)

        # 4. 強化：前線の拠点を優先的にレベル2まで上げる（3は時間がかかるので後回し）
        for f in frontline:
            if state[f][4] < 2 and state[f][3] > 15:
                return (2, f, 0)

        # 5. 拡張：とりあえず中立があれば行く
        for f in frontline:
            targets = [t for t in self.neighbors.get(f, []) if state[t][0] == 0]
            if targets and state[f][3] > 10:
                return (1, f, random.choice(targets))

        return (0, 0, 0)