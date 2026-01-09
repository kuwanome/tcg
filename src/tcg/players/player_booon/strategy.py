from tcg.config import fortress_limit

class Strategy:
    def __init__(self):
        pass

    def get_action(self, state):
        # 防御ロジック
        for i in range(12):
            if state[i][0] == 1:
                if state[i][3] < fortress_limit[state[i][2]] * 0.25:
                    for neighbor in state[i][5]:
                        if state[neighbor][0] == 1 and state[neighbor][3] > 15:
                            return 1, neighbor, i

        # アップグレードロジック
        for i in range(12):
            if state[i][0] == 1:
                level = state[i][2]
                if level < 5 and state[i][4] == 0:
                    required = fortress_limit[level] // 2
                    has_enemy = any(state[n][0] == 2 for n in state[i][5])
                    threshold = 2.0 if has_enemy else 1.2
                    if state[i][3] >= required * threshold:
                        return 2, i, 0

        # 攻撃ロジック
        my_forts = [i for i in range(12) if state[i][0] == 1]
        for my_id in my_forts:
            if state[my_id][3] < 10: continue
            neighbors = state[my_id][5]
            neutrals = [n for n in neighbors if state[n][0] == 0]
            if neutrals:
                return 1, my_id, neutrals[0]
            enemies = [n for n in neighbors if state[n][0] == 2]
            for target_id in enemies:
                if state[my_id][3] > state[target_id][3] * 2:
                    return 1, my_id, target_id

        return 0, 0, 0