import torch
import random

class Strategy:
    def __init__(self, model, action_dim=48):
        self.model = model
        self.action_dim = action_dim
        # 要塞配置図に基づく正確な隣接リスト（2を排除）
        # strategy.py の neighbors を以下に書き換え
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
    def get_action(self, state_vector, epsilon, state_info, my_team):
        if random.random() < epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state_vector)
                action_idx = q_values.max(1)[1].item()

        cmd, sub, to = self._decode_action(action_idx, state_info, my_team)
        
        # 最終チェック：道がない、または対象が2なら待機
        if cmd == 1 and (to not in self.neighbors.get(sub, [])):
            return action_idx, (0, 0, 0)
        
        return action_idx, (cmd, sub, to)

    def _decode_action(self, action_idx, state_info, my_team):
        if action_idx == 0: return (0, 0, 0)
        
        # 1. アップグレード (1-12)
        if 1 <= action_idx <= 12:
            subject = action_idx - 1
            if subject == 2: return (0, 0, 0) # 2は無視
            if subject < len(state_info) and state_info[subject][0] == my_team:
                return (2, subject, 0)
            return (0, 0, 0)
        
        # 2. 移動 (13-48)
        move_offset = action_idx - 13
        subject = move_offset // 3
        dir_idx = move_offset % 3
        
        if subject == 2: return (0, 0, 0) # 2は無視
        if subject in self.neighbors and state_info[subject][0] == my_team:
            targets = self.neighbors[subject]
            if dir_idx < len(targets):
                return (1, subject, targets[dir_idx])
        
        return (0, 0, 0)