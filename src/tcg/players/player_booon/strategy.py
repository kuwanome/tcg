import torch
import random
from tcg.config import fortress_limit

class Strategy:
    def __init__(self, model, action_size):
        self.model = model
        self.action_size = action_size

    def get_action(self, state_tensor, epsilon, state_raw):
        """AI(モデル)の推論と従来のルールを組み合わせた行動選択"""
        if random.random() < epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                action_values = self.model(state_tensor)
                action_idx = torch.argmax(action_values).item()
        
        return action_idx, self._idx_to_command(action_idx, state_raw)

    def _idx_to_command(self, idx, state):
        """AIの出力インデックス(0~47)をゲームのコマンドに変換"""
        sub_id = idx // 4
        act_type = idx % 4

        if state[sub_id][0] != 1:
            return 0, 0, 0

        neighbors = state[sub_id][5]

        if act_type == 0: # 待機
            return 0, 0, 0
        elif act_type == 1: # アップグレード
            return 2, sub_id, 0
        elif act_type == 2: # 攻撃
            targets = [n for n in neighbors if state[n][0] != 1]
            if targets:
                target_id = min(targets, key=lambda n: state[n][3])
                return 1, sub_id, target_id
        elif act_type == 3: # 援護
            friends = [n for n in neighbors if state[n][0] == 1]
            if friends:
                target_id = min(friends, key=lambda n: state[n][3])
                return 1, sub_id, target_id

        return 0, 0, 0