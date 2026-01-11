import torch
import random

class Strategy:
    def __init__(self, model, action_dim=109):
        self.model = model
        self.action_dim = action_dim
        # 正確な隣接リスト（2番を実質除外）
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
        """行動のインデックスと、デコードされた(cmd, sub, to)を返す"""
        if random.random() < epsilon:
            # 探索：ランダムに行動を選択
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # 活用：モデルの予測に基づき行動を選択
            with torch.no_grad():
                q_values = self.model(state_vector)
                action_idx = q_values.max(1)[1].item()

        # インデックスをゲーム用の命令(cmd, sub, to)に変換
        command = self._decode_action(action_idx, state_info, my_team)
        
        return action_idx, command

    def _decode_action(self, action_idx, state_info, my_team):
        if action_idx == 0: return (0, 0, 0) # 待機
        
        # 1. アップグレード (1-12)
        if 1 <= action_idx <= 12:
            subject = action_idx - 1
            if subject < len(state_info) and state_info[subject][0] == my_team:
                return (2, subject, 0)
            return (0, 0, 0)
        
        # 2. 移動 (13-108)
        # 12拠点 × 8方向 = 96アクション分
        move_offset = action_idx - 13
        subject = move_offset // 8  # 移動元の拠点
        dir_idx = move_offset % 8   # どの隣接拠点に行くか
        
        if subject < len(state_info) and state_info[subject][0] == my_team:
            targets = self.neighbors.get(subject, [])
            if dir_idx < len(targets):
                return (1, subject, targets[dir_idx])
        
        return (0, 0, 0)