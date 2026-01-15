import torch
import random

class Strategy:
    def __init__(self, model, action_dim=109):
        self.model = model
        self.action_dim = action_dim
        # 隣接リスト
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

    def get_action(self, state_vector, epsilon, state_info, my_team, current_step=0):
        with torch.no_grad():
            q_values = self.model(state_vector).clone()

        for action_idx in range(1, self.action_dim):
            cmd = self._decode_action(action_idx, state_info, my_team)
            if cmd == (0, 0, 0):
                q_values[0, action_idx] = -1e9
                continue

            c_type, src, dst = cmd
            
            if current_step > 45000 and c_type == 2:
                q_values[0, action_idx] -= 50.0
            
            if (c_type == 1 and dst in [4, 7]) or (c_type == 2 and src in [4, 7]):
                q_values[0, action_idx] += 2.0

            if c_type == 1: 
                source_pawn = state_info[src][3]
                if state_info[dst][0] != my_team:
                    target_pawn = state_info[dst][3]
                    # 相手の兵数 + 5人以上の余裕が必要
                    required = target_pawn + 5
                    if source_pawn < required:
                        q_values[0, action_idx] = -1e9
                else:
                    is_s = all(state_info[n][0] == my_team or state_info[n][0] == 0 for n in self.neighbors[src])
                    is_f = any(state_info[n][0] != my_team and state_info[n][0] != 0 for n in self.neighbors[dst])
                    if is_s and is_f:
                        q_values[0, action_idx] += 3.0
                    if state_info[dst][3] < 10:
                        q_values[0, action_idx] += 5.0

        # 探索と活用の選択
        if random.random() < epsilon:
            valid_indices = [i for i in range(self.action_dim) if q_values[0, i] > -1e7]
            action_idx = random.choice(valid_indices) if valid_indices else 0
        else:
            action_idx = q_values.max(1)[1].item()

        return action_idx, self._decode_action(action_idx, state_info, my_team)


    def _decode_action(self, action_idx, state_info, my_team):
        """インデックスをゲームコマンドに変換。実行不可なら(0,0,0)を返す"""
        if action_idx == 0: return (0, 0, 0) # 待機
        
        # 1. アップグレード (1-12)
        if 1 <= action_idx <= 12:
            subject = action_idx - 1
            # 自分の拠点かつレベルアップ可能なら実行
            if subject < len(state_info) and state_info[subject][0] == my_team:
                return (2, subject, 0)
            return (0, 0, 0)
        
        # 2. 移動アクションのデコード
        move_idx = action_idx - 13
        src = move_idx // 8
        dst_relative = move_idx % 8
        
        if src < 12 and src in self.neighbors:
            possible_dsts = self.neighbors[src]
            if dst_relative < len(possible_dsts):
                dst = possible_dsts[dst_relative]
                if state_info[src][0] == my_team:
                    return (1, src, dst)
        return (0, 0, 0)