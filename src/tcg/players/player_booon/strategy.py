import torch
import random

class Strategy:
    def __init__(self, model, action_dim=48):
        self.model = model
        self.action_dim = action_dim

    def get_action(self, state_vector, epsilon, state_info, my_team):
        # 1. AIまたはランダムによる行動選択
        if random.random() < epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state_vector)
                action_idx = q_values.max(1)[1].item()

        # 2. 数値をコマンドに変換 (ここを _decode_action に修正)
        command, subject, to = self._decode_action(action_idx, state_info, my_team)

        # 3. 兵力ガード（小出し防止ロジック）
        if command == 1: 
            my_hp_val = state_info[subject][3]
            my_total_hp = float(my_hp_val[0] if isinstance(my_hp_val, list) else my_hp_val)
            my_atk = my_total_hp // 2
            
            target_hp_val = state_info[to][3]
            target_hp = float(target_hp_val[0] if isinstance(target_hp_val, list) else target_hp_val)
            
            buffer = 5 if state_info[to][0] == 0 else 2
            
            if state_info[to][0] != my_team:
                if my_atk <= (target_hp + buffer) and my_total_hp < 40:
                    return action_idx, (0, 0, 0)

        return action_idx, (command, subject, to)

    def _decode_action(self, action_idx, state_info, my_team):
        """AIのインデックスをゲームのコマンド (cmd, sub, to) に変換"""
        if action_idx == 0: return (0, 0, 0)
        
        # 1-12: 強化
        if 1 <= action_idx <= 12:
            subject = action_idx - 1
            if subject < len(state_info) and state_info[subject][0] == my_team:
                return (2, subject, 0)
            return (0, 0, 0)
        
        # 13-47: 移動
        subject = (action_idx - 13) // 3
        if subject < len(state_info) and state_info[subject][0] == my_team:
            neighbors = state_info[subject][5]
            if neighbors:
                # 味方以外の拠点をターゲットにする
                targets = [n for n in neighbors if state_info[n][0] != my_team]
                if targets:
                    target_id = targets[action_idx % len(targets)]
                    return (1, subject, target_id)
        
        return (0, 0, 0)