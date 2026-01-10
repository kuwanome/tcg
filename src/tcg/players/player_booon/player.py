import torch
import random
from tcg.controller import Controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class booon(Controller):
    def __init__(self, model, strategy, mode="train"):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.mode = mode
        self.trainer = None  # main_train.py でセットされます
        
        self.epsilon = 1.0 if mode == "train" else 0.05
        self.last_state = None
        self.last_action = None
        self.last_info = None

    def team_name(self):
        """ゲームエンジンに表示されるチーム名"""
        return "booon"

    def _get_state_vector(self, state):
        res = []
        for s in state:
            team = 1.0 if s[0] == 1 else (-1.0 if s[0] == 2 else 0.0)
            res.extend([team, s[1], s[2]/5.0, min(s[3]/50.0, 1.0), s[4]/100.0])
        return torch.FloatTensor(res).unsqueeze(0).to(device)

    def update(self, info):
        team, state, pawn, SpawnPoint, done = info
        current_state_vector = self._get_state_vector(state)

        # 学習フェーズ (Trainerのメモリへ保存)
        if self.mode == "train" and self.last_state is not None and self.trainer is not None:
            from trainer import calculate_reward
            reward = calculate_reward(info, self.last_info)
            self.trainer.memory.push(self.last_state, self.last_action, reward, current_state_vector, done)
            self.trainer.train_step()

        # 行動選択フェーズ (Strategyを使用)
        action_idx, command = self.strategy.get_action(current_state_vector, self.epsilon, state)
        
        self.last_state = current_state_vector
        self.last_action = action_idx
        self.last_info = info
        
        if done:
            self.last_state = None

        return command