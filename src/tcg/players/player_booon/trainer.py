import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
from collections import deque

# --- デバイスの設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class Trainer:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        # GPU性能を活かすためバッチサイズを128に強化（学習速度と安定性の向上）
        self.batch_size = 128 
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = ReplayMemory(20000)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(state_batch).to(device)
        action_batch = torch.tensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.tensor(reward_batch).float().to(device)
        next_state_batch = torch.cat(next_state_batch).to(device)
        done_batch = torch.tensor(done_batch).float().to(device)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        with torch.no_grad():
            # Double DQN的な考え方を取り入れ、ターゲットの計算を安定化
            next_state_values = self.target_model(next_state_batch).max(1)[0]
            expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピングを追加（学習が爆発して壊れるのを防ぐ、プロの技です）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

def calculate_reward(current_info, last_info):
    """
    攻撃的・効率的・戦略的な報酬設計
    """
    reward = 0.0
    team, state, pawn, SpawnPoint, done = current_info
    _, last_state, _, _, _ = last_info
    
    # 1. 拠点の増減（最も重要な指標）
    # 奪取報酬を大きくし、喪失ペナルティをさらに大きくして「防衛」も意識させる
    my_forts = sum(1 for s in state if s[0] == 1)
    last_my_forts = sum(1 for s in last_state if s[0] == 1)
    
    if my_forts > last_my_forts: reward += 10.0  # 拠点奪取！(大幅強化)
    if my_forts < last_my_forts: reward -= 12.0  # 拠点喪失... (守りも大事)

    # 2. 敵拠点のHPを削ったことへの報酬（攻撃性を高める）
    # 拠点奪取に至らなくても「ダメージを与えた」ことを褒める（Dense Reward化）
    enemy_forts_hp = sum(s[2] for s in state if s[0] == 2)
    last_enemy_forts_hp = sum(s[2] for s in last_state if s[0] == 2)
    if enemy_forts_hp < last_enemy_forts_hp:
        reward += 1.0  # 攻撃ヒット！

    # 3. ユニットのレベルアップ（将来への投資）
    # 強い個体を作ることを評価する
    my_levels = sum(s[3] for s in state if s[0] == 1)
    last_my_levels = sum(s[3] for s in last_state if s[0] == 1)
    if my_levels > last_my_levels:
        reward += 0.5  # 育成成功

    # 4. ステップペナルティ（効率性・速攻の促進）
    # 1ステップごとに微小なマイナス。これにより「最短ルートで勝つ」ようになる
    reward -= 0.01

    # 5. 最終結果
    if done:
        if my_forts > 6:
            reward += 50.0  # 完全勝利ボーナス
        else:
            reward -= 50.0  # 敗北ペナルティ
        
    return reward