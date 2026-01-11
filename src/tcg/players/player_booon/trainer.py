import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_get(data, idx):
    if data is None: return 0.0
    val = data[idx]
    return float(val[0] if isinstance(val, list) else val)

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
        self.batch_size = 512
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = ReplayMemory(50000)
        self.train_count = 0 

    def train_step(self):
        self.train_count += 1
        if self.train_count % 5 != 0: return 

        if len(self.memory) < self.batch_size: return
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(state_batch).to(device)
        action_batch = torch.tensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.tensor(reward_batch).float().to(device)
        next_state_batch = torch.cat(next_state_batch).to(device)
        done_batch = torch.tensor(done_batch).float().to(device)

        q_values = self.model(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q = self.target_model(next_state_batch).max(1)[0]
            expected_q = reward_batch + (self.gamma * next_q * (1 - done_batch))

        loss = F.smooth_l1_loss(q_values, expected_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def update_target_model(self):
        """定期的に呼び出して、学習中のモデルを目標モデルにコピーする"""
        self.target_model.load_state_dict(self.model.state_dict())

def calculate_reward(current_info, last_info):
    # 1. まず情報を解体して変数を定義する (順序を修正)
    team, state, moving_pawns, _, done = current_info
    _, last_state, _, _, _ = last_info
    
    my_forts = [s for s in state if s[0] == team]
    last_my_forts = [s for s in last_state if s[0] == team]
    
    # 2. 基本報酬
    reward = -0.5  # 時間ペナルティ
    
    # 3. 危機判定（拠点が1つ以下ならマイナス）
    if len(my_forts) <= 1:
        reward -= 0.1

    # 4. 領土の増減
    diff = len(my_forts) - len(last_my_forts)
    if diff > 0: 
        reward += 250.0 
    elif diff < 0: 
        reward -= 400.0

    # 5. 移動・攻撃行動の評価
    for p in moving_pawns:
        if p[0] == team:
            target_idx = int(p[3])
            if target_idx < len(state):
                target_fort = state[target_idx]
                if target_fort[0] == team:
                    if target_fort[3] < 20: reward += 2.0 
                else:
                    reward += 1.0 
    
    # 6. 砦の状態管理
    last_state_dict = {s[1]: s for s in last_state}
    for s in state:
        if s[0] == team:
            if s[3] >= 45: reward -= 2.0 
            ls = last_state_dict.get(s[1])
            if ls and s[4] > ls[4]:
                reward += 150.0

    # 7. 勝利判定
    if done:
        my_count = len(my_forts)
        if my_count > len(state) // 2:
            reward += 1500.0 
        elif my_count == 0:
            reward -= 1500.0
            
    return reward