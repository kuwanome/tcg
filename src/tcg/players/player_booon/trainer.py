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
        self.gamma = 0.95
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
    team, state, moving_pawns, _, done = current_info
    my_forts = [s for s in state if s[0] == team]
    enemy_forts = [s for s in state if s[0] != team and s[0] != 0]
    
    # 拠点の数に基づく維持報酬
    reward = (len(my_forts) - 6) * 1.0 

    # 拠点増減の評価
    last_my_count = len([s for s in last_info[1] if s[0] == team])
    diff = len(my_forts) - last_my_count

    if diff > 0:
        # 新しく奪った拠点を特定
        new_f = next((s for s in state if s[0] == team and not any(ls[1]==s[1] and ls[0]==team for ls in last_info[1])), None)
        # 残存兵1人につき15点のボーナス（効率制圧を評価）
        pawn_bonus = new_f[3] * 15.0 if new_f else 0
        reward += 1000.0 + (12 - len(enemy_forts)) * 100.0 + pawn_bonus
    elif diff < 0:
        reward -= 800.0

    # 攻撃姿勢への評価
    for p in moving_pawns:
        if p[0] == team:
            target_idx = int(p[3])
            if target_idx < len(state) and state[target_idx][0] != team:
                reward += 5.0 # 敵に向かう姿勢を評価

    # 決着報酬
    if done:
        if len(enemy_forts) == 0:
            reward += 20000.0 # 完全制圧
        elif len(my_forts) > len(enemy_forts):
            reward += 10000.0 # 判定勝ち
        else:
            reward -= 10000.0

    return reward