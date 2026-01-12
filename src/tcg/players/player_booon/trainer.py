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
    
    # 1. 基本報酬（維持報酬を少し下げて、動きを促す）
    reward = len(my_forts) * 2.0 

    last_my_count = len([s for s in last_info[1] if s[0] == team])
    diff = len(my_forts) - last_my_count

    if diff > 0:
        # --- ここがポイント：早期制圧ボーナス ---
        # ステップ 0 で最大 5000点、10000ステップで 0点になるような減衰ボーナス
        early_bonus = max(0, 5000.0 - step_count * 0.5) 
        
        # 敵の拠点がまだ多い（序盤）ほど、さらに価値を高める
        capture_bonus = 2000.0 + early_bonus + (len(enemy_forts) * 200.0)
        reward += capture_bonus
        
    elif diff < 0:
        # 拠点を取られた時のペナルティ（ここも重くして防御意識を持たせる）
        reward -= 3000.0

    # 【重要】「攻め続けている」状態を高く評価する
    # 膠着を防ぐため、移動中の兵がいることへの加点を強化
    if moving_pawns:
        reward += 20.0 
    else:
        # 自分の拠点のどこかに兵が50人以上溜まっているのに動いていないならマイナス
        # これで「溜め込みすぎ」を防ぐ
        if any(s[3] > 50 and s[0] == team for s in state):
            reward -= 10.0

    # 決着報酬（殲滅こそが至高）
    if done:
        if not enemy_forts:
            reward += 50000.0 # さらにアップ
        elif len(my_forts) > len(enemy_forts):
            reward += 10000.0
        else:
            reward -= 10000.0

    return reward