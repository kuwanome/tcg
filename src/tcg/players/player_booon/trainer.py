import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# player.py から呼び出される重要な関数
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

# trainer.py 内の Trainer クラス
class Trainer:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        # 128 から 512 へ大幅アップ（GPUメモリに余裕があるならこれくらいが最適）
        self.batch_size = 512
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        # 記憶容量も少し増やしておきます
        self.memory = ReplayMemory(50000)
        self.train_count = 0  # 学習回数のカウンタ

    def train_step(self):
        self.train_count += 1
        # 5ステップに1回だけ学習することで、処理の「渋滞」を防ぎスピードアップさせます
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

def calculate_reward(current_info, last_info):
    reward = 0.0
    team, state, moving_pawns, _, done = current_info
    _, last_state, _, _, _ = last_info
    
    my_forts = [s for s in state if s[0] == team]
    last_my_forts = [s for s in last_state if s[0] == team]
    
    # ---------------------------------------------------------
    # 1. 領土の増減（一喜一憂させない重み付け）
    # ---------------------------------------------------------
    diff = len(my_forts) - len(last_my_forts)
    if diff > 0: 
        reward += 200.0  # 占領成功
    elif diff < 0: 
        reward -= 400.0  # 喪失は「大失態」として教える（防衛意識の向上）

    # ---------------------------------------------------------
    # 2. 供給行動（後ろから前へ送る動きを直接評価）
    # ---------------------------------------------------------
    # 自分の移動中の兵士をチェック
    for p in moving_pawns:
        if p[0] == team: # 自分の兵
            target_idx = int(p[3])
            target_fort = state[target_idx]
            
            # 命令の目的地が「自分の砦」である場合（供給）
            if target_fort[0] == team:
                # 目的地が「兵数20以下」でピンチなら、送っている最中に加点
                if target_fort[3] < 20:
                    reward += 5.0  # 供給を「善」と教える
    
    # ---------------------------------------------------------
    # 3. 砦の「健康状態」と「レベルアップ」
    # ---------------------------------------------------------
    for i, s in enumerate(state):
        if s[0] == team:
            # 兵数が多すぎる(45人以上)と、生産が止まって「損」だと教える
            if s[3] >= 45:
                reward -= 1.0 
            
            # 逆に兵数が少なすぎる(10人以下)と、常に「不安」だと教える
            if s[3] <= 10:
                reward -= 10.0 # これにより、根元から送らせる動機を作る
            
            # アップグレードへのボーナス（強さの基盤）
            # last_stateと比較してレベルが上がっていれば加点
            for ls in last_state:
                if ls[1] == s[1] and s[2] > ls[2]: # 座標が同じ砦のレベル比較
                    reward += 150.0

    # ---------------------------------------------------------
    # 4. 勝利への執念
    # ---------------------------------------------------------
    if done:
        if len(my_forts) > len(state) // 2:
            reward += 1000.0 # 圧勝
        elif len(my_forts) == 0:
            reward -= 1000.0 # 全滅
            
    return reward