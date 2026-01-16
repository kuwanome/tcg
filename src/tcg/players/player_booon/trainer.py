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


# --- Trainerクラスの改良 ---
class Trainer:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        self.batch_size = 512
        self.gamma = 0.98  # 少し長期的な利益を重視
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = ReplayMemory(100000) # メモリを増やして過去の良質な経験を保持
        self.train_count = 0 
        self.tau = 0.005 # ★追加：ソフトアップデート用の係数

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

        # 現在のQ値
        q_values = self.model(state_batch).gather(1, action_batch)
        
        # 目標Q値の計算（Double DQN的な要素を少し含める）
        with torch.no_grad():
            next_q = self.target_model(next_state_batch).max(1)[0]
            expected_q = reward_batch + (self.gamma * next_q * (1 - done_batch))

        # 損失計算（SmoothL1は外れ値に強いので継続）
        loss = F.smooth_l1_loss(q_values, expected_q.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング（安定化に必須）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # ★追加：ソフトアップデート（少しずつ目標モデルを更新する）
        self.soft_update_target_model()

    def soft_update_target_model(self):
        """目標モデルを少しずつ更新し、学習の激しい変動を抑える"""
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update_target_model(self):
        """(互換性維持のため残す) 完全同期"""
        self.target_model.load_state_dict(self.model.state_dict())

# --- 報酬計算の改良 ---
def calculate_reward(current_info, last_info, current_step=0):
    team, state, moving_pawns, _, done = current_info
    my_forts = [s for s in state if s[0] == team]
    enemy_forts = [s for s in state if s[0] != team and s[0] != 0]
    
    # 1. 基本報酬（桁を落としてスケーリングを適正化：1.0 = 標準的な「良い」状態）
    reward = len(my_forts) * 0.1 

    # レベル（内政）への加点（Gemini勢に対抗するために必須）
    my_levels = sum([s[2] for s in my_forts])
    reward += my_levels * 0.05

    last_my_count = len([s for s in last_info[1] if s[0] == team])
    diff = len(my_forts) - last_my_count

    if diff > 0:
        # 拠点を奪取した（非常に高い評価だが、桁は抑える）
        # ステップが進むほど奪取の価値を高める（逆転を評価するため）
        step_multiplier = 1.0 + (current_step / 50000)
        reward += 10.0 * step_multiplier
        
    elif diff < 0:
        # 拠点を喪失した（強いペナルティ）
        reward -= 15.0

    # 行動の促進（膠着状態の回避）
    if moving_pawns:
        reward += 0.2 
    else:
        # 資源の死蔵（兵が溢れているのに動かない）を罰する
        if any(s[3] > s[2] * 20 + 20 and s[0] == team for s in state):
            reward -= 0.1

    # 決着報酬（報酬の最大値を100〜200程度に抑える。5万は大きすぎて学習が壊れる原因）
    if done:
        if not enemy_forts:
            reward += 100.0 # 完全勝利
        elif len(my_forts) > len(enemy_forts):
            reward += 50.0  # 判定勝ち
        else:
            reward -= 50.0  # 敗北

    return reward