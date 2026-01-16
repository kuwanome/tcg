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
    
    # 自分の拠点と敵の拠点
    my_forts = [s for s in state if s[0] == team]
    enemy_forts = [s for s in state if s[0] != team and s[0] != 0]
    
    # --- 1. 基本報酬 (維持) ---
    reward = len(my_forts) * 0.1 

    # --- 2. 中央支配ボーナス (New!) ---
    # 4番と7番はマップの要所。ここを制圧していると高く評価
    for fid in [4, 7]:
        if state[fid][0] == team:
            reward += 0.3  # 3つ分の拠点維持に相当する価値

    # --- 3. 内政ボーナス (強化) ---
    # レベル合計値への評価
    my_levels = sum([s[2] for s in my_forts])
    
    # ここを 0.08 から 0.2 くらいに上げると、
    # AIは「領土拡大」よりも「レベル上げ」に快感を覚えるようになります。
    reward += my_levels * 0.3

    # --- 4. 変化への報酬 ---
    last_my_count = len([s for s in last_info[1] if s[0] == team])
    diff = len(my_forts) - last_my_count

    if diff > 0:
        # ★ここを強化★
        # 拠点数が少ないうち（＝序盤）の拡大に対し、莫大なボーナスを与える
        if len(my_forts) <= 2:
            reward += 50.0  # 「初手の中立確保」はゲームを決めるほど偉い！
        elif len(my_forts) <= 4:
            reward += 30.0  # 序盤の展開もすごく偉い
        else:
            reward += 15.0  # 通常の拡大
            
    elif diff < 0:
        reward -= 20.0 # 奪われるのは痛い

    # --- 5. 兵力差ボーナス (New!) ---
    # 敵の兵を減らす（倒す）ことを評価
    my_pawns = sum([s[3] for s in my_forts])
    last_my_pawns = sum([s[3] for s in last_info[1] if s[0] == team])
    
    # 単純な増減だけでなく、移動中の兵も含めた総戦力の推移を見たいが
    # ここでは簡易的に「兵が増えている＝良い内政」として評価
    if my_pawns > last_my_pawns:
        reward += 0.01

    # -----------------------------------------------------------
    # 6. 中央要塞化ボーナス (修正版)
    # -----------------------------------------------------------
    for fid in [4, 7]:
        if state[fid][0] == team:
            troops = state[fid][3]
            
            # 【重要】上限（キャップ）を設ける！
            # 「兵50体までは評価するが、それ以上溜めてもボーナスは増えない」
            # これにより、51体目からは「ここに置いておくより、攻めて領土ボーナス(+15)を稼ごう」となる。
            capped_troops = min(troops, 50) 
            
            # 上限がある分、単価は少し高く(0.03 -> 0.05)して、50体までは全力で溜めさせる
            reward += capped_troops * 0.05
            
    # 行動促進
    if moving_pawns:
        reward += 0.1
    else:
        # 兵が溢れているのに動かないとペナルティ
        if any(s[3] > s[2] * 20 + 20 and s[0] == team for s in state):
            reward -= 0.2

    # --- 6. 決着報酬 ---
    if done:
        if not enemy_forts:
            reward += 100.0 # 完全勝利
        elif len(my_forts) > len(enemy_forts):
            reward += 50.0  # 判定勝ち
        else:
            reward -= 50.0  # 敗北

    return reward