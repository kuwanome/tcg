import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from tcg.controller import Controller
from .model import DuelingQNetwork

# --- 追加: GPUデバイスの定義 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class booon(Controller):
    def __init__(self, mode="train"):
        super().__init__()
        self.state_size = 60
        self.action_size = 48
        self.mode = mode
        
        # --- 修正: モデルをGPU(device)へ転送 ---
        self.model = DuelingQNetwork(self.state_size, self.action_size).to(device)
        self.target_model = DuelingQNetwork(self.state_size, self.action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = deque(maxlen=20000)
        
        self.last_state = None
        self.last_action = None
        self.epsilon = 1.0 if mode == "train" else 0.05
        self.total_steps = 0

    # --- 重要: ここを修正 ---
    def _get_state_vector(self, state):
        """リスト形式の状態をGPU上のテンソルに変換"""
        res = []
        for s in state:
            team = 1.0 if s[0] == 1 else (-1.0 if s[0] == 2 else 0.0)
            # [チーム, 種類, レベル, 兵士数, クールダウン]
            res.extend([team, s[1], s[2]/5.0, min(s[3]/50.0, 1.0), s[4]/100.0])
        
        # テンソルを作成し、.to(device) でGPUへ送る
        return torch.FloatTensor(res).unsqueeze(0).to(device)

    def update(self, info):
        team, state, pawn, SpawnPoint, done = info
        # GPUに乗った状態ベクトルを取得
        current_state_vector = self._get_state_vector(state)

        # --- 学習フェーズ ---
        if self.mode == "train" and self.last_state is not None:
            reward = self._calculate_reward(info)
            # メモリに保存（deviceに乗ったまま保存されます）
            self.memory.append((self.last_state, self.last_action, reward, current_state_vector, done))
            self._optimize_model()

        # --- 行動選択フェーズ ---
        action_idx = self._select_action(current_state_vector)
        
        self.last_state = current_state_vector
        self.last_action = action_idx
        self.last_info = info 
        
        self.total_steps += 1
        if done:
            self.last_state = None

        return self._idx_to_command(action_idx, state)

    def _select_action(self, state_tensor):
        """ε-greedy法による行動選択"""
        if self.mode == "train" and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            # state_tensorは既にGPUにあるのでそのまま入力可能
            action_values = self.model(state_tensor)
            return torch.argmax(action_values).item()

    def _calculate_reward(self, info):
        # ... (以前の報酬ロジックと同じ)
        team, state, pawn, SpawnPoint, done = info
        _, last_state, _, _, _ = self.last_info
        reward = 0.0
        my_forts = sum(1 for s in state if s[0] == 1)
        last_my_forts = sum(1 for s in last_state if s[0] == 1)
        if my_forts > last_my_forts: reward += 1.0
        if my_forts < last_my_forts: reward -= 1.0
        if done:
            reward += 10.0 if my_forts > 6 else -10.0
        return reward

    def _optimize_model(self):
        """GPU上でのバッチ学習"""
        if len(self.memory) < 64: return
        
        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states, next_statesは既にGPU上にあるので、catするだけでOK
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        
        # 他のデータもGPUに送る
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        # 現在のQ値
        current_q = self.model(states).gather(1, actions)
        
        # 次の状態の最大Q値
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (0.99 * max_next_q * (1 - dones))

        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _idx_to_command(self, idx, state):
        # ... (以前の48通りの変換ロジック)
        pass # 実装済みであればそのまま記述