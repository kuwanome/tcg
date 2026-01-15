import torch
import os  # パス操作用に追加
import sys
from tcg.controller import Controller
from .model import DuelingQNetwork   # フォルダ内からインポート
from .strategy import Strategy       # フォルダ内からインポート
from .trainer import safe_get        # 必要な関数のみ

try:
    from model import DuelingQNetwork
    from strategy import Strategy
    from trainer import safe_get, calculate_reward
except (ImportError, ValueError):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from model import DuelingQNetwork
    from strategy import Strategy
    from trainer import safe_get, calculate_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class booon(Controller):
    def __init__(self, mode="test", team=1):
        super().__init__()
        
        self.model = DuelingQNetwork(63, 109).to(device)
        self.strategy = Strategy(self.model)
        
        # --- 【追加】ステップ数の初期化 ---
        self.step_count = 0 
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "latest.pth")
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            self.model.eval()
        
        self.mode = mode
        self.team = team
        self.epsilon = 0.05
        
        self.last_state = None
        self.last_action = None
        self.last_info = None

    def team_name(self):
        return "booon"

    def _get_state_vector(self, info):
        # AIが盤面を理解するための情報を整理（63次元）
        team_id, state, pawn, _, _ = info
        res = []
        
        # 1. 拠点情報 (60次元) - 移植した知識と位置を合わせるため先に配置
        for s in state:
            if s[0] == team_id:
                rel_team = 1.0
            elif s[0] == 0:
                rel_team = 0.0
            else:
                rel_team = -1.0
            
            res.extend([
                rel_team, 
                safe_get(s, 1)/100.0, 
                safe_get(s, 2)/100.0, 
                min(safe_get(s, 3)/50.0, 1.0), 
                safe_get(s, 4)/5.0
            ])
            
        # 2. 自分の位置・レベル情報 (3次元) - 新しく追加した神経に対応
        my_unit = next((p for p in pawn if p[0] == team_id), None)
        res.extend([
            safe_get(my_unit, 1)/100.0, # X座標
            safe_get(my_unit, 2)/100.0, # Y座標
            safe_get(my_unit, 3)/5.0    # レベル
        ])
        
        return torch.FloatTensor(res).unsqueeze(0).to(device)

    def update(self, info):
        team_id = info[0]
        self.team = team_id
        
        # --- 【追加】ステップ数をカウントアップ ---
        self.step_count += 1

        current_state_vector = self._get_state_vector(info)
        current_done = info[4]

        # 学習フェーズ (訓練モードの時のみ実行)
        if self.mode == "train" and self.last_state is not None and self.trainer is not None:
            reward = calculate_reward(info, self.last_info)
            self.trainer.memory.push(self.last_state, self.last_action, reward, current_state_vector, current_done)
            for _ in range(3): 
                self.trainer.train_step()

        # --- 【変更】第5引数に self.step_count を渡すように修正 ---
        action_idx, command = self.strategy.get_action(
            current_state_vector, 
            self.epsilon, 
            info[1], 
            team_id, 
            self.step_count  # ここに追加
        )
        
        if self.mode == "train":
            self.last_state = current_state_vector
            self.last_action = action_idx
            self.last_info = info

        # もし試合が終了(done)したら、ステップ数をリセットする処理を入れておくと安心です
        if current_done:
            self.step_count = 0

        return command