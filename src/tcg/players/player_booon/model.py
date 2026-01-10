import torch
import torch.nn as nn  
import torch.nn.functional as F

# デバイスの自動判定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        
        # 共通のベース層
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # 状態価値 V(s) を計算する層（その盤面自体がどれだけ有利か）
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # アドバンテージ A(s, a) を計算する層（各行動がどれだけ良いか）
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # V + (A - mean(A)) で結合
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

# インスタンス化の際にデバイスへ送る
model = DuelingQNetwork(60, 48).to(device)