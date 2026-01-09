import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os  # ディレクトリ作成用に追加
from collections import deque
from tcg.game import Game
from .player import booon
from tcg.players.sample_random import RandomPlayer

# --- デバイスの設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ハイパーパラメータ ---
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 20000
TARGET_UPDATE = 10
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1

# 保存先ディレクトリのパス（環境に合わせて調整してください）
SAVE_DIR = "src/tcg/players/player_booon"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ... (ReplayMemoryクラスはそのまま)

def train():
    # 1. 環境とエージェントの準備
    agent = booon(mode="train")
    # agent.model は player.py 側で .to(device) されている前提
    
    optimizer = optim.Adam(agent.model.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Target Network (GPUへ送る)
    target_net = type(agent.model)(agent.state_size, agent.action_size).to(device)
    target_net.load_state_dict(agent.model.state_dict())
    target_net.eval()

    print(f"--- 訓練開始 (Device: {device}) ---")

    for episode in range(1, 1501):
        # ... (相手の選択、Gameの初期化などはそのまま)
        opponent = RandomPlayer() if episode < 500 else booon(mode="eval")
        game = Game(agent, opponent, window=False)
        done = False
        last_state_vector = None
        
        while not done:
            # --- 1ステップの処理 ---
            info = game.get_info() 
            team, state, pawn, SpawnPoint, done = info
            state_vector = agent._get_state_vector(state)

            if last_state_vector is not None:
                reward = calculate_reward(info, last_info)
                memory.push(last_state_vector, last_action, reward, state_vector, done)

            action_idx = agent.select_action(state_vector)
            command = agent._idx_to_command(action_idx, state)
            game.step(command) 
            
            last_state_vector = state_vector
            last_action = action_idx
            last_info = info

            if len(memory) > BATCH_SIZE:
                optimize_model(agent.model, target_net, memory, optimizer)

        # --- エピソード終了後の処理 (ここに差し込み) ---
        
        # 1. Target Networkの更新
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(agent.model.state_dict())
        
        # 2. Epsilonの減衰
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        # 3. モデルの保存 (ここが差し込みポイント)
        if episode % 50 == 0:
            save_path = os.path.join(SAVE_DIR, f"brain_ep{episode}.pth")
            # GPU上の重みを保存する場合も、ロード時の利便性のために
            # state_dict()をそのまま保存するのがPyTorchの標準です
            torch.save(agent.model.state_dict(), save_path)
            # 最新版として上書き保存
            torch.save(agent.model.state_dict(), os.path.join(SAVE_DIR, "latest.pth"))
            
            print(f"Episode {episode}: Epsilon = {agent.epsilon:.3f} | Saved model.")

def optimize_model(policy_net, target_net, memory, optimizer):
    """バッチ学習の実行（各テンソルをGPUへ送るように修正）"""
    batch = memory.sample(BATCH_SIZE)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    # テンソル化してGPU(device)へ転送
    state_batch = torch.cat(state_batch).to(device)
    action_batch = torch.tensor(action_batch).unsqueeze(1).to(device)
    reward_batch = torch.tensor(reward_batch).float().to(device)
    next_state_batch = torch.cat(next_state_batch).to(device)
    done_batch = torch.tensor(done_batch).float().to(device)

    # ... (損失計算と最適化のロジックはそのまま)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ... (calculate_rewardはそのまま)