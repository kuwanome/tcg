import os
import sys

# =====================================================
# 【最優先】まず最初に「探し場所」をPythonに教える
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# player_booon -> players -> tcg -> src (3つ上が src フォルダ)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir) # 検索順位を1位にする

# claude_player.py がある players フォルダも追加
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# =====================================================
# その後にインポートを行う（これで ModuleNotFoundError を防ぎます）
# =====================================================
import torch
import random
from collections import deque
from tcg.game import Game
from model import DuelingQNetwork
from strategy import Strategy
from player import booon
from trainer import Trainer
from tcg.players.sample_random import RandomPlayer
from claude_player import ClaudePlayer

SAVE_PATH = r"C:\github\tcg\src\tcg\players\player_booon\latest.pth"

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = 63 
    action_dim = 48
    
    model = DuelingQNetwork(input_dim, action_dim).to(device)
    target_model = DuelingQNetwork(input_dim, action_dim).to(device)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    if os.path.exists(SAVE_PATH):
        print(f"--- 脳の移植/ロードを開始します: {SAVE_PATH} ---")
        try:
            old_sd = torch.load(SAVE_PATH, map_location=device, weights_only=True)
            new_sd = model.state_dict()
            for name, param in old_sd.items():
                if name in new_sd:
                    if param.shape == new_sd[name].shape:
                        new_sd[name].copy_(param)
                    elif "feature.0.weight" in name:
                        print(f">>> 拠点知識(60列)を移植中...")
                        new_sd[name][:, :60].copy_(param)
            model.load_state_dict(new_sd)
            print(">>> ロード成功！")
        except Exception as e:
            print(f">>> ロード失敗: {e}")
    
    target_model.load_state_dict(model.state_dict())
    strategy = Strategy(model)
    trainer = Trainer(model, target_model)
    agent = booon(model, strategy, mode="train")
    agent.trainer = trainer

    current_epsilon = 0.4
    win_history = deque(maxlen=100)
    win_count = 0
    total_games = 0

    print("--- 訓練開始：混合対戦モード (Random 70% : Claude 30%) ---")

    for ep in range(1, 1001):
        agent.epsilon = current_epsilon
        
        # 対戦相手の選択
        if random.random() < 0.7:
            enemy = RandomPlayer()
            enemy_label = "Random"
        else:
            enemy = ClaudePlayer()
            enemy_label = "Claude "
        
        # 描画なし(False)で高速化
        game = Game(agent, enemy, window=False)
        winner = game.run()

        if winner is None:
            blue_bases = sum(1 for s in game.state if s[0] == 1)
            red_bases = sum(1 for s in game.state if s[0] == 2)
            winner = 1 if blue_bases > red_bases else 2
        
        total_games += 1
        is_win = (winner == 1)
        win_history.append(1 if is_win else 0)
        if is_win: win_count += 1

        current_win_rate = (sum(win_history) / len(win_history)) * 100
        print(f"Ep {ep:4d} | vs {enemy_label} | Win: {str(is_win):5s} | Rate: {current_win_rate:5.1f}% | Eps: {current_epsilon:.4f}")
        
        current_epsilon = max(0.1, current_epsilon * 0.998)
        
        if ep % 10 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            target_model.load_state_dict(model.state_dict())
            print(f">>> モデル保存完了 (累計勝利: {win_count}/{total_games})")

if __name__ == "__main__":
    run_training()