import torch
import os
import sys
from tcg.game import Game
from collections import deque
from model import DuelingQNetwork
from strategy import Strategy
from player import booon
from tcg.players.sample_random import RandomPlayer
from trainer import Trainer
from tcg.players.sample_random import RandomPlayer

# 自分のフォルダの一つ上のフォルダを検索パスに追加する
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from claude_player import ClaudePlayer  # 追加（ファイル名が claude_player.py の場合）

SAVE_PATH = r"C:\github\tcg\src\tcg\players\player_booon\latest.pth"

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = 63  # 63次元へ進化
    action_dim = 48
    
    model = DuelingQNetwork(input_dim, action_dim).to(device)
    target_model = DuelingQNetwork(input_dim, action_dim).to(device)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    if os.path.exists(SAVE_PATH):
        print(f"--- 脳の移植プロセスを開始します: {SAVE_PATH} ---")
        try:
            # 過去の重み（60次元）をロード
            old_sd = torch.load(SAVE_PATH, map_location=device, weights_only=True)
            new_sd = model.state_dict()

            for name, param in old_sd.items():
                if name in new_sd:
                    if param.shape == new_sd[name].shape:
                        # 他の層は形が同じなのでそのままコピー
                        new_sd[name].copy_(param)
                    elif "feature.0.weight" in name:
                        # 最初の層（60 vs 63）だけ特殊コピー
                        print(f">>> 拠点知識(60列)を新しい脳(63列)の基盤に移植中...")
                        # 0〜59列目までに過去の知識をコピー
                        new_sd[name][:, :60].copy_(param)
                        # 残りの3列（60,61,62）は新しい感覚として初期化
                        print(">>> 自分の位置・レベルを感じる新しい神経を接続しました。")
                else:
                    print(f"注意: {name} は新しいモデルに存在しません。")

            model.load_state_dict(new_sd)
            print(">>> 移植成功！過去のエピソードを継承して再始動します。")
        except Exception as e:
            print(f">>> 移植失敗: {e}\n新しいモデルとして開始します。")
    
    target_model.load_state_dict(model.state_dict())
    
    strategy = Strategy(model)
    trainer = Trainer(model, target_model)
    agent = booon(model, strategy, mode="train")
    agent.trainer = trainer
    
    current_epsilon = 0.5 # 知識があるので0.5からスタート
        
    # 勝敗を記録するための箱（最新100試合分）
    win_history = deque(maxlen=100)
    win_count = 0
    total_games = 0

    print("--- 訓練開始：進化した booon vs Random ---")

    for ep in range(1, 1001):
        agent.epsilon = current_epsilon
        
        # 相手を RandomPlayer から ClaudePlayer に変更
        # 相手チーム(Red)として設定
        enemy = ClaudePlayer() 
        game = Game(agent, enemy, window=False) # 高速化のため False
        
        winner = game.run()

        if winner is None:
            # 拠点の数を数えて、多い方を勝者とする
            blue_bases = sum(1 for s in game.state if s[0] == 1)
            red_bases = sum(1 for s in game.state if s[0] == 2)
            if blue_bases > red_bases:
                winner = 1
            elif red_bases > blue_bases:
                winner = 2
        
        total_games += 1
        if winner == 1:
            win_history.append(1) # booonの勝利
            win_count += 1
        else:
            win_history.append(0) # booonの敗北

        # 直近の勝率を計算
        current_win_rate = (sum(win_history) / len(win_history)) * 100
        
        # 1エピソードごとに状況を表示
        print(f"Episode {ep:4d} | Winner: Team {winner} | "
              f"直近勝率: {current_win_rate:5.1f}% | Epsilon: {current_epsilon:.4f}")
        
        # 学習とセーブのロジック
        current_epsilon = max(0.1, current_epsilon * 0.995)
        
        if ep % 10 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            target_model.load_state_dict(model.state_dict())
            print(f">>> モデル保存完了 (累計勝利数: {win_count}/{total_games})")

if __name__ == "__main__":
    run_training()