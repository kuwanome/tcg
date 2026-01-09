import torch
import os
from tcg.game import Game
from tcg.players.sample_random import RandomPlayer
from .model import DuelingQNetwork
from .strategy import Strategy
from .player import booon
from .trainer import Trainer

# 保存先の設定
SAVE_PATH = "src/tcg/players/player_booon/latest.pth"

def run_training():
    # --- 1. デバイスの設定 (GPUの準備) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 各コンポーネントのインスタンス化 (GPUへ送る) ---
    model = DuelingQNetwork(60, 48).to(device)
    target_model = DuelingQNetwork(60, 48).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # 思考ロジック、肉体(Player)、監督(Trainer)を紐付け
    strategy = Strategy(model, 48)
    agent = booon(model, strategy)
    trainer = Trainer(model, target_model)
    
    # 学習開始時の設定
    agent.epsilon = 1.0 
    
    print("--- 訓練プロセスを開始します ---")

    for ep in range(1, 1001):
        # --- 3. ゲーム環境の構築 ---
        # 学習を高速化するため window=False
        game = Game(agent, RandomPlayer(), window=False)
        
        # --- 4. 実行と学習 ---
        # Player(agent)のupdateメソッド内で、
        # trainer.train_step()が呼ばれるように設計されている前提です。
        game.run()
        
        # --- 5. エピソード終了後のメンテナンス ---
        
        # Epsilon(探索率)を徐々に下げる
        agent.epsilon = max(0.1, agent.epsilon * 0.995)
        
        # 定期的なターゲットネットワークの更新と、モデルの保存
        if ep % 10 == 0:
            # 安定性のためにターゲットを同期
            target_model.load_state_dict(model.state_dict())
            
            # 脳の重みを保存 (GPUからCPU形式に直さずそのままsaveしてOK)
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            
            # 定期的なバックアップ（番号付き）
            if ep % 100 == 0:
                backup_path = SAVE_PATH.replace("latest.pth", f"brain_ep{ep}.pth")
                torch.save(model.state_dict(), backup_path)
            
            print(f"Episode {ep:4d}: Epsilon={agent.epsilon:.3f} - Model Saved.")

if __name__ == "__main__":
    run_training()