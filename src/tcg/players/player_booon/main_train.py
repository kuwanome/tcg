import torch
import os
from tcg.game import Game
from tcg.players.sample_random import RandomPlayer
from model import DuelingQNetwork
from strategy import Strategy
from player import booon
from trainer import Trainer

SAVE_PATH = "src/tcg/players/player_booon/latest.pth"

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DuelingQNetwork(60, 48).to(device)
    target_model = DuelingQNetwork(60, 48).to(device)
    target_model.load_state_dict(model.state_dict())
    
    strategy = Strategy(model, 48)
    agent = booon(model, strategy)
    trainer = Trainer(model, target_model)
    
    # 重要: agentがtrainerを使えるように紐付け
    agent.trainer = trainer
    
    agent.epsilon = 1.0 
    
    print("--- 訓練プロセスを開始します ---")

    for ep in range(1, 1001):
        # 変更前
         # game = Game(agent, RandomPlayer(), window=False)

        # 変更後：自分(agent) vs 自分(agent) のガチンコ勝負！
        game = Game(agent, agent, window=True)
        game.run()
        
        agent.epsilon = max(0.1, agent.epsilon * 0.995)
        
        if ep % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            
            if ep % 100 == 0:
                backup_path = SAVE_PATH.replace("latest.pth", f"brain_ep{ep}.pth")
                torch.save(model.state_dict(), backup_path)
            
            print(f"Episode {ep:4d}: Epsilon={agent.epsilon:.3f} - Model Saved.")

if __name__ == "__main__":
    run_training()