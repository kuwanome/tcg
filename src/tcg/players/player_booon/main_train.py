import torch
import os
from tcg.game import Game
from model import DuelingQNetwork
from strategy import Strategy
from player import booon
from tcg.players.sample_random import RandomPlayer
from trainer import Trainer

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
        print(f"Loading existing model: {SAVE_PATH}")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    
    target_model.load_state_dict(model.state_dict())

    if os.path.exists(SAVE_PATH):
        print(f"Loading existing model: {SAVE_PATH}")
        # weights_only=True をつけると安全に読み込めます
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
        target_model.load_state_dict(model.state_dict())
    else:
        print("過去のモデルが見つかりませんでした。最初から学習を開始します。")
    
    strategy = Strategy(model)
    trainer = Trainer(model, target_model)
    
    agent = booon(model, strategy, mode="train")
    agent.trainer = trainer
    
    
    current_epsilon = 0.5 

    print("--- 訓練開始：booon vs Random 対戦モード ---")

    for ep in range(1, 1001):
        agent.epsilon = current_epsilon
        
        game = Game(agent, RandomPlayer(), window=True) 
        
        game.run()
        
        current_epsilon = max(0.1, current_epsilon * 0.995)
        
        if ep % 10 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            target_model.load_state_dict(model.state_dict())
            print(f"Episode {ep}: モデル保存完了")

if __name__ == "__main__":
    run_training()