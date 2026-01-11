import os
import sys

# =====================================================
# 【最優先】まず最初に「探し場所」をPythonに教える
# =====================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# =====================================================
# その後にインポートを行う
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
from nemesis_player import NemesisPlayer

SAVE_PATH = r"C:\github\tcg\src\tcg\players\player_booon\latest.pth"

# main_train.py 内の UltimateSafeWrapper クラスを以下に差し替えてください

class UltimateSafeWrapper:
    """エンジンが持つ A_coordinate と完全に整合性を取るラッパー"""
    def __init__(self, player):
        self.player = player
        # A_coordinate で 0 以外（タプル）が定義されている場所だけを許可
        self.engine_routes = {
            0: [1, 3, 4],
            1: [0, 2, 4],
            2: [1, 4, 5],
            3: [0, 4, 6, 7],
            4: [0, 1, 2, 3, 5, 6, 7, 8],
            5: [2, 4, 7, 8],
            6: [3, 4, 7, 9],
            7: [3, 4, 5, 6, 8, 9, 10, 11],
            8: [4, 5, 7, 11],
            9: [6, 7, 10],
            10: [7, 9, 11],
            11: [7, 8, 10]
        }

    def update(self, info):
        try:
            cmd = self.player.update(info)
            if not isinstance(cmd, (list, tuple)) or len(cmd) < 3:
                return (0, 0, 0)

            c, fr, to = cmd[0], cmd[1], cmd[2]

            # 移動命令(1)のとき、エンジン側に座標があるか厳密にチェック
            if c == 1:
                # 送り元が自分の拠点かチェック
                state = info[1]
                team_id = info[0]
                if fr >= len(state) or state[fr][0] != team_id:
                    return (0, 0, 0)

                # 隣接ルートに存在しない、または座標データが 0 なら遮断
                if to not in self.engine_routes.get(fr, []):
                    return (0, 0, 0)
            
            return (c, fr, to)
        except Exception:
            return (0, 0, 0)

    def team_name(self): return self.player.team_name()

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = 63 
    action_dim = 48
    
    model = DuelingQNetwork(input_dim, action_dim).to(device)
    target_model = DuelingQNetwork(input_dim, action_dim).to(device)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # --- ロード処理の改善 ---
    # ロードに成功したかどうかのフラグ
    load_success = False
    
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
            print(">>> ロード成功！ 以前の学習を引き継ぎます。")
            load_success = True
        except Exception as e:
            print(f">>> ロード失敗（構造変化またはファイル破損）: {e}")
            print(">>> 【重要】新しい脳（ランダム初期化）で最初からやり直します。")
    else:
        print(">>> 学習ファイルが見つかりません。ゼロから学習を開始します。")

    target_model.load_state_dict(model.state_dict())
    strategy = Strategy(model)
    trainer = Trainer(model, target_model)
    opponent_model = DuelingQNetwork(input_dim, action_dim).to(device)
    agent = booon(model, strategy, mode="train")
    agent.trainer = trainer

    # --- イプシロンの自動調整 ---
    # ロード成功なら 0.6 から、失敗・リセットなら 1.0 からスタート
    current_epsilon = 0.6 if load_success else 1.0
    
    win_history = deque(maxlen=100)
    win_count = 0
    total_games = 0

    print(f"--- 訓練開始：混合対戦モード (Eps開始値: {current_epsilon}) ---")

    for ep in range(1, 1001):
        # --- 1. 自分の役割 (1:先攻, 2:後攻) ---
        my_team_id = 1 if random.random() < 0.5 else 2
        agent.team = my_team_id
        agent.epsilon = current_epsilon

        # --- 2. 対戦相手(enemy)の決定 ---
        dice = random.random()
        if dice < 0.10:
            enemy = RandomPlayer()
            enemy_label = "Random"
        elif dice < 0.20:
            enemy = ClaudePlayer()
            enemy_label = "Claude "
        elif dice < 0.40:
            enemy = NemesisPlayer()
            enemy_label = "Nemesis"
        else:
            opponent_id = 1 if my_team_id == 2 else 2
            opponent_model.load_state_dict(model.state_dict())
            
            # 相手(enemy)にも最新の neighbors を持った Strategy を渡す
            enemy_strategy = Strategy(opponent_model) 
            enemy = booon(opponent_model, enemy_strategy, mode="test", team=opponent_id)
            enemy.epsilon = 0.05
            enemy_label = "booon2"

        # --- 3. ゲームの初期化 (Wrapperを適用してクラッシュを防ぐ) ---
        # 自分のAIも、相手のAI(Random, Claude, Nemesis, booon2)もすべて包みます
        safe_agent = UltimateSafeWrapper(agent)
        safe_enemy = UltimateSafeWrapper(enemy)

        if my_team_id == 1:
            game = Game(safe_agent, safe_enemy, window=True)
        else:
            game = Game(safe_enemy, safe_agent, window=True)
        
       # --- 4. 実行と判定 ---
        winner = game.run() # エンジンの判定は参考程度に

        # 拠点数のカウント（これを勝敗の「絶対基準」にする）
        final_owners = [s[0] for s in game.state]
        my_bases = final_owners.count(my_team_id)
        enemy_id = 1 if my_team_id == 2 else 2
        enemy_bases = final_owners.count(enemy_id)

        # ★修正ポイント：拠点数で勝敗を上書きする
        is_win = (my_bases > enemy_bases)

        # --- 5. 統計の更新 (表示の前に計算する) ---
        total_games += 1
        win_history.append(1 if is_win else 0)
        if is_win: win_count += 1
        current_win_rate = (sum(win_history) / len(win_history)) * 100

        # --- 6. ログ表示 (先攻/後攻 を Blue/Red に変更) ---
        role_str = "Blue" if my_team_id == 1 else "Red "  # 文字数を合わせるため Red の後に半角スペースを入れています
        result_str = "★WIN★" if is_win else " LOSE "
        
        # ログの出力
        print(f"Ep {ep:4d} | {role_str}(ID:{my_team_id}) vs {enemy_label:7s} | {result_str} | My:{my_bases} vs En:{enemy_bases} | Rate:{current_win_rate:5.1f}% | Eps:{current_epsilon:.4f}")

       
        # --- 7. 後処理 ---
        current_epsilon = max(0.05, current_epsilon * 0.998)
        
        if ep % 10 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            target_model.load_state_dict(model.state_dict())
            print(f">>> 学習成果を蓄積（保存完了）: {SAVE_PATH} (勝利: {win_count}/{total_games})")

if __name__ == "__main__":
    run_training()