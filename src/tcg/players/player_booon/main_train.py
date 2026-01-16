import os
import sys
import csv  # ★追加: CSV保存用

# =====================================================
# パス設定の修正版
# =====================================================
# 今実行している main_train.py の場所
current_dir = os.path.dirname(os.path.abspath(__file__)) 

# プロジェクトのルート(C:\github\tcg)を探す
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
src_dir = os.path.join(project_root, "src")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# src フォルダも一応追加
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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

# 敵AI群のインポート
from enemy_AI.claude_player import ClaudePlayer
from enemy_AI.nemesis_player import NemesisPlayer
from enemy_AI.turtle_player import TurtlePlayer
from enemy_AI.sniper_player import SniperPlayer
from enemy_AI.swarm_player import SwarmPlayer
from enemy_AI.killer_player import KillerPlayer
from enemy_AI.gemini1_player import Gemini1Player
from enemy_AI.gemini2_player import Gemini2Player
from enemy_AI.gemini3_player import Gemini3Player
from enemy_AI.gemini4_player import Gemini4Player
from enemy_AI.gemini5_player import Gemini5Player
from enemy_AI.gemini6_player import Gemini6Player
from enemy_AI.gemini7_player import Gemini7Player

SAVE_PATH = r"C:\github\tcg\src\tcg\players\player_booon\latest.pth"
LOG_FILE = os.path.join(os.path.dirname(SAVE_PATH), "training_log.csv") # ★追加: ログ保存先

# main_train.py 内の UltimateSafeWrapper クラス

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
    action_dim = 109  # ★ 48 から 109 に変更
    
    model = DuelingQNetwork(input_dim, action_dim).to(device)
    target_model = DuelingQNetwork(input_dim, action_dim).to(device)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # --- ロード処理の改善 ---
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
    agent = booon(mode="train")
    agent.trainer = trainer

    # --- イプシロンの自動調整 ---
    current_epsilon = 0.5097 if load_success else 1.0
    
    win_history = deque(maxlen=100)
    win_count = 0
    total_games = 0
    

    # --- ★追加: CSVログのヘッダー作成 ---
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            # エピソード, 勝率, イプシロン, 獲得報酬合計, 対戦相手
            writer.writerow(["episode", "win_rate", "epsilon", "total_reward", "enemy"])

    print(f"--- 訓練開始：Phase 2 対Gemini特訓モード (Eps再設定: {current_epsilon}) ---")

    for ep in range(1161, 3001):
        my_team_id = 1 if random.random() < 0.5 else 2
        
        # 既存の agent インスタンスの設定を更新
        agent.team = my_team_id
        agent.mode = "train"
        agent.epsilon = current_epsilon
        
        # ★重要: 1エピソードごとの報酬を計測するためリセット
        # (booonクラス側に total_reward 属性がなくてもエラーにならないよう動的にセット)
        agent.total_reward = 0 

        # --- 2. 対戦相手(enemy)の決定 (Phase 3: バランス型) ---
        dice = random.random()
        
        # 【成功体験ゾーン 40%】(とにかく攻めて勝つ感覚を取り戻す)
        if dice < 0.10:
            enemy = Gemini1Player() 
            enemy_label = "Gemini1 "
        elif dice < 0.20:
            enemy = Gemini2Player()
            enemy_label = "Gemini2Player"
        elif dice < 0.35:
            enemy = Gemini3Player()
            enemy_label = "Gemini3Player"
        elif dice < 0.50:
            enemy = Gemini4Player()
            enemy_label = "Gemini4Player"
        elif dice < 0.65:
            enemy = Gemini5Player()
            enemy_label = "Gemini5Player"

        # 【実戦復帰ゾーン 40%】(Gemini 1, 2 メイン)
        elif dice < 0.75:
            enemy = Gemini6Player()
            enemy_label = "Gemini6Player"
        elif dice < 0.85:
            enemy = Gemini7Player() 
            enemy_label = "Gemini7Player"

        # 【高レベル・同キャラ 10%】
        else:
            # セルフプレイ（最新の自分と対局）
            opponent_id = 1 if my_team_id == 2 else 2
            opponent_model.load_state_dict(model.state_dict())
            enemy_strategy = Strategy(opponent_model) 
            enemy = booon(mode="test", team=opponent_id)
            enemy.model.load_state_dict(opponent_model.state_dict())
            enemy.epsilon = 0.20
            enemy_label = "booon2"

        # --- 3. ゲームの初期化 ---
        safe_agent = UltimateSafeWrapper(agent)
        safe_enemy = UltimateSafeWrapper(enemy)

        if my_team_id == 1:
            game = Game(safe_agent, safe_enemy, window=False)
        else:
            game = Game(safe_enemy, safe_agent, window=False)
        
        # --- 4. 実行と判定 ---
        winner = game.run() 

        # 拠点数のカウント
        final_owners = [s[0] for s in game.state]
        my_bases = final_owners.count(my_team_id)
        enemy_id = 1 if my_team_id == 2 else 2
        enemy_bases = final_owners.count(enemy_id)

        # 拠点数で勝敗判定
        is_win = (my_bases > enemy_bases)

        # --- 5. 統計の更新 ---
        total_games += 1
        win_history.append(1 if is_win else 0)
        if is_win: win_count += 1
        current_win_rate = (sum(win_history) / len(win_history)) * 100
        
        # ★追加: 今回のエピソードで獲得した報酬を取得 (なければ0)
        episode_reward = getattr(agent, "total_reward", 0)

        # --- 6. ログ表示 ---
        role_str = "Blue" if my_team_id == 1 else "Red "
        result_str = "★WIN★" if is_win else " LOSE "
        
        print(f"Ep {ep:4d} | {role_str}(ID:{my_team_id}) vs {enemy_label:7s} | {result_str} | My:{my_bases} vs En:{enemy_bases} | Rate:{current_win_rate:5.1f}% | Rwd:{episode_reward:6.1f} | Eps:{current_epsilon:.4f}")

        # ★追加: CSVへ1行書き込み
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, current_win_rate, current_epsilon, episode_reward, enemy_label])

        # --- 7. 後処理 ---
        current_epsilon = max(0.05, current_epsilon * 0.992)
        
        if ep % 10 == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            target_model.load_state_dict(model.state_dict())
            print(f">>> 学習成果を蓄積（保存完了）: {SAVE_PATH} (勝利: {win_count}/{total_games})")

    print(f"{ep}エピソードに到達しました。最終保存を行って終了します。")
    # Trainerクラスのstaticメソッドではなく、インスタンスのメソッドあるいはtorch.saveを使う形に統一
    torch.save(model.state_dict(), "latest.pth") 
    print("学習が正常に終了しました。お疲れ様でした！")


if __name__ == "__main__":
    run_training()