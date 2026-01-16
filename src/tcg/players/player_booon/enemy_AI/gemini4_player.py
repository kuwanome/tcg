"""
Google AI Studio Player
ClaudePlayerの弱点（慎重すぎる攻撃判断、資源の自然減少）を突き、
高速な展開と徹底的な資源管理で圧倒するAI。
"""
from tcg.controller import Controller
from tcg.config import fortress_cool, fortress_limit

# クラス名を main.py の要求に合わせて 'NewPlayer' に変更しました
class Gemini4Player(Controller):
    """
    対ClaudePlayer特化型戦略AI
    """
    def __init__(self) -> None:
        super().__init__()
        self.step = 0

    def team_name(self) -> str:
        # ゲーム画面に表示される名前
        return "GoogleAIStudio"

    def update(self, info) -> tuple[int, int, int]:
        """
        毎ステップ実行されるメインロジック
        """
        self.step += 1
        my_team, state, moving_pawns, spawning_pawns, done = info

        # === 1. 情報整理 ===
        my_forts = []
        enemy_forts = []
        neutral_forts = []
        
        # 脅威度マップの作成（敵からの攻撃を受けているか）
        under_attack_counts = {i: 0 for i in range(12)}
        for p in moving_pawns:
            p_team, _, _, to_idx, _ = p
            if p_team != my_team:
                under_attack_counts[to_idx] += 1

        for i in range(12):
            team = state[i][0]
            if team == my_team:
                my_forts.append(i)
            elif team == 0:
                neutral_forts.append(i)
            else:
                enemy_forts.append(i)

        actions = [] # (score, command, subject, target)

        # === 2. 各拠点ごとの行動評価 ===
        for my_idx in my_forts:
            # 拠点情報の展開
            _, kind, level, soldiers, upgrade_timer, neighbors = state[my_idx]
            max_soldiers = fortress_limit[level]
            
            # --- 戦略A: オーバーフロー回避（最優先） ---
            # マニュアルP11: 上限以上だと減少する -> これを絶対に防ぐ
            is_overflowing = soldiers >= max_soldiers * 0.95
            if is_overflowing:
                # アップグレード可能なら即実行
                cost = max_soldiers // 2
                if upgrade_timer == -1 and level < 5 and soldiers >= cost:
                    # 溢れるくらいなら即投資
                    return 2, my_idx, 0 
                
                # アップグレードできないなら、一番安全または敵に近い味方に兵を逃がす
                # とにかく兵を減らして無駄にしない
                best_escape = neighbors[0]
                for n in neighbors:
                    if state[n][0] != my_team: # 敵なら攻撃になるので良し
                        best_escape = n
                        break
                # コマンド1: 移動
                return 1, my_idx, best_escape

            # --- 戦略B: 確実な攻撃 (Sniping) ---
            # 自分の兵の半分を送る
            attack_power = soldiers // 2
            
            for target_idx in neighbors:
                t_team, t_kind, t_level, t_soldiers, _, _ = state[target_idx]
                
                if t_team != my_team:
                    # 敵または中立
                    
                    # 勝利判定の計算
                    # 相手の生産速度
                    prod_rate = fortress_cool[t_kind][t_level]
                    # 到着までの推定時間 (隣接は近いので固定値または少なめに見積もる)
                    travel_time = 60 # おおよその移動フレーム
                    
                    future_enemy_soldiers = t_soldiers
                    if t_team != 0 and prod_rate > 0: # 中立は増えない
                        future_enemy_soldiers += travel_time / prod_rate
                    
                    # 攻撃成功条件: 攻撃力 > (敵兵数 + 増援) * マージン
                    # ClaudePlayerは1.2倍だが、こちらは1.05倍程度で攻める（速度優先）
                    margin = 1.05
                    # 四角い砦(kind 1)からの攻撃は強いのでボーナス
                    if kind == 1:
                        attack_power_adjusted = attack_power * 1.2 # 推定
                    else:
                        attack_power_adjusted = attack_power

                    if attack_power_adjusted > future_enemy_soldiers * margin:
                        # 確実に落とせるなら高スコア
                        # 中立なら取りやすいのでさらに優先
                        score = 1000
                        if t_team == 0:
                            score += 500 # 初動の展開速度重視
                        if t_kind == 1:
                            score += 300 # 四角い砦は価値が高い
                        
                        actions.append((score, 1, my_idx, target_idx))

            # --- 戦略C: 経済成長 (Upgrade) ---
            # 安全な後方基地は積極的にLv5を目指す
            enemy_neighbor_count = sum(1 for n in neighbors if state[n][0] != my_team)
            is_safe = (enemy_neighbor_count == 0)
            
            cost = max_soldiers // 2
            if upgrade_timer == -1 and level < 5 and soldiers >= cost:
                score = 0
                if is_safe:
                    # 安全なら即座にレベル上げ（兵站基地化）
                    score = 200 + (level * 50)
                elif soldiers >= max_soldiers * 0.8:
                    # 前線でも溢れそうなら上げる
                    score = 150
                
                if score > 0:
                    actions.append((score, 2, my_idx, 0))

            # --- 戦略D: 兵站輸送 (Reinforce) ---
            # 敵に接していない砦から、敵に接している砦へ送る
            if is_safe and soldiers > max_soldiers * 0.4:
                # 送り先を探す
                best_target = -1
                max_danger = -1
                
                for n in neighbors:
                    if state[n][0] == my_team:
                        # 隣の砦の敵隣接数を見る
                        n_neighbors = state[n][5]
                        danger = sum(1 for nn in n_neighbors if state[nn][0] != my_team)
                        if danger > max_danger:
                            max_danger = danger
                            best_target = n
                
                if best_target != -1 and max_danger > 0:
                    # 前線への輸送
                    actions.append((50, 1, my_idx, best_target))

        # === 3. 最適な行動の選択 ===
        if actions:
            # スコアが高い順にソート
            actions.sort(key=lambda x: x[0], reverse=True)
            _, cmd, subj, tgt = actions[0]
            return cmd, subj, tgt

        # 何もしない
        return 0, 0, 0