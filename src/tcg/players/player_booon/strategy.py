import torch
import random
from collections import deque

# 砦の兵士上限数定義（グローバルまたはクラス内で定義）
fortress_limit = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}

class Strategy:
    def __init__(self, model, action_dim=109):
        self.model = model
        self.action_dim = action_dim
        # 隣接リスト
        self.neighbors = {
            0: [1, 3, 4], 1: [0, 2, 4], 2: [1, 4, 5],
            3: [0, 4, 6, 7], 4: [0, 1, 2, 3, 5, 6, 7, 8], 5: [2, 4, 7, 8],
            6: [3, 4, 7, 9], 7: [3, 4, 5, 6, 8, 9, 10, 11], 8: [4, 5, 7, 11],
            9: [6, 7, 10], 10: [7, 9, 11], 11: [7, 8, 10]
        }

    # --- Gemini5Player から移植した脳みそ (距離計算) ---
    def get_distance_to_enemy(self, state_info, my_team):
        """BFSで敵（および中立）までの距離マップを作成"""
        distances = {i: 999 for i in range(len(state_info))}
        queue = deque()
        
        enemy_id = 1 if my_team == 2 else 2
        
        # ターゲット設定：敵拠点と、まだ取れていない中立拠点
        # これらを「距離0（目的地）」として、そこからの距離を測る
        targets = []
        for i, info in enumerate(state_info):
            owner = info[0]
            if owner == enemy_id:
                targets.append(i)
            elif owner == 0:
                targets.append(i)

        for t in targets:
            distances[t] = 0
            queue.append(t)

        while queue:
            curr = queue.popleft()
            if curr not in self.neighbors: continue
            
            for n in self.neighbors[curr]:
                if distances[n] > distances[curr] + 1:
                    distances[n] = distances[curr] + 1
                    queue.append(n)
        return distances

    # --- Gemini5Player から移植した脳みそ (未来予測) ---
    def predict_future_balance(self, state_info, moving_pawns, my_team):
        """未来予測: 着弾後の実質兵数を計算"""
        enemy_id = 1 if my_team == 2 else 2
        future_balance = {}

        # 1. 現在の兵士数
        for i, info in enumerate(state_info):
            owner, _, _, troops, _, _ = info
            if owner == my_team:
                future_balance[i] = troops
            elif owner == enemy_id:
                future_balance[i] = -troops
            else:
                # 中立は壁としてマイナス計上（確実に落とすため少し厳しめに +2）
                future_balance[i] = -(troops + 2)

        # 2. 移動中の兵士を加算
        if moving_pawns:
            for p in moving_pawns:
                # pの構造: [team, kind, src, dst, count, ...] 
                # ※tcgの仕様に合わせてインデックス調整してください
                if len(p) < 4: continue
                p_team, p_kind, _, p_to = p[0], p[1], p[2], p[3]
                
                # 攻撃力補正（四角い砦=kind1は強い）
                power = 1.2 if p_kind == 1 else 0.8 # 簡易計算

                if p_to in future_balance:
                    if p_team == my_team:
                        future_balance[p_to] += 1.0 # 味方の援軍
                    elif p_team == enemy_id:
                        future_balance[p_to] -= power # 敵の攻撃

        return future_balance


    # --- メイン行動決定関数 ---
    # 【重要】引数に moving_pawns を追加してください
    def get_action(self, state_vector, epsilon, state_info, moving_pawns, my_team, current_step=0):
        with torch.no_grad():
            q_values = self.model(state_vector).clone()

        # --- 状況分析 ---
        enemy_id = 1 if my_team == 2 else 2
        enemy_base_count = sum(1 for info in state_info if info[0] == enemy_id)
        is_finishing_mode = (enemy_base_count == 1)

        # 未来予測マップ & 距離マップの作成
        future_map = self.predict_future_balance(state_info, moving_pawns, my_team)
        dist_map = self.get_distance_to_enemy(state_info, my_team)

        # 自分の拠点リスト
        my_bases = [i for i, info in enumerate(state_info) if info[0] == my_team]

        for action_idx in range(self.action_dim):
            # ---------------------------
            # 0: 待機アクションのチェック
            # ---------------------------
            if action_idx == 0:
                # 緊急回避: 兵が溢れそうな拠点（90%以上）があるなら、待機禁止（強制排出）
                should_force_action = False
                for b in my_bases:
                    cap = fortress_limit.get(state_info[b][2], 100)
                    if state_info[b][3] >= cap * 0.9:
                        should_force_action = True
                        break
                
                if should_force_action:
                    q_values[0, action_idx] = -1e9
                continue

            # デコード
            cmd = self._decode_action(action_idx, state_info, my_team)
            if cmd == (0, 0, 0):
                q_values[0, action_idx] = -1e9
                continue

            c_type, src, dst = cmd
            
            # ---------------------------
            # Type 2: レベルアップ
            # ---------------------------
            if c_type == 2:
                # 未来予測で「敵に取られそう(マイナス)」な場所ではレベル上げ禁止（資材の無駄）
                if future_map[src] < 0:
                    q_values[0, action_idx] = -1e9
                
                # 前線（距離0）かつ兵が少ないときは、レベル上げより防衛優先
                elif dist_map[src] == 0 and state_info[src][3] < 30:
                     q_values[0, action_idx] = -1e9
                
                pass # それ以外は許可

            # ---------------------------
            # Type 1: 移動・攻撃
            # ---------------------------
            elif c_type == 1:
                src_val = state_info[src][3]
                tgt_owner = state_info[dst][0]
                moving_pawn = src_val // 2
                future_val = future_map[dst] # 未来予測（着弾後の敵兵数）

                # === A. 攻撃 (敵 or 中立) ===
                if tgt_owner != my_team:
                    
                    # (1) 対 中立拠点 (ここを劇的に強化！)
                    if tgt_owner == 0:
                        # 敵AI(Gemini1/4)の強みを取り入れる：
                        # 「未来予測で勝てる(>0)」なら、レベルに関係なく即座に取る！
                        # これにより、開幕の展開速度が敵と同等になります。
                        if future_val + moving_pawn > 0:
                            # ただし、ピコピコ防止のため「最低3体」の制限だけは残す
                            if moving_pawn < 3:
                                q_values[0, action_idx] = -1e9
                                continue
                            pass # 許可！(GOサイン)

                        # 勝てないなら、無駄撃ち禁止
                        else:
                            q_values[0, action_idx] = -1e9
                            continue

                    # (2) 対 敵プレイヤー
                    else: # tgt_owner == enemy_id
                         # ここは慎重に（前回と同じ）
                        if is_finishing_mode: pass
                        elif (future_val + moving_pawn > 0) or (moving_pawn >= 5): pass
                        else:
                            q_values[0, action_idx] = -1e9
                            continue

                # === B. 輸送 (味方) ===
                else: 
                    # (前回と同じロジック)
                    if future_val < 0: pass # 救援
                    elif dist_map[src] > dist_map[dst]: # 前線輸送
                        tgt_cap = fortress_limit.get(state_info[dst][2], 100)
                        # 混雑緩和：送り先が満員なら送らない
                        if state_info[dst][3] > tgt_cap * 0.9:
                            q_values[0, action_idx] = -1e9
                        elif moving_pawn < 3: 
                            q_values[0, action_idx] = -1e9
                    else:
                        q_values[0, action_idx] = -1e9 # 逆走禁止
                        continue

        # 探索と活用
        if random.random() < epsilon:
            valid_indices = [i for i in range(self.action_dim) if q_values[0, i] > -1e7]
            if valid_indices:
                action_idx = random.choice(valid_indices)
            else:
                action_idx = 0 
        else:
            action_idx = q_values.max(1)[1].item()

        return action_idx, self._decode_action(action_idx, state_info, my_team)

    def _decode_action(self, action_idx, state_info, my_team):
        if action_idx == 0: return (0, 0, 0)
        
        # レベルアップ
        if 1 <= action_idx <= 12:
            subject = action_idx - 1
            if subject < len(state_info) and state_info[subject][0] == my_team:
                return (2, subject, 0)
            return (0, 0, 0)
        
        # 移動
        move_idx = action_idx - 13
        src = move_idx // 8
        dst_relative = move_idx % 8
        
        if src < 12 and src in self.neighbors:
            possible_dsts = self.neighbors[src]
            if dst_relative < len(possible_dsts):
                dst = possible_dsts[dst_relative]
                if state_info[src][0] == my_team:
                    return (1, src, dst)
        return (0, 0, 0)