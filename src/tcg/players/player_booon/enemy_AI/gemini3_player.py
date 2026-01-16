from tcg.controller import Controller
import random

class Gemini3Player(Controller):
    """
    ユーザー戦略実装版AI
    戦略: 後方基地(0,2)をLvMaxまで育て、その後中央(4,7)へ戦力を集中投下する。
    """

    def team_name(self) -> str:
        return "LvMaxRusher"

    def update(self, info) -> tuple[int, int, int]:
        # info: [my_team_id, state, moving_pawns, spawning_pawns, done]
        my_team_id, state, moving_pawns, spawning_pawns, done = info
        
        # 自分の所有する要塞を取得
        my_fortresses = [i for i, f in enumerate(state) if f[0] == my_team_id]
        
        if not my_fortresses:
            return 0, 0, 0

        # =================================================================
        # マップの解釈 (自分の視点に合わせてターゲットを設定)
        # =================================================================
        # ゲームの仕様上、プレイヤー視点は常に「手前が自分」になるよう調整される場合がありますが、
        # ここでは「盤面のY座標」や「接続」を見て動的に後方・中央を判断します。
        
        # 簡易判定: インデックス9,10,11が「下(手前)」、0,1,2が「上(奥)」
        # 自分が9,10,11を持っているなら「下側スタート」、0,1,2なら「上側スタート」と仮定
        is_bottom_start = any(i in my_fortresses for i in [9, 10, 11])
        
        if is_bottom_start:
            # 自分は下側(Blue想定): 後方は9, 11。 目指す中央は7(手前中央), 4(奥中央)
            back_bases = [9, 11]
            primary_center = 7
            secondary_center = 4
        else:
            # 自分は上側(Red想定): 後方は0, 2。 目指す中央は4(手前中央), 7(奥中央)
            back_bases = [0, 2]
            primary_center = 4
            secondary_center = 7

        # =================================================================
        # アクションの決定
        # =================================================================
        
        # 1. まず、自分の全ての要塞についてチェック
        for f_idx in my_fortresses:
            f_state = state[f_idx]
            level = f_state[2]
            pawn_count = f_state[3]
            max_pawn = level * 10
            
            # --- ロジックA: 後方基地(0,2 相当)の場合 ---
            if f_idx in back_bases:
                # 戦略: Lv5になるまではアップグレード最優先
                if level < 5:
                    upgrade_cost = max_pawn / 2
                    # コストが払えるなら即アップグレード
                    if pawn_count >= upgrade_cost:
                        return 2, f_idx, 0
                    else:
                        # コスト不足なら何もしない（兵を溜める＝移動しない）
                        continue
                
                else:
                    # Lv5になったら中央へ兵を送る (兵士がある程度溜まったら)
                    if pawn_count > 10: 
                        # まず手前の中央(primary)が自分のものか確認
                        p_center_owner = state[primary_center][0]
                        
                        # 手前中央がまだ自分のものじゃない、あるいは敵・中立なら攻撃/移動
                        if p_center_owner != my_team_id:
                            return 1, f_idx, primary_center
                        
                        # 手前中央が自分のものなら、奥の中央(secondary)へ送ることを検討
                        # ただし直接つながっていない場合は、一度手前中央へ送るしかない
                        return 1, f_idx, primary_center

            # --- ロジックB: それ以外の基地（中央や前線） ---
            else:
                # 基本動作: 兵があふれそうならアップグレード or 攻撃
                
                # ターゲット決定: 敵または中立の隣接基地を優先
                neighbors = f_state[5]
                enemy_neighbors = [n for n in neighbors if state[n][0] != my_team_id]
                
                # 攻撃可能な敵がいれば攻撃
                if enemy_neighbors and pawn_count > 5:
                    # 兵数が少ない敵を狙う
                    target = min(enemy_neighbors, key=lambda x: state[x][3])
                    # 自分の兵の半分を送っても勝てる、あるいは削れるならGO
                    if pawn_count / 2 > state[target][3] or state[target][0] != 0: # 中立以外は積極攻撃
                         return 1, f_idx, target

                # 敵がいない、または兵が少ない場合
                # アップグレード（あふれ防止）
                if pawn_count >= max_pawn * 0.8 and pawn_count >= max_pawn / 2 and level < 5:
                    return 2, f_idx, 0
                
                # それでもあふれそうなら、適当な味方基地へ逃がす（前線へ送る）
                if pawn_count >= max_pawn * 0.9:
                    # 敵に近い味方基地を探す
                    safe_target = random.choice(neighbors)
                    return 1, f_idx, safe_target

        # 何もすることがない
        return 0, 0, 0