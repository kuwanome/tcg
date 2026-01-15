import math
from tcg.controller import Controller
from tcg.config import fortress_limit

class Gemini1Player(Controller):
    """
    ParallelExpander (並列進行・自律分散型)
    
    戦略:
    全体での同期（待ち合わせ）を廃止し、各拠点が個別に判断して動きます。
    
    [上側ルート定義]
    1 -> (0, 2)
    0 -> 3 -> 4
    2 -> 5 -> 4
    
    [下側ルート定義]
    10 -> (9, 11)
    9 -> 6 -> 7
    11 -> 8 -> 7
    
    行動ロジック:
    1. 自分がLv5未満 -> ひたすら投資 (他は見ない)
    2. 自分がLv5到達 -> 指定された「次のターゲット」を見る
       A. ターゲットが敵/中立 -> 攻撃
       B. ターゲットが味方(Lv5未満) -> 育成支援 (兵を送る)
       C. ターゲットが味方(Lv5完了) -> 前線送り (兵を託す)
    """

    def team_name(self) -> str:
        return "ParallelExpander"

    def update(self, info) -> tuple[int, int, int]:
        team_id, state, moving_pawns, spawning_pawns, done = info
        
        my_forts = [i for i in range(12) if state[i][0] == 1]
        if not my_forts: return 0, 0, 0

        # --- 1. 攻略ルートの定義 (ターゲットツリー) ---
        # key: 現在地, value: [次の攻略目標リスト]
        # これにより「次はどこへ兵を流すべきか」を定義します
        
        target_map = {}

        # 自分の陣地に合わせてマップを切り替え
        is_top = (1 in my_forts)
        is_bottom = (10 in my_forts)
        
        if is_bottom and not is_top:
            # 下側スタート (10 -> 9,11 -> 6,8 -> 7)
            target_map = {
                10: [9, 11],
                9:  [6],
                11: [8],
                6:  [7],
                8:  [7],
                7:  [] # ゴール
            }
        else:
            # 上側スタート (1 -> 0,2 -> 3,5 -> 4)
            # または両方持っている場合のデフォルト
            target_map = {
                1: [0, 2],
                0: [3],
                2: [5],
                3: [4],
                5: [4],
                4: [] # ゴール
            }

        # --- 2. 各拠点の自律行動 ---
        best_action = (0, 0, 0)
        best_score = -9999

        for my_f in my_forts:
            my_level = state[my_f][2]
            my_pawns = state[my_f][3]
            my_limit = fortress_limit[my_level]
            upgrade_cost = my_limit // 2
            is_upgrading = state[my_f][4] > 0
            
            # ------------------------------------------------
            # Rule 1: 自分が未熟なら、何をおいても自己研鑽
            # ------------------------------------------------
            if my_level < 5:
                if not is_upgrading and my_pawns >= upgrade_cost:
                    return (2, my_f, 0) # 即アップグレード
                
                # 兵が溢れそう(95%)でなければ、貯金のために何もしない
                if my_pawns < my_limit * 0.95:
                    continue
            
            # ------------------------------------------------
            # Rule 2: 自分がLv5 (or 溢れそう) なら、次へ進む
            # ------------------------------------------------
            sending_pawns = my_pawns // 2
            if sending_pawns < 5: continue # 少数は動かさない

            # 自分の「次の目標」を取得。定義がない場所(ゴール地点など)は全隣接を対象にする
            targets = target_map.get(my_f, [])
            if not targets:
                # 定義外（ゴール到達後など）は、敵がいる隣接すべてをターゲットに
                neighbors = state[my_f][5]
                targets = [n for n in neighbors if state[n][0] != 1]
            
            # ターゲットの中からベストな行動を探す
            for target in targets:
                t_info = state[target]
                t_team = t_info[0]
                t_level = t_info[2]
                t_pawns = t_info[3]
                
                score = -9999

                # A. 敵・中立への攻撃
                if t_team != 1:
                    # 確実に勝てるなら攻める
                    if sending_pawns > t_pawns + 2:
                        score = 10000
                    # ターゲットがボスの場合は、勝てなくても特攻して削る
                    elif (is_bottom and target == 7) or (not is_bottom and target == 4):
                         score = 5000
                
                # B. 味方への輸送 (バケツリレー)
                else: 
                    # 相手がLv5未満 -> 育成支援
                    if t_level < 5:
                        score = 3000 # 攻撃の次に優先
                    
                    # 相手もLv5 -> さらに先へ兵を送るための「送り込み」
                    # 自分の兵が半分以上あるなら、前線(ターゲット)へ押し出す
                    elif my_pawns > my_limit * 0.5:
                        score = 1000

                if score > best_score:
                    best_score = score
                    best_action = (1, my_f, target)
        
        return best_action