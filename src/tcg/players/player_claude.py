"""
AI Player

戦略的に考えて行動する強いAIプレイヤー
Claudeが作りました
"""

from tcg.config import fortress_cool, fortress_limit
from tcg.controller import Controller


class ClaudePlayer(Controller):
    """
    戦略的AIプレイヤー

    改善された戦略:
    1. ゲームフェーズ（序盤/中盤/終盤）に応じた戦略
    2. 攻撃成功率の計算
    3. 経済成長と軍事拡大のバランス
    4. 重要拠点の優先的確保
    """

    # 要塞の重要度（接続数と位置に基づく）
    FORTRESS_IMPORTANCE = {
        0: 3, 1: 4, 2: 3,      # 上側エリア
        3: 6, 4: 10, 5: 6,     # 中央上
        6: 6, 7: 10, 8: 6,     # 中央下
        9: 3, 10: 4, 11: 3     # 下側エリア
    }

    def __init__(self) -> None:
        super().__init__()
        self.step = 0

    def team_name(self) -> str:
        return "Strategic"

    def estimate_attack_success(self, attacker_troops: float, defender_troops: float,
                                defender_level: int, defender_kind: int, travel_time: int) -> bool:
        """
        攻撃が成功するか予測

        Args:
            attacker_troops: 攻撃側の部隊数
            defender_troops: 防御側の部隊数
            defender_level: 防御側のレベル
            defender_kind: 防御側の種類
            travel_time: 到着までの推定時間（ステップ数）

        Returns:
            攻撃が成功しそうならTrue
        """
        # 移動中に失う部隊は半分
        attacking_force = attacker_troops / 2

        # 到着までに敵が生産する部隊数を推定
        production_rate = fortress_cool[defender_kind][defender_level]
        if production_rate > 0:
            additional_troops = travel_time / production_rate
        else:
            additional_troops = 0

        total_defense = defender_troops + additional_troops

        # 攻撃力の計算（kind 0 = 0.65, kind 1 = 0.95 のダメージ）
        # 簡略化: 平均攻撃力 0.8 と仮定
        damage = attacking_force * 0.8

        # 成功条件: ダメージが防御側の部隊数を上回る + 余裕
        return damage > total_defense * 1.2

    def count_enemy_neighbors(self, fortress_id: int, state) -> int:
        """指定要塞に隣接する敵要塞の数を数える"""
        neighbors = state[fortress_id][5]
        return sum(1 for n in neighbors if state[n][0] == 2)

    def update(self, info) -> tuple[int, int, int]:
        """
        戦略的な判断でコマンドを選択
        """
        team, state, moving_pawns, spawning_pawns, done = info
        self.step += 1

        # ゲームフェーズの判定
        if self.step < 3000:
            phase = "early"
        elif self.step < 15000:
            phase = "mid"
        else:
            phase = "late"

        # 優先度付きアクションリスト
        actions = []

        # 自分の要塞と敵の要塞を分類
        my_fortresses = [i for i in range(12) if state[i][0] == 1]
        enemy_fortresses = [i for i in range(12) if state[i][0] == 2]
        neutral_fortresses = [i for i in range(12) if state[i][0] == 0]

        # === 序盤戦略: 中立要塞の制圧を最優先 ===
        if phase == "early":
            # 重要な中立要塞を優先的に取る
            for my_fort in my_fortresses:
                if state[my_fort][3] >= 4:
                    neighbors = state[my_fort][5]
                    for neighbor in neighbors:
                        if state[neighbor][0] == 0:
                            # 重要度が高い中立要塞を優先
                            importance = self.FORTRESS_IMPORTANCE[neighbor]
                            # 部隊数が少ない方が取りやすい
                            ease = max(0, 30 - state[neighbor][3])
                            priority = 150 + importance * 5 + ease
                            actions.append((priority, 1, my_fort, neighbor))

        # === 中立要塞への攻撃 ===
        for my_fort in my_fortresses:
            if state[my_fort][3] >= 6:
                neighbors = state[my_fort][5]
                for neighbor in neighbors:
                    if state[neighbor][0] == 0:
                        # 攻撃成功率を計算
                        if self.estimate_attack_success(
                            state[my_fort][3],
                            state[neighbor][3],
                            state[neighbor][2],
                            state[neighbor][1],
                            100  # 推定到着時間
                        ):
                            importance = self.FORTRESS_IMPORTANCE[neighbor]
                            priority = 120 + importance * 3
                            actions.append((priority, 1, my_fort, neighbor))

        # === 敵要塞への攻撃 ===
        for my_fort in my_fortresses:
            if state[my_fort][3] >= 10:
                neighbors = state[my_fort][5]
                for neighbor in neighbors:
                    if state[neighbor][0] == 2:
                        # 攻撃成功率を計算
                        if self.estimate_attack_success(
                            state[my_fort][3],
                            state[neighbor][3],
                            state[neighbor][2],
                            state[neighbor][1],
                            150
                        ):
                            # 敵の重要拠点を優先
                            importance = self.FORTRESS_IMPORTANCE[neighbor]
                            # 部隊が少ない敵要塞を優先
                            weakness = max(0, 25 - state[neighbor][3])
                            priority = 100 + importance * 2 + weakness
                            actions.append((priority, 1, my_fort, neighbor))

        # === アップグレード戦略 ===
        # 序盤: 重要拠点のみアップグレード
        # 中盤: 積極的にアップグレード
        # 終盤: 攻撃を優先、余裕があればアップグレード

        upgrade_priority_base = 90 if phase == "mid" else 70

        # 中央の重要拠点は常に優先
        for fort_id in [4, 7]:
            if state[fort_id][0] == 1:
                level = state[fort_id][2]
                if (state[fort_id][4] == -1 and
                    level <= 4 and
                    state[fort_id][3] >= fortress_limit[level] * 0.6):
                    priority = upgrade_priority_base + 30 + level * 5
                    actions.append((priority, 2, fort_id, 0))

        # その他の要塞のアップグレード
        for my_fort in my_fortresses:
            if my_fort not in [4, 7]:
                level = state[my_fort][2]
                importance = self.FORTRESS_IMPORTANCE[my_fort]
                # 敵に隣接している要塞は優先的にアップグレード
                enemy_neighbors = self.count_enemy_neighbors(my_fort, state)

                if (state[my_fort][4] == -1 and
                    level <= 4 and
                    state[my_fort][3] >= fortress_limit[level] * 0.65):
                    priority = upgrade_priority_base + importance + level * 3 + enemy_neighbors * 5
                    actions.append((priority, 2, my_fort, 0))

        # === 防御支援 ===
        # 攻撃されている要塞を検出
        under_attack = {}
        for pawn in moving_pawns:
            pawn_team, kind, from_, to, pos = pawn
            if pawn_team == 2 and state[to][0] == 1:
                if to not in under_attack:
                    under_attack[to] = 0
                under_attack[to] += 1

        for target_fort, threat_level in under_attack.items():
            # 脅威が大きい場合は優先度を上げる
            neighbors = state[target_fort][5]
            for my_fort in neighbors:
                if state[my_fort][0] == 1 and state[my_fort][3] >= 5:
                    priority = 110 + threat_level * 10
                    actions.append((priority, 1, my_fort, target_fort))

        # === 部隊の再配置 ===
        # 後方の安全な要塞から前線へ部隊を送る
        for my_fort in my_fortresses:
            level = state[my_fort][2]
            enemy_neighbors = self.count_enemy_neighbors(my_fort, state)

            # 敵に隣接していない要塞で部隊が溜まっている場合
            if enemy_neighbors == 0 and state[my_fort][3] >= fortress_limit[level] * 0.7:
                neighbors = state[my_fort][5]
                # 前線の味方要塞を探す
                for neighbor in neighbors:
                    if state[neighbor][0] == 1:
                        neighbor_enemy_count = self.count_enemy_neighbors(neighbor, state)
                        if neighbor_enemy_count > 0:
                            priority = 50 + neighbor_enemy_count * 5
                            actions.append((priority, 1, my_fort, neighbor))

        # === 積極的な中立要塞制圧（中盤以降で余裕がある場合）===
        if phase in ["mid", "late"] and len(my_fortresses) > len(enemy_fortresses):
            for my_fort in my_fortresses:
                if state[my_fort][3] >= 8:
                    neighbors = state[my_fort][5]
                    for neighbor in neighbors:
                        if state[neighbor][0] == 0 and state[neighbor][3] <= 15:
                            priority = 85
                            actions.append((priority, 1, my_fort, neighbor))

        # 最も優先度の高いアクションを実行
        if actions:
            actions.sort(reverse=True, key=lambda x: x[0])
            _, command, subject, to = actions[0]
            return command, subject, to

        # 何もすることがない場合
        return 0, 0, 0
