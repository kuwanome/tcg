"""
Gemini2Player (Builder / Economy Focus)
戦略:
1. 「攻撃」よりも「レベル上げ」を最優先にする。
2. 兵力が溜まったら、まずLv5を目指して投資する。
3. アップグレードできない時だけ、勝てる敵を攻める。
"""
import random
from tcg.config import fortress_limit
from tcg.controller import Controller

class Gemini2Player(Controller):

    def team_name(self):
        return "Gemini2 (Builder)"

    def update(self, info) -> tuple[int, int, int]:
        self.team, self.state, self.moving_pawns, self.spawning_pawns, self.done = info

        my_fortresses = [i for i in range(12) if self.state[i][0] == self.team]
        random.shuffle(my_fortresses)

        for subject in my_fortresses:
            fortress_team, kind, level, pawns, upgrade_timer, neighbors = self.state[subject]
            
            # Lv5対応のリミット取得
            try:
                limit = fortress_limit[level]
            except IndexError:
                limit = fortress_limit[4]

            attack_rate = 0.95 if kind == 1 else 0.65
            
            # 兵力が少なすぎる時は何もしない
            if pawns < limit * 0.4:
                continue

            # ---------------------------------------------------
            # 【変更点】 優先順位 1: レベル上げ（内政）
            # 攻撃よりも先に、レベル上げを検討する
            # ---------------------------------------------------
            if pawns > limit * 0.85 and level < 5:
                return 2, subject, neighbors[0] # toは関係ないので隣を入れる

            # ---------------------------------------------------
            # 【変更点】 優先順位 2: 攻撃
            # レベル上げができない（または不要な）場合に、初めて攻撃を検討する
            # ---------------------------------------------------
            sending_pawns = pawns // 2
            real_power = sending_pawns * attack_rate

            best_target = -1
            best_score = -9999

            for target in neighbors:
                t_team = self.state[target][0]
                t_kind = self.state[target][1]
                t_pawns = self.state[target][3]

                if t_team != self.team:
                    if real_power > t_pawns + 2:
                        score = 100 - t_pawns
                        
                        # 4番・7番の優先度
                        if t_kind == 1:
                            if t_team != 0: score += 500
                            else: score += 200
                        elif t_team == 0:
                             score += 50
                        
                        if score > best_score:
                            best_score = score
                            best_target = target

            if best_target != -1:
                return 1, subject, best_target


            # ---------------------------------------------------
            # 戦略3: 【特攻】満タンでどうしようもないなら敵を削る
            # ---------------------------------------------------
            if pawns > limit * 0.9:
                target_to_attack = -1
                
                # 優先度1: 敵の4 or 7
                for target in neighbors:
                    t_team = self.state[target][0]
                    t_kind = self.state[target][1]
                    if t_team != self.team and t_team != 0 and t_kind == 1:
                        target_to_attack = target
                        break
                
                # 優先度2: なければ最弱の敵
                if target_to_attack == -1:
                    min_enemy_pawns = 9999
                    for target in neighbors:
                        t_team = self.state[target][0]
                        t_pawns = self.state[target][3]
                        if t_team != self.team:
                            if t_pawns < min_enemy_pawns:
                                min_enemy_pawns = t_pawns
                                target_to_attack = target

                if target_to_attack != -1:
                    return 1, subject, target_to_attack

            # ---------------------------------------------------
            # 戦略4: 【支援】味方に送る（Lv5未満優先）
            # ---------------------------------------------------
            if pawns > limit * 0.8:
                target_low_level = []
                target_high_level = []

                for target in neighbors:
                    if self.state[target][0] == self.team:
                        t_level = self.state[target][2]
                        
                        try:
                            t_limit = fortress_limit[t_level]
                        except IndexError:
                            t_limit = fortress_limit[4]

                        if self.state[target][3] < t_limit * 0.9:
                            if t_level < 5:
                                target_low_level.append(target)
                            else:
                                target_high_level.append(target)
                
                if target_low_level:
                    return 1, subject, target_low_level[0]
                elif target_high_level:
                    return 1, subject, target_high_level[0]

        return 0, 0, 0