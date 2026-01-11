"""Game class for Fortress Conquest."""

import random
import pygame

from .config import (
    FPS,
    HEIGHT,
    SPEEDRATE,
    STEPLIMIT,
    WIDTH,
    A_coordinate,
    A_fortress_set,
    color_fortress,
    color_pawn,
    fortress_cool,
    fortress_limit,
    n_fortress,
    pos_fortress,
    swap_number_l,
)
from .controller import Controller
from .utils import flip_board_view


class Game:
    def __init__(self, controller1: Controller, controller2: Controller, window: bool = True):
        self.controller1 = controller1  # bottom
        self.controller2 = controller2  # up
        self.window_enabled = window

        self.team1 = self.controller1.team_name()
        self.team2 = self.controller2.team_name()

        if self.window_enabled:
            pygame.init()
            self.font = pygame.font.Font(None, 16)
            self.font_number = pygame.font.Font(None, 36)
            self.back_color = [150, 255, 150]
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            self.fps = pygame.time.Clock().tick
        self.seconds = 0

        self.state = [
            [0, 0, 1, 10, -1, [1, 3, 4]],
            [2, 0, 2, 20, -1, [0, 2, 4]],
            [0, 0, 1, 10, -1, [1, 4, 5]],
            [0, 0, 2, 20, -1, [0, 4, 6, 7]],
            [0, 1, 3, 30, -1, [0, 1, 2, 3, 5, 6, 7, 8]],
            [0, 0, 2, 20, -1, [2, 4, 7, 8]],
            [0, 0, 2, 20, -1, [3, 4, 7, 9]],
            [0, 1, 3, 30, -1, [3, 4, 5, 6, 8, 9, 10, 11]],
            [0, 0, 2, 20, -1, [4, 5, 7, 11]],
            [0, 0, 1, 10, -1, [6, 7, 10]],
            [1, 0, 2, 20, -1, [7, 9, 11]],
            [0, 0, 1, 10, -1, [7, 8, 10]],
        ]

        self.step = 0
        self.spawning_pawns = []
        self.moving_pawns = []
        self.score = 0
        self.win_team = "Both"
        self.Red_fortress = 1
        self.Blue_fortress = 1
        self.isGameOver = False
        self.isGameOver_loop = False
        self.Overed = False
        self.done = False

    def draw_fortress(self):
        if not self.window_enabled: return
        r = 0
        for x, y in pos_fortress:
            if r == 4 or r == 7:
                pygame.draw.rect(self.window, color_fortress[self.state[r][0]], pygame.Rect(x - 40, y - 40, 80, 80))
            else:
                pygame.draw.circle(self.window, color_fortress[self.state[r][0]], (x, y), 45)
            r += 1

    def draw_road(self):
        if not self.window_enabled: return
        for i in range(n_fortress):
            for j in range(n_fortress):
                if A_fortress_set[i][j] == 1:
                    pygame.draw.line(self.window, [200, 150, 50], pos_fortress[i], pos_fortress[j], 25)

    def draw_number(self):
        if not self.window_enabled: return
        for i in range(12):
            text = self.font.render(f"Lv {self.state[i][2]}", True, (0, 0, 0))
            self.window.blit(text, (pos_fortress[i][0] - 20, pos_fortress[i][1] - 35))
            text = self.font_number.render(f"{int(self.state[i][3])}", True, (0, 0, 0))
            self.window.blit(text, (pos_fortress[i][0] - 15, pos_fortress[i][1] - 5))
            if self.state[i][4] != -1:
                text = self.font.render(f"{int(self.state[i][4] // 2)}", True, (0, 0, 0))
                self.window.blit(text, (pos_fortress[i][0] + 25, pos_fortress[i][1] - 5))

    def draw_team_name(self):
        if not self.window_enabled: return
        self.window.blit(self.font_number.render(f"Red : {self.team2}", True, (200, 25, 25)), (10, 10))
        self.window.blit(self.font_number.render(f"Blue: {self.team1}", True, (25, 25, 200)), (10, HEIGHT - 50))

    def draw_pawn(self):
        if not self.window_enabled: return
        for team, kind, _, _, pos in self.moving_pawns:
            if kind == 0:
                pygame.draw.circle(self.window, color_pawn[team], pos, 5)
            else:
                pygame.draw.rect(self.window, color_pawn[team], pygame.Rect(pos[0]-2, pos[1]-2, 8, 8))

    def pawn_born(self):
        for i in range(12):
            if self.step % fortress_cool[self.state[i][1]][self.state[i][2]] == 0:
                if self.state[i][3] < fortress_limit[self.state[i][2]]:
                    self.state[i][3] += 1

    def pawn_over(self):
        for i in range(12):
            if self.step % 40 == 0 and self.state[i][3] > fortress_limit[self.state[i][2]]:
                self.state[i][3] -= 1

    def deliver(self, team, from_, to):
        if team == self.state[from_][0] and self.state[from_][3] >= 2:
            pos = [pos_fortress[from_][0] + A_coordinate[from_][to][0] * 42,
                   pos_fortress[from_][1] + A_coordinate[from_][to][1] * 42]
            self.spawning_pawns.append([team, self.state[from_][1], self.state[from_][3] // 2, from_, to, pos])
            self.state[from_][3] -= self.state[from_][3] // 2

    def upgrade(self, team, subject):
        if team == self.state[subject][0] and self.state[subject][3] >= fortress_limit[self.state[subject][2]] // 2 \
           and self.state[subject][4] == -1 and 1 <= self.state[subject][2] <= 4:
            self.state[subject][4] = 200
            self.state[subject][3] -= fortress_limit[self.state[subject][2]] // 2

    def check_upgrade(self):
        for i in range(n_fortress):
            if self.state[i][4] > 0: self.state[i][4] -= 1
            elif self.state[i][4] == 0:
                self.state[i][4] = -1
                self.state[i][2] += 1

    def pawn_departure(self):
        # 学習モードなら出現頻度を上げて密度を調整
        mod_val = 1 if not self.window_enabled else 7
        for i in range(len(self.spawning_pawns)):
            team, kind, pawn_number, from_, to, pos = self.spawning_pawns[i]
            if self.step % mod_val == 0 and pawn_number > 0:
                r = random.random() - 0.5
                p = [pos[0] + A_coordinate[from_][to][1] * r * 10,
                     pos[1] + A_coordinate[from_][to][0] * -1 * r * 10]
                self.moving_pawns.append([team, kind, from_, to, p])
                self.spawning_pawns[i][2] -= 1
        self.spawning_pawns = [p for p in self.spawning_pawns if p[2] > 0]

    def pawn_move(self):
        # 学習モードならスピードを5倍にする
        mult = 5.0 if not self.window_enabled else 1.0
        for i in range(len(self.moving_pawns)):
            team, kind, f, t, pos = self.moving_pawns[i]
            spd = 1.5 if kind == 0 else 1.0
            self.moving_pawns[i][4] = [pos[0] + A_coordinate[f][t][0] * spd * mult,
                                       pos[1] + A_coordinate[f][t][1] * spd * mult]

        arrived = []
        for p in self.moving_pawns:
            tx, ty = pos_fortress[p[3]]
            if (tx - p[4][0])**2 + (ty - p[4][1])**2 <= 45**2: arrived.append(p)
        for p in arrived: self.pawn_arrive(p)

    def pawn_arrive(self, pawn):
        team, kind, _, to, _ = pawn
        if team == self.state[to][0]: self.state[to][3] += 1
        else:
            self.state[to][3] -= 0.65 if kind == 0 else 0.95
            if self.state[to][3] < 0:
                self.state[to] = [team, self.state[to][1], 1, 0, -1, self.state[to][5]]
        if pawn in self.moving_pawns: self.moving_pawns.remove(pawn)

    def order(self, team, command, subject, to):
        if command == 1: self.deliver(team, subject, to)
        elif command == 2: self.upgrade(team, subject)

    def CheckGameOver(self):
        self.Blue_fortress = sum(1 for s in self.state if s[0] == 1)
        self.Red_fortress = sum(1 for s in self.state if s[0] == 2)
        self.win_team = "Red" if self.Red_fortress > self.Blue_fortress else "Blue" if self.Blue_fortress > self.Red_fortress else "Both"
        return self.Red_fortress == 0 or self.Blue_fortress == 0

    def check_event(self, event):
        if not self.window_enabled: return False
        return any(e.type == event for e in pygame.event.get())

    def run(self):
        while True:
            if self.window_enabled: self.seconds = pygame.time.get_ticks() // 1000
            if self.isGameOver or self.step > STEPLIMIT: break

            # 学習モードなら1フレームの間に物理演算を20回進める
            loops = int(SPEEDRATE) if self.window_enabled else 20 
            
            # --- AIの判断はループの外で1回だけ行う（フレームスキップ） ---
            if not self.isGameOver and not self.done:
                i1 = [1, self.state, self.moving_pawns, self.spawning_pawns, self.done]
                i2 = flip_board_view([2, self.state, self.moving_pawns, self.spawning_pawns, self.done])
                
                # ここで AI が意思決定 ＆ 学習（train_step）を行う
                c1, s1, t1 = self.controller1.update(i1)
                c2, s2, t2 = self.controller2.update(i2)

            # --- 物理演算（移動など）だけを高速で回す ---
            for _ in range(loops):
                if self.isGameOver or self.step >= STEPLIMIT or self.done:
                    self.isGameOver = True
                    break
                if self.window_enabled and self.check_event(pygame.QUIT): exit(0)

                self.pawn_move()
                self.done = self.CheckGameOver() or self.step >= STEPLIMIT - 1
                
                # 前の判断に従って命令を出し続ける
                self.order(1, c1, s1, t1)
                self.order(2, c2, swap_number_l[s2], swap_number_l[t2])

                self.pawn_departure()
                self.pawn_born()
                self.pawn_over()
                self.check_upgrade()
                self.step += 1

            if self.window_enabled:
                self.window.fill(self.back_color)
                self.draw_road(); self.draw_fortress(); self.draw_pawn(); self.draw_number(); self.draw_team_name()
                pygame.display.update()
                self.fps(int(FPS))
            
            if self.CheckGameOver(): self.isGameOver = True
        return self.win_team