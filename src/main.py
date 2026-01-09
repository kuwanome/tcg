import pygame

from tcg.game import Game
# あなたのAI
from tcg.players.player_booon import booon
# ランダムAI（このパスを修正しました）
from tcg.players.sample_random import RandomPlayer

if __name__ == "__main__":
    # 対戦カードの表示
    print("=== booon (Blue) vs RandomPlayer (Red) ===")

    # ゲームの実行
    # 万が一これでも動かない場合は RandomPlayer() の代わりに 
    # ClaudePlayer() などを試してみてください。
    game = Game(booon(), RandomPlayer(), window=True)
    game.run()

    pygame.quit()