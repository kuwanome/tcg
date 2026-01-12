"""
Template AI Player

このファイルをコピーして、あなた独自のAIプレイヤーを実装してください。

使い方:
1. このファイルをコピー: cp template_player.py player_yourname.py
2. クラス名を変更: TemplatePlayer -> YourPlayerName
3. team_name() の返り値を変更
4. update() メソッドに戦略を実装
"""

from tcg.controller import Controller


class TemplatePlayer(Controller):
    """
    テンプレートAIプレイヤー

    このクラスをベースに独自の戦略を実装してください。
    """

    def __init__(self) -> None:
        super().__init__()
        self.step = 0

    def team_name(self) -> str:
        """
        プレイヤー名を返す

        トーナメント結果の表示に使用されます。

        Returns:
            str: プレイヤー名
        """
        return "TemplateName"

    def update(self, info) -> tuple[int, int, int]:
        """
        毎ステップ呼ばれるメソッド

        ゲームの状態を受け取り、実行するコマンドを返します。

        Args:
            info: ゲーム情報のタプル
                - team (int): 自分 1、相手 2、中立 0
                - state (list): 12個の要塞の状態
                    state[i] = [team, kind, level, pawn_number, upgrade_time, [to_set]]
                - moving_pawns (list): 移動中の部隊情報
                - spawning_pawns (list): 出発待ちの部隊
                - done (bool): ゲーム終了フラグ

        Returns:
            tuple[int, int, int]: (command, subject, to)
                - command: 0=何もしない, 1=部隊移動, 2=アップグレード
                - subject: 対象の要塞ID (0-11)
                - to: 移動先の要塞ID (commandが1の場合のみ有効)
        """
        # ゲーム情報を展開
        team, state, moving_pawns, spawning_pawns, done = info
        self.step += 1

        # ==========================================
        # ここにあなたの戦略を実装してください
        # ==========================================

        # 例1: 何もしない
        command, subject, to = 0, 0, 0

        # 例2: 自分の要塞から敵要塞へ攻撃
        # for i in range(12):
        #     if state[i][0] == 1 and state[i][3] > 10:  # 自分の要塞で部隊が10以上
        #         neighbors = state[i][5]  # 隣接要塞
        #         enemy_neighbors = [n for n in neighbors if state[n][0] == 2]
        #         if enemy_neighbors:
        #             return 1, i, enemy_neighbors[0]  # 攻撃

        # 例3: アップグレード
        # for i in range(12):
        #     if state[i][0] == 1 and state[i][4] == 0:  # 自分の要塞でアップグレード可能
        #         level = state[i][2]
        #         if state[i][3] >= fortress_limit[level] // 2:
        #             return 2, i, 0  # アップグレード

        return command, subject, to
