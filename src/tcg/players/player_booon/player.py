import torch
from tcg.controller import Controller
from trainer import safe_get, calculate_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class booon(Controller):
    def __init__(self, model, strategy, mode="train"):
        super().__init__()
        self.model = model
        self.strategy = strategy
        self.mode = mode
        self.trainer = None
        self.epsilon = 1.0 if mode == "train" else 0.05
        self.last_state = None
        self.last_action = None
        self.last_info = None

    def team_name(self):
        return "booon"

    def _get_state_vector(self, info):
        # AIが盤面を理解するための情報を整理（63次元）
        team_id, state, pawn, _, _ = info
        res = []
        my_unit = next((p for p in pawn if p[0] == team_id), None)
        
        # 自分の位置・レベル情報 (3次元)
        res.extend([safe_get(my_unit, 3)/100.0, safe_get(my_unit, 4)/100.0, safe_get(my_unit, 2)/100.0])
        
        # 拠点情報 (自分のチーム=1.0, 中立=0.0, 敵=-1.0)
        for s in state:
            if s[0] == team_id:
                rel_team = 1.0
            elif s[0] == 0:
                rel_team = 0.0
            else:
                rel_team = -1.0
            res.extend([rel_team, safe_get(s,1)/100.0, safe_get(s,2)/5.0, min(safe_get(s,3)/50.0, 1.0), safe_get(s,4)/100.0])
        
        return torch.FloatTensor(res).unsqueeze(0).to(device)

    def update(self, info):
        team_id = info[0]
        current_state_vector = self._get_state_vector(info)

        if self.mode == "train" and self.last_state is not None and self.trainer is not None:
            reward = calculate_reward(info, self.last_info)
            self.trainer.memory.push(self.last_state, self.last_action, reward, current_state_vector, info[4])
            self.trainer.train_step()

        action_idx, command = self.strategy.get_action(current_state_vector, self.epsilon, info[1], team_id)
        
        if self.mode == "train":
            self.last_state = current_state_vector
            self.last_action = action_idx
            self.last_info = info

        return command