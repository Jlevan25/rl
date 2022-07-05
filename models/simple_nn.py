import torch
import torch.nn as nn
from typing import Tuple, Union
from configs import Config


class SimpleAC(nn.Module):

    def __init__(self, states: int, actions: Union[Tuple[int], int], hidden_n=64, obs_capacity=1):
        super().__init__()

        num_decisions, num_actions = actions if isinstance(actions, tuple) and len(actions) == 2 else (1, actions)

        self.states = states
        self.actions = actions
        self.obs_capacity = obs_capacity

        input_dim = (num_decisions + 1) * obs_capacity*states

        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_n),
                                 nn.ReLU(),
                                 nn.Linear(hidden_n, hidden_n))

        self.decision_makers = [nn.Linear(hidden_n, num_decisions) for _ in range(num_actions)]
        self.value = nn.Linear(hidden_n, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs):
        x = obs.flatten(1)
        x = self.mlp(x)

        #logits = torch.stack([decision(x) for decision in self.decision_makers], dim=-1)
        logits = torch.stack([decision(x) for decision in self.decision_makers], dim=1)
        # logits = self.softmax(logits)

        if self.training:
            v = self.value(x)
            return logits, v.flatten()

        return logits, None
