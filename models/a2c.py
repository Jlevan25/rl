import torch
import torch.nn as nn
from typing import Tuple, Union
from configs import Config


class ActorCritic(nn.Module):

    def __init__(self, states: int, actions: Union[Tuple[int], int], hidden_n=64, obs_capacity=1):
        super().__init__()

        num_decisions, num_actions = actions if isinstance(actions, tuple) and len(actions) == 2 else (1, actions)

        self.states = states
        self.actions = actions
        self.obs_capacity = obs_capacity

        self.convs = nn.Sequential(nn.Conv2d(self.states, hidden_n, kernel_size=3),
                                   # nn.LayerNorm(hidden_n),
                                   nn.ReLU(),
                                   nn.Conv2d(hidden_n, hidden_n, kernel_size=(3, min(3, num_actions - 1))),
                                   # nn.LayerNorm(hidden_n),
                                   nn.ReLU())

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # self.mlp = nn.Sequential(nn.Linear(hidden_n, hidden_n),
        #                          nn.ReLU(),
        #                          nn.Linear(hidden_n, hidden_n),
        #                          nn.ReLU())

        # self.mlp = nn.Sequential(nn.Linear(hidden_n, hidden_n),
        #                          nn.ReLU(),
        #                          nn.Linear(hidden_n, hidden_n))

        self.opponent_patterns = nn.Linear(hidden_n, num_actions)
        # self.decision_makers = [nn.Linear(hidden_n, num_actions) for _ in range(num_decisions)]
        self.decision_makers = [nn.Linear(hidden_n, num_decisions) for _ in range(num_actions)]

        self.value = nn.Linear(hidden_n, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs):
        x = obs.reshape(obs.shape[0], -1, *obs.shape[-2:])
        x = self.convs(x)
        x = self.gap(x).flatten(1)
        # x = self.mlp(x)

        logits = torch.stack([decision(x) for decision in self.decision_makers], dim=1)

        if self.training:
            v = self.value(x)
            return logits, v.flatten()

        return logits, None

        # todo
        # self_obs, opponents_obs = x[:, 0], x[:, 1:]
        # v = self.value(self_obs.detach())
        # logits_self = torch.stack([decision(self_obs) for decision in self.decision_makers], dim=1)
        # logits = logits_self + self.opponent_patterns(opponents_obs) / 4
        # logits = self.softmax(logits.transpose(-1, 1))
        # return logits, v.flatten()
