from typing import Tuple, Union

import torch
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical

from collections import deque

from models import ActorCritic
from configs import Config
from agents.agent_base import Agent
from torch.nn import functional as F


# n-step TD A2C
from utils import to_one_hot


class A2CAgent(Agent):

    # def __init__(self, id: int, states: int, actions: Union[Tuple[int], int], cfg: Config):
    def __init__(self, id_: int, model, obs_spaces_shape: tuple, cfg: Config):
        super().__init__(id_, model)
        self.cfg = cfg

        self.values = []
        self.obs_spaces_shape = obs_spaces_shape
        self.obs_capacity = model.obs_capacity
        self.e_greedy = self.cfg.e_greedy
        self.obs = deque(maxlen=self.obs_capacity)
        self.reset()

    def reset(self):
        self.obs.extend(torch.zeros(self.obs_capacity, *self.obs_spaces_shape))

    def take_action(self, obs):
        self.obs.append(torch.as_tensor(obs).float())
        state = torch.stack(list(self.obs), dim=-2)
        logits, value = self.model(state.to(self.cfg.device))
        dist = Categorical(logits=logits)

        if self.training and torch.rand((1,)).item() < self.e_greedy:
            *shape, high = logits.shape
            action = torch.randint(high, shape)
        else:
            action = dist.sample()

        if self.training:
            probs = F.softmax(logits, dim=-1)
            action_prob = torch.sum(probs * to_one_hot(action, self.cfg.players - 1), dim=-1)

            self.probs.append(action_prob)
            self.entropies.append(dist.entropy())
            self.values.append(value)

        return action
