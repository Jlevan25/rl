from typing import Tuple, Union

import torch
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical

from collections import deque

from models import ActorCritic
from configs import Config


class DummyAgent(nn.Module):
    # def __init__(self, id: int, states: int, actions: Union[Tuple[int], int], cfg: Config):
    def __init__(self, id_: int, action):

        super().__init__()
        self.id = id_
        self.action = action
        self.label = f'агент {id_ + 1}'

    def act(self, obs):
        return self.action, 0, 0
