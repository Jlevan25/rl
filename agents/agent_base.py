from abc import abstractmethod

import torch
from torch import nn

from configs import Config


class Agent(nn.Module):
    def __init__(self, id_: int, model):
        super().__init__()
        self.id = id_
        self.label = f'агент {id_ + 1}'

        self.probs, self.entropies, self.rewards = [], [], []

        self.model = model

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def take_action(self, obs):
        raise NotImplementedError()

    def get_reward(self, reward):
        if self.training:
            self.rewards.append(torch.as_tensor(reward))

    def save_model(self) -> dict:
        return dict(model=self.model.state_dict())

    def load_model(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint['model'])
