from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from agents.agent_base import Agent
from utils import timer
from losses.loss_base import ActorCriticLoss


class GAE(ActorCriticLoss):
    def __init__(self, agents: List[Agent],
                 discount: float = 0.96, lambda_: float = 0.95, entropy: float = 0.01,
                 is_normalize: bool = False):
        super().__init__(agents)
        self.discount = discount
        self.lambda_ = lambda_
        self.entropy = entropy
        self.is_normalize = is_normalize
        self._powers = dict()

    def _calc_discount_powers(self, episodes):
        powers = torch.maximum(torch.arange(episodes)[..., None] - torch.arange(episodes), torch.zeros(1))
        self._powers[str(episodes)] = powers.float()

    def get_loss(self, eps=1e-8):

        batch_size, agents, episodes = self.rewards.shape
        next_values = torch.zeros_like(self.values)
        next_values[..., :-1] = self.values[..., 1:]

        # A_t = SUM_l((gamma*lambda)^l * TD_(t+l))
        if str(episodes) not in self._powers:
            self._calc_discount_powers(episodes)

        discount = self.discount ** self._powers[str(episodes)]
        lambda_ = self.lambda_ ** self._powers[str(episodes)]

        g = self.rewards[..., None] * torch.tril(torch.ones(episodes, episodes))

        delta = self.rewards + self.discount * next_values - self.values
        td_g = delta[..., None] * torch.tril(torch.ones(episodes, episodes))

        returns = torch.sum(g * discount, dim=2)
        advantage = torch.sum(td_g * lambda_ * discount, dim=2).detach()

        if self.is_normalize:
            advantage = advantage - torch.mean(advantage)
            std = torch.sqrt(torch.mean(advantage ** 2))
            advantage = advantage / (std + eps)

        value_loss = torch.mean((returns - self.values) ** 2, dim=2)

        if len(self.logprobs.shape) > 3:
            advantage = advantage[..., None]
            value_loss = value_loss[..., None]

        policy_loss = -torch.sum(self.logprobs * advantage + self.entropy * self.entropies, dim=2)

        # return torch.mean(policy_loss + value_loss / 2, dim=0).round_(decimals=6)
        return torch.mean(policy_loss + value_loss / 2, dim=0)

        # value_loss = torch.sum(torch.mean(delta**2, dim=0), dim=list(range(1, len(delta.shape) - 1)))
        # policy_loss = torch.mean(torch.sum(logprobs * advantage - self.entropy * entropies, dim=-1), dim=0)

        # returns = []
        # gae = 0
        # for i in reversed(range(episodes)):
        #     delta = rewards[..., i, None] + self.discount * next_values[..., i, :] - values[..., i, :]
        #     gae = delta + self.discount * self.lambda_ * gae
        #     returns.insert(0, gae + values[..., i, :])
        #
        # adv = torch.as_tensor(returns) - values

        # gae, advantage = 0., []
        # for timestep in reversed(range(episodes)):
        #     delta = rewards[:, timestep] + self.discount * values[:, timestep + 1] - values[:, timestep]  # TD
        #     gae = gae * self.discount * self.lambda_ + delta  # A_t = SUM_l((gamma*lambda)^l * TD_(t+l))
        #     advantage.insert(0, gae)
        # advantage = torch.stack(advantage, dim=1).T
