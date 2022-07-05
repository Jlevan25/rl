from typing import List

import torch

from agents.agent_base import Agent
from losses.loss_base import ActorCriticLoss


class CriticLoss(ActorCriticLoss):
    def __init__(self, agents: List[Agent],
                 discount: float = 0.96):
        super().__init__(agents)
        self.discount = discount
        self._powers = dict()

    def _calc_discount_powers(self, episodes):
        powers = torch.maximum(torch.arange(episodes)[..., None] - torch.arange(episodes), torch.zeros(1))
        self._powers[str(episodes)] = powers.float()

    def get_loss(self, eps=1e-8):

        batch_size, agents, episodes = self.rewards.shape

        # A_t = SUM_l((gamma*lambda)^l * TD_(t+l))
        if str(episodes) not in self._powers:
            self._calc_discount_powers(episodes)

        discount = self.discount ** self._powers[str(episodes)]

        g = self.rewards[..., None] * torch.tril(torch.ones(episodes, episodes))

        returns = torch.sum(g * discount, dim=2)

        value_loss = torch.mean((returns - self.values) ** 2, dim=2)

        return torch.mean(value_loss / 2, dim=0)

