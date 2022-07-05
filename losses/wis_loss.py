from typing import List

import torch
from torch import tensor, nn
from torch.nn import functional as F

from agents.a2c_poolswim import A2CAgentSwim
from losses.loss_base import OPELoss
from utils import to_one_hot


class WISLoss(OPELoss):
    def __init__(self, agents: List[A2CAgentSwim], window_size):
        super().__init__(agents)
        self.choose_probs = []
        self.win_size = window_size
        self.weights = torch.linspace(1 / window_size, 1, window_size)
        for agent in self.agents:
            agent.choosing_layer.register_forward_hook(self._save_choose)

    def _save_choose(self, module, input_, output):
        if module.training:
            self.choose_probs.append(output)

    def get_loss(self) -> tensor:
        # choose_probs = F.softmax(torch.stack(self.choose_probs), dim=-1)
        # ope_rewards = torch.mean(self.rewards[..., None] * self.ope_values, dim=-2)

        # importance_sampling
        ope_rewards = self.rewards[..., None] * self.ope_values

        # one_hot
        target = to_one_hot(torch.argmax(ope_rewards, dim=-1), num_columns=self.ope_values.shape[-1])
        # target = (torch.arange(self.ope_values.shape[-1]) == torch.argmax(ope_rewards, dim=-1, keepdim=True)) * 1.

        # outputs
        choose_probs = torch.stack(self.choose_probs, dim=1).reshape(target.shape)

        # cross-entropy
        loss = -(target * torch.log_softmax(choose_probs, dim=-1)).sum(-1).mean((0, -1))

        self.choose_probs = []

        return loss
