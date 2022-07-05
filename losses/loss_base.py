from abc import abstractmethod, ABC
from typing import List

import torch
from torch import nn, tensor
from agents.agent_base import Agent


class Loss(nn.Module):
    def __init__(self, agents: List[Agent]):
        super().__init__()
        self.agents = agents
        self.attributes: dict[List] = dict(probs=[], rewards=[], entropies=[])

    def forward(self, *args, **kwargs):
        self._collect_history()
        loss = self.get_loss(*args, **kwargs)
        self._clear_history()

        return loss

    @abstractmethod
    def get_loss(self, *args, **kwargs) -> tensor:
        raise NotImplementedError()

    def _collect_history(self, *attributes: List[str]):
        for agent in self.agents:
            for attr, list_ in self.attributes.items():
                list_.append(torch.stack(getattr(agent, attr), dim=1))

        for attr, list_ in self.attributes.items():
            setattr(self, attr, torch.stack(list_, dim=1))

    def _clear_history(self, *attributes: List[str]):
        for attr in self.attributes:
            self.attributes[attr] = []
            for agent in self.agents:
                setattr(agent, attr, [])


class ActorCriticLoss(Loss, ABC):
    def __init__(self, agents: List[Agent]):
        super().__init__(agents)
        self.attributes['values'] = []


class OPELoss(Loss, ABC):
    def __init__(self, agents: List[Agent]):
        super().__init__(agents)
        self.attributes['ope_values'] = []

    # def get_history(self):
    #     logprobs, rewards, entropies = [], [], []
    #     for agent in self.agents:
    #         logprobs.append(torch.stack(agent.logprobs, dim=1))
    #         rewards.append(torch.stack(agent.rewards, dim=1))
    #         entropies.append(torch.stack(agent.entropies, dim=1))
    #
    #     self.logprobs = torch.stack(logprobs, dim=1)
    #     self.rewards = torch.stack(rewards, dim=1)
    #     self.entropies = torch.stack(entropies, dim=1)
    #
    # def clear_history(self):
    #     self.logprobs, self.rewards, self.entropies = None, None, None
    #     for agent in self.agents:
    #         agent.logprobs = []
    #         agent.rewards = []
    #         agent.entropies = []
