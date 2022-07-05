from collections import deque
from typing import Union, List

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from agents.a2c import A2CAgent
from configs import Config
from utils import to_one_hot


class A2CAgentSwim(A2CAgent):
    def __init__(self, id_: int, model, obs_spaces_shape: tuple, checkpoint,
                 cfg: Config):

        super().__init__(id_, model, obs_spaces_shape, cfg)

        self.test_mode: bool = False

        self.policies_weights, self.policies_biases = None, None
        self.model.value.register_forward_hook(self._save_features)
        self.load_model(checkpoint)

        self.choosing_layer = nn.Sequential(nn.Linear(self.cfg.h_space, self.cfg.h_space),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(self.cfg.h_space, self.cfg.h_space),
                                            nn.ReLU(),
                                            nn.Dropout(),
                                            nn.Linear(self.cfg.h_space, len(self.policies_weights)))

        self.features = None
        self.ope_values = []

    def _save_features(self, module, input_, output):
        self.features = input_[0] if isinstance(input_, tuple) else input_

    def train(self, mode: bool = True):
        if mode:
            self.training = mode
            self.model.training = mode
            # self.model.value.train()
            self.choosing_layer.train()
            return self
        else:
            return super().train(mode)

    def test(self):
        self.test_mode = True
        return self.eval()

    def take_action(self, obs):
        self.obs.append(torch.as_tensor(obs).float())
        state = torch.stack(list(self.obs), dim=-2)
        _, value = self.model(state.to(self.cfg.device))
        logits = self.choosing_layer(self.features)
        choose_probs = F.softmax(logits, dim=-1).reshape(*logits.shape, *[1] * len(self.policies_weights.shape[2:]))
        policy_weights = torch.sum(self.policies_weights * choose_probs[..., None], dim=1)
        policy_bias = torch.sum(self.policies_biases * choose_probs, dim=1) if self.policies_biases is not None else 0

        policy_logits = torch.sum(policy_weights * self.features[..., None, None, :], dim=-1) + policy_bias

        dist = Categorical(logits=policy_logits)

        if self.training and torch.rand((1,)).item() < self.e_greedy:
            *shape, high = policy_logits.shape
            action = torch.randint(high, shape)
        else:
            action = dist.sample()

        if self.training:

            policies_logits = torch.matmul(self.features, self.policies_weights.transpose(-1, -2))
            policies_logits = policies_logits + self.policies_biases.unsqueeze(-2)

            probs = F.softmax(logits, dim=-1)
            policies_probs = F.softmax(policies_logits.transpose(2, 1), dim=-1)

            one_hot_action = to_one_hot(action, num_columns=self.cfg.players - 1)
            action_prob = torch.sum(probs * one_hot_action, dim=-1)
            actions_policies_probs = torch.sum(policies_probs * one_hot_action, dim=-1)

            ope_values = actions_policies_probs / action_prob

            # prod of actions probs
            self.ope_values.append(ope_values.prod(-1).T)

            self.probs.append(action_prob)
            self.entropies.append(dist.entropy())
            self.values.append(value)

        return action

    def save_model(self) -> dict:
        return dict(model=self.model.state_dict(),
                    choosing_layer=self.choosing_layer,
                    pool_weights=self.policies_weights,
                    pool_biases=self.policies_biases)

    def load_model(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint['model'])
        if [k for k in checkpoint if 'pool_' in k]:
            self.policies_weights = checkpoint['pool_weights']
            self.policies_biases = checkpoint['pool_biases']
        else:
            raise ValueError('Need pool_weights')

        self.model.eval()

    def parameters(self, recurse: bool = True):
        return list(self.choosing_layer.parameters()) + list(self.model.value.parameters())

