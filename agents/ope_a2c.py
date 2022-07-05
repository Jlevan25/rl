from collections import deque
from typing import Union, List

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from agents.a2c import A2CAgent
from configs import Config
from utils import to_one_hot


class A2CAgentOPE(A2CAgent):
    def __init__(self, id_: int, model, obs_spaces_shape: tuple, decision_layers: Union[List[nn.Module], nn.Module],
                 cfg: Config):

        super().__init__(id_, model, obs_spaces_shape, cfg)

        self.test_mode: bool = False

        self.decision_layers = decision_layers
        for layer in self.decision_layers:
            layer.register_forward_hook(self._save_features)

        # self.policies_pool = {'weight': [], 'bias': []}
        self.policies_weights, self.policies_biases = None, None

        self.features = []
        self.ope_values = []

    def train(self, mode: bool = True):
        if mode:
            for layer in self.decision_layers:
                layer.reset_parameters()

        return super().train(mode)

    # def train(self, mode: bool = True):
    #     if mode:
    #         self.training = mode
    #         self.model.training = mode
    #         self.model.value.train()
    #         for layer in self.decision_layers:
    #             layer.reset_parameters()
    #             layer.train()
    #         return self
    #     else:
    #         return super().train(mode)

    # def eval(self):
    #     # self.policies_weights = torch.stack(self.policies_pool['weight'])
    #     # self.policies_biases = torch.stack(self.policies_pool['bias']) if self.policies_pool['bias'] else None
    #     return super().train(False)

    def test(self):
        self.test_mode = True
        return self.eval()

    def _save_features(self, module, input_, output):
        if not self.training and self.policies_weights is not None:
            self.features.append(*input_)
            # self.policies = torch.stack(self.policies_pool, dim=1)
            # logits = torch.dot(x, self.weight.data.T) + (self.bias.data if self.bias is not None else 0)

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

        if self.features:
            features = torch.stack(self.features)
            # policies_weights = torch.stack(self.policies_pool['weight'])
            # policies_biases = torch.stack(self.policies_pool['bias']) if self.policies_pool['bias'] else None

            # TODO change from only_linear to variety
            # shape(num_policies, num_decision_layes, batch_size, opponent_agents/actions)
            policies_logits = torch.matmul(features, self.policies_weights.transpose(-1, -2))
            policies_logits = policies_logits + self.policies_biases.unsqueeze(-2)
            # policies_logits = F.linear(features, policies_weights, policies_biases)

            # one_hot_action = (torch.arange(self.cfg.players - 1) == action[..., None]) * 1.
            one_hot_action = to_one_hot(action, num_columns=self.cfg.players - 1)

            # norm logits := logits - logits.logsumexp(dim=-1, keepdim=True)

            policies_log_probs = torch.log_softmax(policies_logits.transpose(2, 1), dim=-1)

            policies_action_log_probs = torch.sum(policies_log_probs * one_hot_action, dim=-1)

            ope_values = policies_action_log_probs - dist.log_prob(action)

            # prod of actions probs
            self.ope_values.append(ope_values.sum(-1).T)

            self.features = []

        if self.training:
            probs = F.softmax(logits, dim=-1)
            action_prob = torch.sum(probs * to_one_hot(action, self.cfg.players - 1), dim=-1)

            self.probs.append(action_prob)
            self.entropies.append(dist.entropy())
            self.values.append(value)

        return action

    # def _save_decision_layer(self):
    #     weights, biases = [], []
    #     for layer in self.decision_layers:
    #         weights.append(layer.weight)
    #         if layer.bias is not None:
    #             biases.append(layer.bias)
    #
    #     self.policies_pool['weight'].append(torch.stack(weights))
    #     if biases:
    #         self.policies_pool['bias'].append(torch.stack(biases))

    def _save_decision_layer(self):
        weights, biases = [], []
        for layer in self.decision_layers:
            weights.append(layer.weight)
            if layer.bias is not None:
                biases.append(layer.bias)

        weights = torch.stack(weights)[None, ...]
        self.policies_weights = weights if self.policies_weights is None else \
            torch.cat((self.policies_weights, weights), dim=0)

        if biases:
            biases = torch.stack(biases)[None, ...]
            self.policies_biases = biases if self.policies_biases is None else \
                torch.cat((self.policies_biases, biases), dim=0)

    def get_reward(self, reward):
        self.rewards.append(torch.as_tensor(reward))

        if not all((self.training, self.test_mode)) and len(self.rewards) == self.cfg.test_episodes:
            # if self.policies_pool['weight']:
            if self.policies_weights is not None:  # save policy in pool or not
                ope_values = torch.stack(self.ope_values, dim=1)
                rewards = torch.stack(self.rewards)
                ope_rewards = torch.mean(rewards * ope_values, dim=list(range(0, len(ope_values.shape) - 1)))
                if torch.all(ope_rewards < 0):
                    print(f'Welcome policy #{len(ope_rewards) + 1} to agent{self.id} pool! Other policies:',
                          ope_rewards.tolist())
                    self._save_decision_layer()
                self.ope_values = []
            else:  # if pool is empty -> just save
                self._save_decision_layer()

            self.rewards = []

    def save_model(self) -> dict:
        return dict(model=self.model.state_dict(),
                    pool_weights=self.policies_weights,
                    pool_biases=self.policies_biases)

    def load_model(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint['model'])
        if [k for k in checkpoint if 'pool_' in k]:
            self.policies_weights = checkpoint['pool_weights']
            self.policies_biases = checkpoint['pool_biases']
        else:
            self._save_decision_layer()

        self.model.eval()
