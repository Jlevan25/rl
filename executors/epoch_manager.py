import os
from typing import Iterator, List

import numpy as np

import torch
from torch import nn, tensor
from torch.optim import Optimizer

from configs import Config
# from elo_systems import MeanElo
# from agents import A2CAgent
# from utils import timer

# from metrics import CoopsMetric, ActionMap, AvgCoopsMetric, AvgRewardMetric, RewardMetric


import time

from utils import to_one_hot


class EpochManager:

    def __init__(self,
                 env,
                 agents,
                 peers,
                 # agents_type: List[Type],
                 # peers_type: List[Type],
                 optimizers: List[Optimizer],
                 criteria: List[nn.Module],
                 cfg: Config,
                 scheduler=None,
                 exploration_scheduler=None,
                 writer=None,
                 metrics: dict = None):

        self.cfg = cfg

        self.criteria = criteria
        self.scheduler = scheduler
        self.env = env

        # self.agents = [agent(id=i, states=env.states, actions=env.actions, cfg=cfg)
        #                for i, agent in enumerate(agents_type)]
        #
        # self.peers = [peer(id=(cfg.agents + i + 1), states=env.states, actions=env.actions, cfg=cfg).eval()
        #               for i, peer in enumerate(peers_type)]

        self.agents, self.peers = agents, peers

        self.players = self.agents + self.peers

        self.optimizers = optimizers

        # self.optimizer = optimizer_type(params=sum([list(agent.parameters()) for agent in self.agents], []),
        #                                 **optimizer_kwargs)

        self.exploration_scheduler = exploration_scheduler

        self.metrics = metrics
        self.writer = writer
        self._global_step = dict()

    @torch.no_grad()
    def inference(self, stage_key, i_epoch=None):
        for agent in self.agents:
            agent.eval()

        mean_reward = np.zeros(self.cfg.players)
        for step, rewards in enumerate(self._epoch_generator(stage_key, self.cfg.test_episodes, i_epoch)):
            mean_reward = (mean_reward + rewards.mean(0))
            ovr_reward = np.round(mean_reward / (step + 1), 3)

            self._write_scalars(f'{stage_key}/Rewards', ovr_reward, self._global_step[stage_key])

            if (step + 1) % self.cfg.show_each == 0 or (step + 1) == self.cfg.test_episodes:
                print((step + 1),
                      'Rewards', ovr_reward)

    def train(self, stage_key, i_epoch):
        for agent in self.agents:
            agent.train()

        epoch_reward, epoch_loss = np.zeros(self.cfg.players), np.zeros(self.cfg.agents)
        loss_step = 0

        start = time.perf_counter()

        if self.exploration_scheduler is not None:
            self.exploration_scheduler.step()

        for step, rewards in enumerate(self._epoch_generator(stage_key, self.cfg.train_episodes, i_epoch)):
            epoch_reward = (epoch_reward + rewards.mean(0))
            mean_reward = np.round(epoch_reward / (step + 1), 3)

            if (step + 1) % self.cfg.steps == 0 or step == (self.cfg.train_episodes - 1):
                loss_step += 1
                agents_losses = []
                for criterion in self.criteria:
                    agents_loss = criterion()
                    iter_loss = agents_loss.mean(-1) if len(agents_loss.shape) > 1 else agents_loss
                    agents_losses.extend(iter_loss.detach().numpy())
                    loss = iter_loss.sum()
                    loss.backward()

                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss = (epoch_loss + np.array(agents_losses))

            mean_loss = np.round(epoch_loss / max(loss_step, 1), 3)

            self._write_scalars(f'{stage_key}/Loss', mean_loss, self._global_step[stage_key])

            self._write_scalars(f'{stage_key}/Rewards', mean_reward, self._global_step[stage_key])

            if (step + 1) % self.cfg.show_each == 0 or step == (self.cfg.train_episodes - 1):
                print((step + 1),
                      'Rewards', mean_reward,
                      'Loss', mean_loss)

        end = time.perf_counter()
        print(f"Epoch #{i_epoch + 1} end in {end - start:0.8f} seconds")

        if self.scheduler is not None:
            self.scheduler.step()

    def _get_global_step(self, stage):
        self._global_step[stage] = -1

    def _write_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _write_scalars(self, tag, values, step):
        if self.writer is not None:
            self.writer.add_scalars(tag, {str(k): v for k, v in enumerate(values)}, step)

    def _epoch_generator(self, stage, episodes, epoch=None) -> Iterator[tensor]:

        if stage not in self._global_step:
            self._get_global_step(stage)

        calc_metrics = self.metrics[stage] is not None and len(self.metrics[stage]) > 0
        print('\n_______', stage, f'epoch{epoch + 1}' if epoch is not None else '', '_______')

        obs = self.env.reset()
        for episode in range(episodes):
            self._global_step[stage] += 1
            debug = self.cfg.debug and episode % self.cfg.show_each == 0

            actions = self._make_act(obs)
            obs, rewards = self.env.step(actions)
            for idx, player in enumerate(self.players):
                player.get_reward(reward=rewards[..., idx])

            # if calc_metrics:
            #     self._calc_batch_metrics(predictions.argmax(1).detach().cpu(), targets.cpu(), stage, debug)

            yield rewards

        # if calc_metrics:
        #     print('\n___', f'Epoch Summary', '___')
        #     self._calc_epoch_metrics(stage)

    def _make_act(self, obs):
        actions = np.zeros((self.cfg.batch_size, 2, self.cfg.players, self.cfg.players), dtype=np.int32)
        num_opponents = self.cfg.players - 1

        for player_id, (player, player_obs) in enumerate(zip(self.players, self._preprocess(obs))):
            action = player.take_action(player_obs)
            """
                shifted_actions = one_hot(action + shift)
                shift example: (1st agent for 3rd is 1st in actions matrix) in 3 player env
                for second agent (1, 0) ->  (1, 0, 0), (0, 0, 1)
            """
            shifted_actions = (action.numpy() + player_id + 1) % self.cfg.players
            actions[..., player_id, :] = to_one_hot(shifted_actions, num_columns=self.cfg.players)
            # actions[..., player_id, :] = (np.arange(self.cfg.players) == shifted_actions[..., None]) * 1

        return actions

    def _preprocess(self, obs):
        # if self.cfg.matrix_game:
        #     return np.ones((self.cfg.players, *obs.shape[:-1], 1))

        return [np.roll(obs, (-i, -i), axis=(-2, -1)) for i in range(self.cfg.players)]
        # return [np.concatenate((obs[:, i:], obs[:, :i]), axis=1) for i in range(self.cfg.players)]

    def save_models(self, epoch, path=None):
        path = self.cfg.SAVE_PATH if path is None else path

        path = os.path.join(path, f'epoch{epoch}')
        models_path = os.path.join(path, 'models')
        optimizers_path = os.path.join(path, 'optimizers')

        if not os.path.exists(path):
            os.makedirs(models_path)
            os.makedirs(optimizers_path)

        for i, agent in enumerate(self.agents):
            agent_checkpoint = agent.save_model()
            torch.save(agent_checkpoint, os.path.join(models_path, f'agent{i}_{type(agent).__name__}.pth'))

        for i, optim in enumerate(self.optimizers):
            torch.save(optim.state_dict(), os.path.join(optimizers_path, f'optim{i}_{type(optim).__name__}.pth'))

        checkpoint = dict(epoch=self._global_step,
                          exploration_scheduler=self.exploration_scheduler.state_dict())

        torch.save(checkpoint, os.path.join(path, 'checkpoint.pth'))
        print('model saved, epoch:', epoch)

    def load_models(self, path):
        checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))

        models_path = os.path.join(path, 'models')
        optimizers_path = os.path.join(path, 'optimizers')

        agents_name = {f.split('_')[0]: f for f in os.listdir(models_path)}
        optims_name = {f.split('_')[0]: f for f in os.listdir(optimizers_path)}

        for i, agent in enumerate(self.agents):
            agent_checkpoint = torch.load(os.path.join(models_path, agents_name[f'agent{i}']),
                                          map_location=torch.device(self.cfg.device))
            agent.load_model(agent_checkpoint)

        for i, optim in enumerate(self.optimizers):
            optim_checkpoint = torch.load(os.path.join(optimizers_path, optims_name[f'optim{i}']))
            optim.load_state_dict(optim_checkpoint)

        self._global_step = checkpoint['epoch']
        self.exploration_scheduler = checkpoint['exploration_scheduler']
        print('model loaded')
