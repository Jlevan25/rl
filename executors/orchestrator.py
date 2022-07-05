import numpy as np
import multiprocessing as mp

import torch

from logger import RunLogger
from configs import Config
# from elo_systems import MeanElo
from agents import A2CAgent


class Orchestrator:

    def __init__(self, states: int, actions: int, cfg: Config, name: str, queue: mp.Queue):
        self.cfg = cfg
        # self.mean_elo = MeanElo(cfg.players)
        self.train = None

        self.agents = [A2CAgent(id_=i, states=states, actions=actions, cfg=cfg)
                       for i in range(cfg.agents)]

        is_ond = True
        labelsx = [agent.label for agent in self.agents]
        labelsy = [agent.label for agent in self.agents]

        metrics = (
            # ActionMap(cfg.actions_key, labelsy, labelsx, is_ond, log_on_train=False),
            # SumArtifact(cfg.reward_key, labelsy, name + '2', log_on_train=False, is_global=True),
            # SumArtifact(cfg.reward_key, labelsy, name, log_on_eval=False, is_global=True),
            # EMAArtifact(cfg.reward_key, labelsy, name, log_on_train=False, is_global=True),
            # EMAArtifact(cfg.reward_key, labelsy, log_on_eval=False)
        )

    def reset(self, train: bool):
        # self.train = train
        # self.logger.set_mode(train)
        for agent in self.agents:
            agent.reset()
        # self.mean_elo.reset()

    def act(self, obs):
        obs = self._preprocess(obs)
        actions = [agent.act(obs) for agent in self.agents]
        # self.logger.log({self.cfg.actions_key: actions})
        return actions

    def rewarding(self, rewards, next_obs, last):
        next_obs = self._preprocess(next_obs)
        result = False
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):

            res = agent.rewarding(reward, next_obs, last)
            if not self.train and res is not None and res:
                result = True

        # self.logger.log({self.cfg.reward_key: rewards})

        # if not self.train:
        #     self._changed[-1] += 1
        #     if last:
        #         self.logger.call('ema_plots', self._changed[:-1] if len(self._changed) > 1 else None)
        #     if result or last:
        #         self._changed.append(self._changed[-1])
        #         self.logger.call('action_map')

        # if not self.train:
        #     elos = self.mean_elo.step(rewards)
        #     for agent, elo in zip(self.agents, elos):
        #         self.logger.log({agent.elo_key: elo})

    def _preprocess(self, obs):
        return obs.flatten(1) if not self.cfg.matrix_game else 1
