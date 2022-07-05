import os
import time
from dataclasses import dataclass, asdict, field, replace
from typing import Union
import torch


class Config:
    def __init__(self,
                 agents: int = 1,  # trainable
                 players: int = 5,  # all amount
                 experiment_name: str = None,
                 epochs: int = 1,
                 exploration_epochs: int = 0,
                 repeats: int = 1,
                 batch_size=32,

                 environment: str = 'pluel',
                 matrix_game: bool = False,

                 debug=True,
                 show_each=1000,
                 seed: int = None,

                 # model
                 num_decisions=2,
                 h_space: Union[int, tuple] = 64,

                 # train
                 train_episodes: int = 50000,
                 discount: float = 0.95,
                 steps: int = 20,  # TD update
                 obs_capacity: int = 5,
                 prob_thr: float = 0.00001,
                 entropy: float = 0.01,
                 e_greedy: float = 0.05,
                 lr: Union[float, tuple] = 0.0025,
                 grad_clip: float = 10.,

                 # test
                 test_episodes: int = 10000,
                 enable_eval_agents: bool = True,
                 eval_agents: list = 3,

                 # keys for logging
                 loss_key: str = 'loss',
                 act_loss_key: str = 'actor_loss',
                 crt_loss_key: str = 'critic_loss',
                 actions_key: str = 'acts',
                 reward_key: str = 'reward',
                 elo_key: str = 'elo',
                 log_avg: int = 5,

                 ROOT: str = None,
                 SAVE_PATH: str = None,

                 # device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 device: str = 'cpu'):

        if agents > players:
            raise 'Number of Agent must be smaller than players'

        self.agents = agents  # trainable
        self.players = players  # all amount
        self.experiment_name = experiment_name if experiment_name is not None\
            else f'{environment}_agents{agents}_players_{players}'
        self.epochs: int = epochs
        self.exploration_epochs = exploration_epochs
        self.repeats: int = repeats
        self.batch_size = batch_size

        self.environment: str = environment
        self.matrix_game: bool = matrix_game

        self.debug = debug
        self.show_each = show_each
        self.seed: int = seed

        # model
        self.num_decisions = num_decisions
        self.h_space: Union[int, tuple] = h_space

        # train
        self.train_episodes: int = train_episodes
        self.discount: float = discount
        self.steps: int = steps
        self.obs_capacity: int = obs_capacity
        self.prob_thr: float = prob_thr
        self.entropy: float = entropy
        self.e_greedy: float = e_greedy
        self.lr: Union[float, tuple] = lr
        self.grad_clip: float = grad_clip

        # test
        self.test_episodes: int = test_episodes
        self.enable_eval_agents: bool = enable_eval_agents
        self.eval_agents: list = eval_agents

        # keys for logging
        self.loss_key: str = loss_key
        self.act_loss_key: str = act_loss_key
        self.crt_loss_key: str = crt_loss_key
        self.actions_key: str = actions_key
        self.reward_key: str = reward_key
        self.elo_key: str = elo_key
        self.log_avg: int = log_avg

        # device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.ROOT = ROOT if ROOT is not None else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.SAVE_PATH = SAVE_PATH if SAVE_PATH is not None\
            else os.path.join(self.ROOT, 'checkpoints', self.experiment_name + f'_{str(time.time())}')
