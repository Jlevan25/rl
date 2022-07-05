import os
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from executors.epoch_manager import EpochManager
from configs import Config
from models import ActorCritic, SimpleAC
from losses import GAE, WISLoss
from agents import A2CAgent, DummyAgent, A2CAgentOPE, A2CAgentSwim
from envs import PluelEnv
from schedulers import ExplorationScheduler

# from metrics import BalancedAccuracy
# from utils import split_params4weight_decay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = Config(agents=3, players=3, device='cpu',
             batch_size=1, lr=1e-3, matrix_game=False,
             train_episodes=10000, test_episodes=1000,
             epochs=150, exploration_epochs=1,
             ROOT=ROOT,
             debug=True, show_each=1000, seed=1)

keys = train_key, inference_key = 'train', 'inference'

if cfg.seed is not None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

env = PluelEnv(cfg.players, rewards=[-1, 1], batch_size=cfg.batch_size, debug=cfg.debug)

agents_model = [SimpleAC(env.states, env.actions, hidden_n=cfg.h_space, obs_capacity=cfg.obs_capacity)
                for _ in range(cfg.agents)]

peers_model = [SimpleAC(env.states, env.actions, hidden_n=cfg.h_space, obs_capacity=cfg.obs_capacity)
               for _ in range(cfg.players - cfg.agents)]

checkpoint = torch.load(os.path.join(cfg.ROOT, 'checkpoints',
                                     r'pluel_agents3_players_3_1656244188.069767\epoch87\models',
                                     'agent0_A2CAgentOPE.pth'), map_location=torch.device(cfg.device))

swim_agent = A2CAgentSwim(id_=0, model=agents_model[0], checkpoint=checkpoint,
                          obs_spaces_shape=env.obs_spaces.shape, cfg=cfg)

agents = [swim_agent] + [A2CAgent(id_=i + 1, model=model, obs_spaces_shape=env.obs_spaces.shape, cfg=cfg)
                         for i, model in enumerate(agents_model[1:])]

peers = [A2CAgent(id_=i + cfg.agents, model=model, obs_spaces_shape=env.obs_spaces.shape, cfg=cfg).eval()
         for i, model in enumerate(peers_model)]

swim_optim = optim.Adam(params=swim_agent.parameters(), lr=cfg.lr)
optimizers = [swim_optim] + [optim.Adam(params=model.parameters(), lr=cfg.lr) for model in agents_model[1:]]

criteria = [WISLoss(agents[:1], window_size=cfg.steps), GAE(agents[1:])]

writer = SummaryWriter(log_dir=os.path.join(ROOT, 'ope_choose_logs', str(time.time())))
# writer = None
metrics = []
metrics_dict = {train_key: metrics, inference_key: metrics}

exploration_scheduler = ExplorationScheduler(agents, epsilon=cfg.e_greedy, end_step=cfg.exploration_epochs)

epoch_manager = EpochManager(env=env, cfg=cfg,
                             agents=agents,
                             peers=peers,
                             optimizers=optimizers,
                             criteria=criteria,
                             exploration_scheduler=exploration_scheduler,
                             writer=writer, metrics=metrics_dict)

epochs = cfg.epochs
save_each = 5

train = True

for epoch in range(epochs):
    epoch_manager.train(train_key, epoch)
    epoch_manager.inference(inference_key, epoch)
    if (epoch+1) % save_each == 0:
        epoch_manager.save_models(epoch)
