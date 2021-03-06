import gym
import numpy as np


class OADEnv(gym.Env):
    """
    Игроки должны выбрать кого атаковать и от кого защищаться.
    """
    OFFEND_ID: int = 0
    DEFEND_ID: int = 1
    WIN_REWARD: float = 1.
    LOSS_REWARD: float = -1.

    def __init__(self, players: int, batch_size: int, debug: bool = False):
        self.players = players
        self.batch_size = batch_size
        self.debug = debug

        # gym specific
        self.observation_space = gym.spaces.MultiBinary((players, 2 * (players + 1)))  # one-hot enco
        self.action_space = gym.spaces.Box(0, 1, (self.players - 1,), np.int)  # attack/offend
        self.reward_range = (self.LOSS_REWARD, self.WIN_REWARD)

    def reset(self, **kwargs):
        # obs = np.zeros(self.args.traj_batch_size, dtype=np.int32)
        # obs = to_onehot(obs, dim=5)
        # return obs

        obs = np.zeros((self.batch_size, *self.observation_space.shape), dtype=np.float32)
        obs[..., (self.players, self.observation_space.shape[-1] - 1)] = 1.
        return obs

    def step(self, action: np.ndarray):
        """
        Нам нужно лишь проверить, что на агента не было совершенно успешное нападение.
        Если он не смог защититься, то reward[agent_id] = -1
        """
        action_shape = (self.batch_size, self.players, self.players)
        self._print('\nНовый раунд: ')
        rewards = np.sum(np.array(
            action[:, self.OFFEND_ID] > action[:, self.DEFEND_ID].T.reshape(action_shape)
        ), axis=-1, dtype=np.float32)
        rewards[rewards > 0] = self.LOSS_REWARD / max(rewards[rewards > 0].size, 1)
        rewards[rewards == 0] = self.WIN_REWARD / max(rewards[rewards == 0].size, 1)

        zeros = np.zeros((self.batch_size, self.players, 1), dtype=np.int32)

        obs = np.concatenate((action[:, self.OFFEND_ID], zeros, action[:, self.DEFEND_ID], zeros),
                             axis=-1, dtype=np.float32)

        self._print(f'Награды {rewards}')

        return obs, rewards

    def render(self, mode="human"):
        pass

    def _print(self, text):
        if self.debug:
            print(text)


if __name__ == '__main__':
    env = OADEnv(players=3, debug=True)
    print(env.reset())
    print(env.step([[1, 2], [0, 0], [0, 0]]))
    print(env.step([[0, 0], [1, 1], [2, 2]]))
