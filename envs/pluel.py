import numpy as np


class PluelEnv:
    """
    'duel' between N opponents / pluel - 'duel à plusieurs'
    """

    STRIKE_ID: int = 0
    EVADE_ID: int = 1

    def __init__(self, players: int, rewards: list, batch_size: int = 1, debug: bool = False):
        self.players = players
        self.batch_size = batch_size
        self.debug = debug

        # self.obs_spaces = np.zeros((self.batch_size, 2, players, (players + 1)), dtype='float32')  # one-hot enco

        actions_for_player = 2  # strike/evade
        self.states = players * actions_for_player
        # todo
        self.actions = (players - 1, actions_for_player)

        self.obs_spaces = np.zeros((self.batch_size, self.actions[-1], players, players)) + np.eye(players)  # one-hot
        self.obs_spaces = self.obs_spaces.astype('float32')
        self.rewards = {k: r for k, r in zip(['loss', 'win'], rewards)}

    def reset(self):
        return self.obs_spaces

    def step(self, action: np.ndarray):
        """
        Нам нужно лишь проверить, что на агента не было совершенно успешное нападение.
        Если он не смог защититься, то reward[agent_id] = -1
        """
        action_shape = (self.batch_size, self.players, self.players)

        is_defeat = np.any(action[:, self.EVADE_ID] < action[:, self.STRIKE_ID].transpose((0, 2, 1)), axis=-1)
        defeat_players = is_defeat.sum(-1, keepdims=True)

        rewards = np.zeros(is_defeat.shape, dtype='float32')
        non_zero_criteria = np.all((self.players > defeat_players, defeat_players > 0), axis=(0, -1))

        loss_players = is_defeat[non_zero_criteria] / defeat_players[non_zero_criteria]
        win_players = ~is_defeat[non_zero_criteria] / (self.players - defeat_players[non_zero_criteria])

        rewards[non_zero_criteria] = loss_players * self.rewards['loss'] + win_players * self.rewards['win']

        # --> for states (N, Lose, Win)
        # player_state = 2 - is_defeat  # 2 - won, 1 - lost
        # obs = (np.arange(self.states) == player_state[..., None]).astype(np.float32)  # one_hot_states

        return action.astype('float32'), rewards

        # zeros = np.zeros((self.batch_size, self.players, 1), dtype=np.int32)
        # obs = np.concatenate((action[:, self.OFFEND_ID], zeros, action[:, self.DEFEND_ID], zeros),
        #                      axis=-1, dtype=np.float32)
