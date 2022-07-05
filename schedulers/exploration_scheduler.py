from abc import ABC


class ExplorationScheduler(ABC):
    def __init__(self, agents, epsilon, end_step):
        self._epsilon = epsilon
        self.agents = agents
        self.end_step = end_step
        self._step = -1
        self._current_epsilon = 1. if end_step > 0 else epsilon

    def state_dict(self):
        # todo
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'agents'}

    def load_state_dict(self, state_dict):
        # todo
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        self._step += 1

        if self._step == self.end_step:
            self._current_epsilon = self._epsilon

        if self._step in [0, self.end_step]:
            for agent in self.agents:
                agent.e_greedy = self._current_epsilon

    def get_epsilon(self):
        return self._epsilon


class EpsilonGreedyStepSchedular(ExplorationScheduler):

    def __init__(self, agents, epsilon, each_step, end_step):
        super().__init__(agents, epsilon, end_step)

    def step(self):
        pass

    def get_epsilon(self):
        pass
