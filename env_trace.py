import numpy as np
from env import MecBCEnv
from mobility_adapter import MobilityAdapter
GAIN_INDEX = 2

class MecBCEnvTrace(MecBCEnv):

    def __init__(self, n_agents, dataset, seed=0, **kwargs):
        self.adapter = MobilityAdapter(dataset, n_agents, seed=seed)
        self.dataset = dataset
        self._t = 0
        super().__init__(n_agents=n_agents, **kwargs)

    def _apply_trace_gain(self, state, step):
        gains = self.adapter.gains(step)
        self.S_gain = gains.copy()
        state = np.array(state, dtype=np.float64)
        state[:, GAIN_INDEX] = gains
        return state

    def reset(self):
        self._t = 0
        state = super().reset()
        return self._apply_trace_gain(state, self._t)

    def step(self, action):
        self._t += 1
        out = super().step(action)
        state = out[0]
        state = self._apply_trace_gain(state, self._t)
        return (state,) + tuple(out[1:])
