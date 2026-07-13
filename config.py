import random
import numpy as np
ALGOS = ('maa2c', 'mappo', 'maddpg', 'random', 'greedy', 'fifo')
DEFAULT_ALGO = 'maa2c'
DATASETS = ('synthetic', 'DCU', 'KAIST')
DEFAULT_DATASET = 'synthetic'
ALGO_HPARAMS = {'maa2c': dict(noise=0.04, tau=1400, bound=600, actor_lr=0.001, critic_lr=0.001), 'mappo': dict(noise=0, tau=300, actor_lr=0.001, critic_lr=0.001), 'maddpg': dict(actor_lr=0.001, critic_lr=0.001)}

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

def build_agent(algo, env):
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f'unknown algo {algo!r}; choose from {ALGOS}')
    hp = ALGO_HPARAMS[algo]
    common = dict(env=env, n_agents=env.n_agents, state_dim=env.state_size, action_dim=env.action_size, action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound, actor_lr=hp['actor_lr'], critic_lr=hp['critic_lr'])
    if algo == 'maa2c':
        from multi_agent_a2c import MAA2C
        return MAA2C(noise=hp['noise'], bound=hp['bound'], tau=hp['tau'], **common)
    if algo == 'mappo':
        from multi_agent_ppo import MAPPO
        return MAPPO(env=env, state_dim=env.state_size, action_dim=env.action_size, n_agents=env.n_agents, action_lower_bound=env.action_lower_bound, action_higher_bound=env.action_higher_bound, noise=hp['noise'], tau=hp['tau'], actor_lr=hp['actor_lr'], critic_lr=hp['critic_lr'])
    from multi_agent_ddpg import MADDPG
    return MADDPG(**common)
