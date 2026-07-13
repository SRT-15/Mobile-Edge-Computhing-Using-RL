import os
import csv
import time
import argparse
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import ALGOS, DATASETS, DEFAULT_ALGO, DEFAULT_DATASET, set_global_seeds, build_agent
from Model import NUMBER
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def _build_action_random(state, env):
    n = env.n_agents
    a = np.zeros((n, env.action_size), dtype=float)
    for i in range(n):
        cur_ch = int(state[i][0])
        a[i][0] = np.random.randint(0, 2)
        a[i][1] = np.random.randint(0, 4)
        if int(a[i][1]) == cur_ch:
            a[i][1] = (cur_ch + 1) % 4
        a[i][2] = np.random.uniform(0.01, 0.99)
        a[i][3] = np.random.uniform(env.action_lower_bound[3], env.action_higher_bound[3])
        a[i][4] = np.random.uniform(env.action_lower_bound[4], env.action_higher_bound[4])
        a[i][5] = np.random.uniform(env.action_lower_bound[5], env.action_higher_bound[5])
    return a

def _build_action_fifo(state, env, cursor):
    n = env.n_agents
    a = np.zeros((n, env.action_size), dtype=float)
    for i in range(n):
        cur_ch = int(state[i][0])
        offload = i == cursor
        a[i][0] = 1 if offload else 0
        a[i][1] = (cur_ch + 1) % 4 if offload else cur_ch
        a[i][2] = 0.5
        a[i][3] = env.action_higher_bound[3]
        a[i][4] = 0.5
        a[i][5] = 18.0 if offload else 8.0
    return a

def _build_action_greedy(state, env):
    n = env.n_agents
    a = np.zeros((n, env.action_size), dtype=float)
    gains = state[:, 2]
    powers = state[:, 1]
    sizes = state[:, 3]
    ddls = state[:, 6]
    score = gains * powers / np.maximum(0.1, sizes)
    thresh = float(np.median(score))
    for i in range(n):
        cur_ch = int(state[i][0])
        offload = score[i] >= thresh or ddls[i] < 0.9
        a[i][0] = 1 if offload else 0
        a[i][1] = (cur_ch + 1) % 4 if offload else cur_ch
        a[i][2] = 0.8 if offload else 0.35
        a[i][3] = env.action_higher_bound[3] if not offload else 1.4
        a[i][4] = _clip(0.7 if ddls[i] < 1.0 else 0.5, env.action_lower_bound[4], env.action_higher_bound[4])
        a[i][5] = _clip(20.0 if offload else 10.0, env.action_lower_bound[5], env.action_higher_bound[5])
    return a

def _run_baseline(algo, env, episodes, on_episode=None, should_stop=None, log=print):
    episode_list = []
    reward_list = []
    fifo_cursor = 0
    for ep in range(1, episodes + 1):
        if should_stop is not None and should_stop():
            log(f'  [train] stop requested at episode {ep - 1}')
            break
        state = env.reset()
        done = False
        step_rewards = []
        while not done:
            if algo == 'random':
                action = _build_action_random(state, env)
            elif algo == 'greedy':
                action = _build_action_greedy(state, env)
            else:
                action = _build_action_fifo(state, env, fifo_cursor)
                fifo_cursor = (fifo_cursor + 1) % env.n_agents
            (state, reward, done, *_) = env.step(action)
            step_rewards.append(float(np.mean(reward)))
        ep_reward = float(np.mean(step_rewards)) if step_rewards else 0.0
        episode_list.append(ep)
        reward_list.append(ep_reward)
        if on_episode is not None:
            on_episode(ep, ep_reward)
        log(f'  [train] {algo} episode {ep}  mean_reward {ep_reward:.3f}')
    return (episode_list, reward_list)

def make_env(dataset, n_agents, seed):
    if dataset == 'synthetic':
        from env import MecBCEnv
        return MecBCEnv(n_agents=n_agents, S_DDL=1)
    from env_trace import MecBCEnvTrace
    return MecBCEnvTrace(n_agents=n_agents, dataset=dataset, seed=seed, S_DDL=1)

def run_training(algo=DEFAULT_ALGO, dataset=DEFAULT_DATASET, episodes=50, seed=0, n_agents=None, on_episode=None, should_stop=None, log=print):
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f'algo must be one of {ALGOS}')
    if dataset not in DATASETS:
        raise ValueError(f'dataset must be one of {DATASETS}')
    n_agents = n_agents or NUMBER
    set_global_seeds(seed)
    env = make_env(dataset, n_agents, seed)
    t0 = time.time()
    if algo in ('random', 'greedy', 'fifo'):
        (episodes_list, rewards_list) = _run_baseline(algo=algo, env=env, episodes=episodes, on_episode=on_episode, should_stop=should_stop, log=log)
        episodes_done = len(episodes_list)
    else:
        agent = build_agent(algo, env)
        last_reported = 0
        while agent.n_episodes < episodes:
            if should_stop is not None and should_stop():
                log(f'  [train] stop requested at episode {agent.n_episodes}')
                break
            agent.interact()
            if agent.n_episodes >= 0:
                agent.train()
            while len(agent.mean_rewards) > last_reported:
                ep = agent.episodes[last_reported]
                rew = float(agent.mean_rewards[last_reported])
                last_reported += 1
                if on_episode is not None:
                    on_episode(ep, rew)
                log(f'  [train] {algo}/{dataset} episode {ep}  mean_reward {rew:.3f}')
        episodes_list = list(agent.episodes)
        rewards_list = [float(x) for x in agent.mean_rewards]
        episodes_done = int(agent.n_episodes)
    meta = {'algo': algo, 'dataset': dataset, 'episodes_target': episodes, 'episodes_done': episodes_done, 'seed': seed, 'n_agents': n_agents, 'runtime_s': round(time.time() - t0, 2), 'state_size': env.state_size, 'action_size': env.action_size, 'comparable_to_synthetic': dataset == 'synthetic'}
    return (episodes_list, rewards_list, meta)

def save_results(episodes, rewards, meta, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{meta['algo']}_{meta['dataset']}_seed{meta['seed']}"
    csv_path = os.path.join(out_dir, f'{tag}.csv')
    png_path = os.path.join(out_dir, f'{tag}.png')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'mean_reward'])
        for (e, r) in zip(episodes, rewards):
            w.writerow([e, r])
    plt.figure()
    plt.plot(episodes, rewards, marker='o', ms=3)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    title = f"{meta['algo'].upper()} / {meta['dataset']}"
    if not meta['comparable_to_synthetic']:
        title += '  (NOT comparable to synthetic curve)'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    return (csv_path, png_path)

def _parse_csv_list(value):
    return [x.strip() for x in str(value).split(',') if x.strip()]

def _mean_std(values):
    if not values:
        return (float('nan'), float('nan'))
    m = float(np.mean(values))
    s = float(np.std(values))
    return (m, s)

def run_benchmark(bench_algos, dataset, episodes, seeds, n_agents=None, out_dir=RESULTS_DIR, last_k=5):
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    print(f'  [bench] algos={bench_algos} dataset={dataset} episodes={episodes} seeds={seeds}')
    for algo in bench_algos:
        for seed in seeds:
            try:
                (eps, rews, meta) = run_training(algo=algo, dataset=dataset, episodes=episodes, seed=seed, n_agents=n_agents, log=lambda *_: None)
                final_reward = float(np.mean(rews[-last_k:])) if rews else float('nan')
                best_reward = float(np.max(rews)) if rews else float('nan')
                runs.append({'algo': algo, 'seed': int(seed), 'dataset': dataset, 'episodes_target': int(episodes), 'episodes_done': int(meta['episodes_done']), 'points': int(len(rews)), 'final_reward_last_k': final_reward, 'best_reward': best_reward, 'runtime_s': float(meta['runtime_s']), 'status': 'ok', 'error': ''})
                print(f'  [bench] ok {algo}/seed{seed} final(last{last_k})={final_reward:.3f}')
            except Exception as e:
                runs.append({'algo': algo, 'seed': int(seed), 'dataset': dataset, 'episodes_target': int(episodes), 'episodes_done': 0, 'points': 0, 'final_reward_last_k': float('nan'), 'best_reward': float('nan'), 'runtime_s': float('nan'), 'status': 'error', 'error': repr(e)})
                print(f'  [bench] error {algo}/seed{seed}: {e!r}')
    tag = f"benchmark_{dataset}_ep{episodes}_seeds{'-'.join((str(s) for s in seeds))}"
    runs_csv = os.path.join(out_dir, f'{tag}_runs.csv')
    summary_csv = os.path.join(out_dir, f'{tag}_summary.csv')
    run_fields = ['algo', 'seed', 'dataset', 'episodes_target', 'episodes_done', 'points', 'final_reward_last_k', 'best_reward', 'runtime_s', 'status', 'error']
    with open(runs_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for r in runs:
            w.writerow(r)
    summary_rows = []
    for algo in bench_algos:
        rows = [r for r in runs if r['algo'] == algo and r['status'] == 'ok']
        finals = [r['final_reward_last_k'] for r in rows if not math.isnan(r['final_reward_last_k'])]
        bests = [r['best_reward'] for r in rows if not math.isnan(r['best_reward'])]
        runtimes = [r['runtime_s'] for r in rows if not math.isnan(r['runtime_s'])]
        done_eps = [r['episodes_done'] for r in rows]
        (fm, fs) = _mean_std(finals)
        (bm, bs) = _mean_std(bests)
        (rm, rs) = _mean_std(runtimes)
        (dm, ds) = _mean_std(done_eps)
        summary_rows.append({'algo': algo, 'runs_ok': len(rows), 'runs_total': len([r for r in runs if r['algo'] == algo]), 'final_reward_last_k_mean': fm, 'final_reward_last_k_std': fs, 'best_reward_mean': bm, 'best_reward_std': bs, 'runtime_s_mean': rm, 'runtime_s_std': rs, 'episodes_done_mean': dm, 'episodes_done_std': ds})
    summary_fields = ['algo', 'runs_ok', 'runs_total', 'final_reward_last_k_mean', 'final_reward_last_k_std', 'best_reward_mean', 'best_reward_std', 'runtime_s_mean', 'runtime_s_std', 'episodes_done_mean', 'episodes_done_std']
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    return (runs_csv, summary_csv)

def main():
    ap = argparse.ArgumentParser(description='Combined MEC-DRL trainer')
    ap.add_argument('--algo', default=DEFAULT_ALGO, choices=ALGOS)
    ap.add_argument('--dataset', default=DEFAULT_DATASET, choices=DATASETS)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-agents', type=int, default=None)
    ap.add_argument('--out', default=RESULTS_DIR)
    ap.add_argument('--benchmark', action='store_true', help='Run multiple algorithms across multiple seeds and export a comparison table')
    ap.add_argument('--bench-algos', default=','.join(ALGOS), help='Comma-separated algos for benchmark mode')
    ap.add_argument('--seeds', default='0,1,2', help='Comma-separated seeds for benchmark mode')
    ap.add_argument('--last-k', type=int, default=5, help='Final-score window size for benchmark summary')
    args = ap.parse_args()
    if args.benchmark:
        bench_algos = _parse_csv_list(args.bench_algos)
        bad_algos = [a for a in bench_algos if a not in ALGOS]
        if bad_algos:
            raise ValueError(f'unknown benchmark algos {bad_algos}; choose from {ALGOS}')
        seeds = [int(s) for s in _parse_csv_list(args.seeds)]
        (runs_csv, summary_csv) = run_benchmark(bench_algos=bench_algos, dataset=args.dataset, episodes=args.episodes, seeds=seeds, n_agents=args.n_agents, out_dir=args.out, last_k=args.last_k)
        print(f'  [bench] wrote {runs_csv}')
        print(f'  [bench] wrote {summary_csv}')
        return
    if args.dataset != 'synthetic':
        print('  [train] WARNING: trace-driven mode -- reward is NOT numerically comparable to the synthetic curve (only the channel-gain provenance changes; see env_trace.py).')
    (episodes, rewards, meta) = run_training(algo=args.algo, dataset=args.dataset, episodes=args.episodes, seed=args.seed, n_agents=args.n_agents)
    (csv_path, png_path) = save_results(episodes, rewards, meta, out_dir=args.out)
    print(f'  [train] done: {meta}')
    print(f'  [train] wrote {csv_path}')
    print(f'  [train] wrote {png_path}')
if __name__ == '__main__':
    main()
