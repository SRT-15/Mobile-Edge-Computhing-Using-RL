import os
import glob
import numpy as np
from env import MIN_GAIN, MAX_GAIN
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATASET_DIRS = {'DCU': os.path.join(DATA_ROOT, 'DCU'), 'KAIST': os.path.join(DATA_ROOT, 'KAIST'), 'DCU_Easy': os.path.join(DATA_ROOT, 'DCU_Easy'), 'KAIST_Easy': os.path.join(DATA_ROOT, 'KAIST_Easy')}

def dataset_dir(dataset):
    if dataset not in DATASET_DIRS:
        raise ValueError(f'unknown dataset {dataset!r}; choose from {sorted(DATASET_DIRS)}')
    return DATASET_DIRS[dataset]

def list_trace_files(dataset):
    d = dataset_dir(dataset)
    files = sorted(glob.glob(os.path.join(d, '*.txt')))
    return files

def load_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.asarray(rows, dtype=np.float64)

class MobilityAdapter:

    def __init__(self, dataset, n_agents, seed=0):
        self.dataset = dataset
        self.n_agents = n_agents
        self.files = list_trace_files(dataset)
        if len(self.files) < n_agents:
            raise ValueError(f'dataset {dataset!r} has {len(self.files)} traces, need at least n_agents={n_agents}')
        rng = np.random.RandomState(seed)
        chosen = rng.choice(len(self.files), size=n_agents, replace=False)
        self.traces = [load_trace(self.files[i]) for i in chosen]
        self.chosen_files = [os.path.basename(self.files[i]) for i in chosen]
        allxy = np.vstack([t[:, 1:3] for t in self.traces])
        self.bs = allxy.mean(axis=0)
        d = np.sqrt(((allxy - self.bs) ** 2).sum(axis=1))
        self.dmax = float(d.max()) if d.max() > 0 else 1.0

    def position(self, agent_id, step):
        tr = self.traces[agent_id]
        idx = min(step, len(tr) - 1)
        return (tr[idx, 1], tr[idx, 2])

    def distance_to_bs(self, agent_id, step):
        (x, y) = self.position(agent_id, step)
        return float(np.hypot(x - self.bs[0], y - self.bs[1]))

    def gain_for(self, agent_id, step):
        dnorm = min(self.distance_to_bs(agent_id, step) / self.dmax, 1.0)
        return MAX_GAIN - (MAX_GAIN - MIN_GAIN) * dnorm

    def gains(self, step):
        return np.array([self.gain_for(n, step) for n in range(self.n_agents)], dtype=np.float64)

    def diagnostics(self):
        out = []
        for (i, tr) in enumerate(self.traces):
            xy = tr[:, 1:3]
            disp = np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1)) if len(xy) > 1 else np.zeros(1)
            out.append({'file': self.chosen_files[i], 'points': int(len(tr)), 'mean_step_disp': float(disp.mean()), 'max_step_disp': float(disp.max()), 'bbox_x': (float(xy[:, 0].min()), float(xy[:, 0].max())), 'bbox_y': (float(xy[:, 1].min()), float(xy[:, 1].max())), 'mean_dist_to_bs': float(np.mean([np.hypot(px - self.bs[0], py - self.bs[1]) for (px, py) in xy]))})
        return out
