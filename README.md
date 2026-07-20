# Energy Aware Resource Scheduling in Mobile Edge Computing Using Deep Reinforcement Learning

A comprehensive framework for evaluating multiple multi agent reinforcement learning (RL) algorithms and baseline heuristics for resource scheduling in Mobile Edge Computing (MEC). This project combines trace aware evaluation with RL based schedulers to optimize energy efficiency and minimize latency in edge computing environments.

## Overview

This project implements a unified training and benchmarking pipeline for MEC resource scheduling that includes and compares:
  **RL Algorithms:** MAA2C, MAPPO, MADDPG
  **Baseline Heuristics:** Random, Greedy, FIFO
  **Datasets:** Synthetic (controlled), DCU (trace driven), KAIST (trace driven)

The framework models joint offloading and resource control decisions, evaluates policies under dynamic conditions, and provides a web dashboard for training visualization and benchmark management.

## Key Features

  **Multi Agent RL Support**: Implements MAA2C, MAPPO, and MADDPG for distributed scheduling decisions
  **Realistic Mobility Traces**: Supports trace driven evaluation using real mobility datasets (DCU, KAIST) alongside synthetic scenarios
  **Benchmark Infrastructure**: Automated multi seed benchmarking with CSV output for reproducible research
  **Web Dashboard**: Real time training metrics, topology visualization, and benchmark result tables
  **Baseline Comparisons**: Includes simple heuristics for performance comparison
  **Energy Aware Optimization**: Explicitly optimizes for energy efficiency and delay constraints

## Project Structure

```
├── config.py                    # Configuration: algorithms, datasets, hyperparameters
├── env.py                       # MEC environment and reward definitions
├── train.py                     # Training and benchmarking utilities
├── Model.py                     # Network architecture and model utilities
├── Memory.py                    # Replay buffer implementations
├── mobility_adapter.py          # Dataset loading and mobility trace processing
├── multi_agent_a2c.py          # MAA2C algorithm implementation
├── multi_agent_ppo.py          # MAPPO algorithm implementation
├── multi_agent_ddpg.py         # MADDPG algorithm implementation
├── dashboard_server.py         # Flask web dashboard server
├── dashboard.html              # Frontend for the web dashboard
├── env_trace.py                # Trace based environment wrapper
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── data/                       # Datasets
│   ├── DCU/                    # DCU mobility traces (30 second intervals)
│   ├── DCU_Easy/               # Simplified DCU traces
│   ├── KAIST/                  # KAIST mobility traces
│   └── KAIST_Easy/             # Simplified KAIST traces
├── results/                    # Benchmark outputs and CSV results
└── tests/                      # Unit tests
    ├── test_dashboard_api.py
    ├── test_datasets.py
    └── test_regression_synthetic.py
```

## Installation

### Prerequisites
  Python 3.9+
  pip or conda

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/shrutituteja/Desktop/MEC_DRL_combined
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3  m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install  r requirements.txt
   ```

## Dependencies

The project requires:
  **matplotlib** (≥3.7.0): Plotting and visualization
  **numpy** (≥1.24.0): Numerical computations
  **torch** (≥2.0.0): Deep learning framework
  **Flask** (≥2.3.3): Web server for dashboard
  **Flask SocketIO** (≥5.3.6): Real time communication for dashboard
  **pytest** (≥8.3.2): Testing framework
  **xlrd/xlutils/xlwt** (≥2.0.0): Excel file handling for data processing

## Usage

### 1. Training a Single Agent

Train an agent on a specific dataset using the default configuration:

```bash
python train.py   algo maa2c   dataset synthetic   episodes 200   seed 0
```

**Arguments:**
  `  algo`: Algorithm choice (maa2c, mappo, maddpg, random, greedy, fifo)
  `  dataset`: Dataset choice (synthetic, DCU, KAIST)
  `  episodes`: Number of training episodes (default: 200)
  `  seed`: Random seed for reproducibility (default: 0)

### 2. Running Benchmarks

Execute a multi seed benchmark comparing multiple algorithms:

```bash
python train.py benchmark   dataset DCU   episodes 1320   seeds 0 1 2   algos maa2c mappo maddpg random greedy fifo
```

This generates:
  `benchmark_DCU_ep1320_seeds0 1 2_runs.csv`: Run level results
  `benchmark_DCU_ep1320_seeds0 1 2_summary.csv`: Aggregated summary statistics

### 3. Starting the Web Dashboard

Launch the interactive dashboard for real time training visualization:

```bash
python dashboard_server.py
```

Then open your browser to `http://localhost:5000`

**Features:**
  Real time training curves
  Network topology visualization
  Benchmark result tables
  Algorithm and dataset selection
  Live metrics updates via WebSocket

### 4. Running Tests

Validate the setup with unit tests:

```bash
pytest tests/  v
```

## Configuration

### Algorithm Hyperparameters

Hyperparameters for each algorithm are defined in `config.py`:

| Algorithm | Learning Rate (Actor) | Learning Rate (Critic) | Noise | Tau | Notes |
|           |                      |                      |       |     |       |
| MAA2C     | 0.001               | 0.001               | 0.04  | 1400| Advantage Actor Critic |
| MAPPO     | 0.001               | 0.001               | 0.0   | 300 | Policy Proximal Optimization |
| MADDPG    | 0.001               | 0.001               | N/A   | N/A | Deterministic Policy Gradient |

### Environment Parameters

Key environment settings in `env.py`:

  **Energy weights:** λ_E = 0.6, λ_φ = 0.4
  **Channels:** K_CHANNEL = 4
  **Task parameters:** Size range [0.2, 50] MB, cycle range [0.05, 2] GHz
  **Power budget:** MAX_POWER = 24 W
  **Channel gain range:** [5, 10] dB

## Results Format

Benchmark outputs include CSV files with the following key metrics:

  `final_reward_last_k_mean`: Mean final reward (last k episodes)
  `final_reward_last_k_std`: Standard deviation of final reward
  `runtime_s_mean`: Average runtime per run (seconds)
  `runs_ok / runs_total`: Successful execution ratio

Example benchmark summary:
```
Algorithm, Runs (ok/total), Final Reward Mean, Final Reward Std, Runtime Mean (s)
MAA2C, 2/2,  3.099, 0.318, 61.545
MAPPO, 2/2, 0.242, 0.130, 43.515
MADDPG, 2/2, 0.353, 0.028, 229.345
Random, 2/2,  1.222, 0.076, 14.435
Greedy, 2/2,  0.199, 0.088, 12.635
FIFO, 2/2,  0.425, 0.105, 10.685
```

## System Model

The MEC system consists of:

  **Mobile Devices (Agents):** n agents with energy constraints, executing tasks locally or offloading to edge servers
  **Edge Servers:** Receive offloaded tasks with bandwidth and compute limitations
  **Channel Conditions:** Time varying wireless channels with 4 available channel options per agent
  **Mobility Traces:** Optional trace driven user mobility from DCU or KAIST datasets

### Reward Definition

The reward combines:
1. **Energy Efficiency:** Minimizes total energy consumption (device + transmission + edge processing)
2. **Latency:** Penalizes task completion delays
3. **Success:** Rewards successful task completion within deadline

$$R =  \lambda_E \cdot E   \lambda_\phi \cdot D$$

where E is energy consumed and D is latency.

## Multi Agent Algorithms

### MAA2C (Multi Agent Advantage Actor Critic)
  Policy and value learning with centralized training
  Advantage based policy updates
  Noise injection for exploration

### MAPPO (Multi Agent Proximal Policy Optimization)
  PPO style clipped policy updates
  Stabilized gradient estimates
  Lower runtime compared to MADDPG

### MADDPG (Multi Agent Deterministic Policy Gradient)
  Deterministic policy learning
  Critic based gradient optimization
  Highest final performance but computationally expensive

## Performance Insights

From representative DCU benchmark (1320 episodes, 2 seeds):

  **MADDPG**: Best final reward but ~5× runtime overhead vs MAPPO
  **MAPPO**: Excellent performance runtime tradeoff; recommended for practical deployment
  **MAA2C**: Good stability but lower final rewards than MAPPO/MADDPG
  **Greedy Baseline**: Competitive with RL on short horizons; much lower compute cost
  **FIFO/Random**: Simplest but lowest performance

## Development & Testing

### Run All Tests
```bash
pytest tests/  v   tb=short
```

### Run Specific Test Suite
```bash
pytest tests/test_regression_synthetic.py  v
```

### Run with Coverage Report
```bash
pytest tests/   cov=.   cov report=html
```

## Extending the Framework

### Adding a New Algorithm

1. Create a new file (e.g., `multi_agent_custom.py`)
2. Implement the algorithm class with `train()` and `predict()` methods
3. Update `config.py` to register the algorithm:
   ```python
   ALGOS = (..., 'custom')
   ALGO_HPARAMS['custom'] = dict(...)  # your hyperparams
   ```
4. Add factory logic in `build_agent()` in `config.py`

### Adding a New Dataset

1. Place trace data in `data/<DATASET_NAME>/`
2. Update `DATASETS` in `config.py`
3. Add loading logic in `mobility_adapter.py` if needed
4. Run benchmarks: `python train.py benchmark   dataset <DATASET_NAME> ...`

## Known Limitations & Future Work

  Current implementation supports up to 4 channel options; extend for more flexible channel models
  Dashboard real time updates limited to single training instance at a time
  Trace datasets assume fixed temporal resolution (30 second intervals)
  Future: Distributed training support across multiple machines

## References & Related Work

  Chen et al., "Deep Reinforcement Learning for Offloading and Power Allocation in MEC"
  Mao et al., "Learning Scheduling Algorithms for Data Processing Clusters"
  Schulman et al., "Proximal Policy Optimization Algorithms" (PPO)
  Lillicrap et al., "Deterministic Policy Gradient Methods for Reinforcement Learning" (DDPG)
  Konda & Tsitsiklis, "Actor Critic Algorithms" (AC foundation)

## License

This project is provided as is for research and educational purposes.

## Contact & Support

For issues, questions, or contributions, please refer to the project documentation or contact the authors.

   

**Last Updated:** July 2026  
**Python Version:** 3.9+  
**Status:** Active Development
