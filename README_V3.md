# RL Navigation Platform V3 - Quick Start Guide

## Overview
This is a modular, configurable research platform for studying navigation with reinforcement learning. It addresses three core scientific questions through easy configuration switches.

## Core Scientific Questions

1. **BASELINE**: What is the performance of standard DRL agents (PPO+MLP) in fixed-target navigation?
2. **ROLE OF MEMORY**: How much does short-term memory (LSTM) improve sample efficiency in blind search tasks?
3. **VALUE OF COLLECTIVE EXPERIENCE**: Can ACO pheromone systems accelerate learning for memory-enabled agents?

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision gymnasium matplotlib numpy pillow scipy
```

### 2. Run Different Experiments

#### Baseline Experiment (Fixed Target + MLP)
Edit `config.py`:
```python
EXPERIMENT_MODE = 'BASELINE'
AGENT_ARCHITECTURE = 'MLP'
REWARD_FUNCTION_TYPE = 'DISTANCE_SHAPING'
```
Run: `python main.py`

#### Memory Importance (Random Target + LSTM)
Edit `config.py`:
```python
EXPERIMENT_MODE = 'SEARCH_RL'
AGENT_ARCHITECTURE = 'LSTM'
REWARD_FUNCTION_TYPE = 'SPARSE'
```
Run: `python main.py`

#### Collective Experience (Random Target + LSTM + ACO)
Edit `config.py`:
```python
EXPERIMENT_MODE = 'SEARCH_HYBRID'
AGENT_ARCHITECTURE = 'LSTM'
REWARD_FUNCTION_TYPE = 'PHEROMONE_SHAPING'
```
Run: `python main.py`

### 3. View Results
- Training plots: `experiments/{mode}_{arch}_{timestamp}/`
- Saved models: `experiments/{mode}_{arch}_{timestamp}/`
- Configuration logs: `experiments/{mode}_{arch}_{timestamp}/config.txt`

## Platform Files

- `config.py` - Central configuration (modify this to switch experiments)
- `environment.py` - Active particle navigation environment
- `network.py` - Unified MLP/LSTM actor-critic networks
- `aco_system.py` - Dual pheromone ACO system
- `ppo_agent.py` - PPO algorithm with LSTM support
- `main.py` - Training loop with GPU optimizations
- `demo.py` - Configuration demonstration
- `test_platform.py` - Platform verification tests

## Key Features

- **Modular Design**: Change experiments through config.py only
- **Dual Architecture**: Seamless MLP ↔ LSTM switching
- **GPU Optimized**: Mixed precision, large batches, cuDNN tuning
- **Scientific Rigor**: Proper experiment isolation and reproducibility
- **Easy Comparison**: Compare architectures, reward functions, ACO guidance

## GPU Optimizations (RTX 4080 SUPER)

The platform includes optimizations for high-end GPUs:
- Mixed precision training (1.5-3x speedup)
- Large batch sizes (4096+ timesteps)
- cuDNN benchmarking for optimal algorithms
- Pinned memory for faster data transfer

Adjust `TIMESTEPS_PER_BATCH` in config.py based on your GPU memory.

## Experiment Comparison

| Experiment | Mode | Architecture | Reward | Purpose |
|------------|------|--------------|--------|---------|
| Baseline | BASELINE | MLP | Distance | Test basic navigation |
| Memory | SEARCH_RL | LSTM | Sparse | Test memory importance |
| Collective | SEARCH_HYBRID | LSTM | Pheromone | Test collective experience |

## Architecture Comparison

| Component | MLP | LSTM |
|-----------|-----|------|
| Parameters | ~4.6K | ~33.7K |
| Memory | None | Hidden states |
| Use Case | Fixed targets | Random targets |
| Performance | Fast | Better generalization |

Run `python demo.py` to see detailed configuration examples.