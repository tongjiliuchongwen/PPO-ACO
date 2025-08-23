# config.py - RL Navigation Platform V3 Central Configuration
# This file controls all experimental variables for the modular research platform

# ==============================================================================
# --- 1. EXPERIMENT MODE SELECTION (EXPERIMENT_MODE) ---
# ==============================================================================
# 'BASELINE':      Fixed target, relative coordinate observation, distance reward (validate basic ability)
# 'SEARCH_RL':     Random target, relative coordinate observation, sparse reward (validate memory ability)  
# 'SEARCH_HYBRID': Random target, relative coordinate observation, dual pheromone reward (validate collective experience)
EXPERIMENT_MODE = 'BASELINE'

# ==============================================================================
# --- 2. TRAINING AND ENVIRONMENT PARAMETERS ---
# ==============================================================================
# Training flow
TOTAL_MAJOR_RESETS = 20     # Total number of "major resets" (only effective in SEARCH modes)
ITERATIONS_PER_RESET = 200  # Number of "small iterations" per "major reset" cycle
SAVE_FREQUENCY = 5

# Environment
ENV_BOUNDS = 10.0
MAX_STEPS_PER_EPISODE = 250
OBSTACLES = [(2.0, 3.0, 1.0), (-5.0, -5.0, 1.5), (0.0, -7.0, 0.8)]

# Physics
V0 = 1.0
OMEGA_MAX = 2.0
DT = 0.1
TARGET_RADIUS = 0.5
ENABLE_NOISE = True
TRANSLATIONAL_NOISE_STD = 0.1
ROTATIONAL_NOISE_STD = 0.2

# ==============================================================================
# --- 3. AGENT AND ALGORITHM HYPERPARAMETERS (AGENT & ALGORITHM) ---
# ==============================================================================
# Agent architecture ('MLP' or 'LSTM')
AGENT_ARCHITECTURE = 'MLP' 

# PPO algorithm
TIMESTEPS_PER_BATCH = 4096   # Large batch size for RTX 4080 SUPER
N_UPDATES_PER_ITERATION = 10
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP = 0.2
GAE_LAMBDA = 0.95  # GAE lambda parameter

# --- Parameters effective only in 'SEARCH_HYBRID' mode ---
# Dual pheromone system (ACO)
NAV_EVAPORATION_RATE = 0.05
NAV_DEPOSIT_AMOUNT = 100.0
EXP_EVAPORATION_RATE = 0.3
EXP_DEPOSIT_AMOUNT = 1.0
ACO_KERNEL_SIZE = 7

# ==============================================================================
# --- 4. REWARD FUNCTION CONFIGURATION (REWARD_CONFIG) ---
# ==============================================================================
# Reward function type ('DISTANCE_SHAPING' or 'PHEROMONE_SHAPING' or 'SPARSE')
REWARD_FUNCTION_TYPE = 'DISTANCE_SHAPING'

# Reward values
STEP_PENALTY = -0.01
TARGET_REWARD = 100.0
COLLISION_PENALTY = -50.0
BOUNDARY_PENALTY = -10.0

# Reward scaling factors
DISTANCE_SHAPING_SCALING = 10.0  # Only used for 'DISTANCE_SHAPING'
NAV_PHEROMONE_SCALING = 50.0     # Only used for 'PHEROMONE_SHAPING'
EXP_PHEROMONE_SCALING = 20.0     # Only used for 'PHEROMONE_SHAPING'

# ==============================================================================
# --- 5. NETWORK ARCHITECTURE PARAMETERS ---
# ==============================================================================
# MLP parameters
MLP_HIDDEN_SIZES = [64, 64]

# LSTM parameters  
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1

# ==============================================================================
# --- 6. ACO SYSTEM PARAMETERS ---
# ==============================================================================
ACO_GRID_SIZE = 50
ACO_DIFFUSION_RATE = 0.1

# ==============================================================================
# --- 7. GPU OPTIMIZATION PARAMETERS (RTX 4080 SUPER) ---
# ==============================================================================
USE_MIXED_PRECISION = True
PIN_MEMORY = True
ENABLE_CUDNN_BENCHMARK = True
MAX_BATCH_SIZE_GPU = 16384  # Can be adjusted based on available GPU memory

# ==============================================================================
# --- 8. LOGGING AND VISUALIZATION ---
# ==============================================================================
LOG_FREQUENCY = 10
VISUALIZE_TRAINING = True
SAVE_MODELS = True