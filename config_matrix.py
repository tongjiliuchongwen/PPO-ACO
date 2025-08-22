# config_matrix.py - 基于信息素矩阵的配置

# ========= 训练流程与元任务设定 =========
TOTAL_MAJOR_RESETS = 15
ITERATIONS_PER_RESET = 300
META_EPISODES_PER_TASK = 4
SAVE_FREQUENCY = 2

TOTAL_ITERATIONS = TOTAL_MAJOR_RESETS * ITERATIONS_PER_RESET

# ========= 信息素矩阵参数 =========
PHEROMONE_MATRIX_SIZE = 7        # n*n 矩阵大小 (7x7=49个值)
PHEROMONE_MATRIX_RANGE = 3.0     # 矩阵覆盖的物理范围 (±3.0 单位)

# ========= 环境参数 =========
GRID_SIZE = 50
ENV_BOUNDS = 8.0                 # 恢复到中等难度
MAX_STEPS_PER_EPISODE = 200

# ========= 物理参数 =========
V0 = 1.2
OMEGA_MAX = 2.0
DT = 0.1
TARGET_RADIUS = 0.8              # 中等大小的目标
ENABLE_NOISE = True
TRANSLATIONAL_NOISE_STD = 0.08
ROTATIONAL_NOISE_STD = 0.12

# ========= 障碍物设置 =========
OBSTACLES = [
    (3.0, 4.0, 1.0),
    (-4.0, -3.0, 1.2),
    (1.0, -6.0, 0.8)
]

# ========= PPO 超参数 =========
TIMESTEPS_PER_BATCH = 1024
N_UPDATES_PER_ITERATION = 4
LEARNING_RATE = 5e-5
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP = 0.2
ENTROPY_COEF = 0.015
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# ========= ACO 参数 =========
NAV_EVAPORATION_RATE = 0.05
NAV_DEPOSIT_AMOUNT = 100.0
EXP_EVAPORATION_RATE = 0.3
EXP_DEPOSIT_AMOUNT = 1.0

ALPHA_PPO_PREFERENCE = 0.6
BETA_ACO_GUIDANCE = 0.0          # 初始关闭
ACO_KERNEL_SIZE = 5

# ========= 奖励参数 =========
STEP_PENALTY = -0.03
TARGET_REWARD = 30.0
COLLISION_PENALTY = -8.0
BOUNDARY_PENALTY = -5.0
DISTANCE_SHAPING_COEF = 2.5      # 强距离塑形

NAV_PHEROMONE_SCALING = 0.0
EXP_PHEROMONE_SCALING = 0.0

# ========= 候选动作集合 =========
CANDIDATE_OFFSETS = [-0.3, 0.0, 0.3]
ACO_GUIDANCE_EVERY = 5

# ========= 设备/服务 =========
DASHBOARD_REFRESH_EVERY = 10