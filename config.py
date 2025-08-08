# config.py

# --- 环境参数 ---
GRID_SIZE = 50  # 将连续空间离散化为 50x50 的网格用于信息素
ENV_BOUNDS = 10.0  # 连续空间的边界大小 (-10 到 +10)
MAX_STEPS_PER_EPISODE = 250  # 每回合最大步数

# --- 用户交互：障碍物和噪声 ---
# 用户以(x, y, radius)的元组列表形式定义圆形障碍物
OBSTACLES = [
    (2.0, 3.0, 1.0),
    (-5.0, -5.0, 1.5),
    (0.0, -7.0, 0.8)
]
# 用户控制噪声强度的开关和大小
ENABLE_NOISE = True
TRANSLATIONAL_NOISE_STD = 0.1  # 平动噪声标准差 (对应 sqrt(2*D0*dt))
ROTATIONAL_NOISE_STD = 0.2   # 旋转噪声标准差 (对应 sqrt(2*D_theta*dt))

# --- PPO 算法超参数 ---
TIMESTEPS_PER_BATCH = 2048
N_UPDATES_PER_ITERATION = 10
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP = 0.2

# --- ACO 算法超参数 ---
EVAPORATION_RATE = 0.05  # 信息素蒸发率
DEPOSIT_AMOUNT = 100.0   # 每次成功路径投放的信息素总量

# --- 混合算法参数 ---
ALPHA_Q_VALUE = 0.7  # PPO策略网络输出的权重
BETA_PHEROMONE = 0.3  # 信息素的权重

# --- 物理参数 ---
V0 = 1.0  # 智能体基础速度
OMEGA_MAX = 2.0  # 最大角速度
DT = 0.1  # 时间步长
TARGET_RADIUS = 0.5  # 到达目标的判定半径

# --- 训练参数 ---
TOTAL_ITERATIONS = 1000  # 总训练迭代次数
SAVE_FREQUENCY = 100  # 模型保存频率
RENDER_FREQUENCY = 50  # 可视化频率

# --- 奖励函数参数 ---
STEP_PENALTY = -0.01  # 每步的时间惩罚
TARGET_REWARD = 100.0  # 到达目标的奖励
COLLISION_PENALTY = -50.0  # 碰撞障碍物的惩罚