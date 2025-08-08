# PPO-ACO Hybrid Algorithm Framework

一个完整的、模块化的、可交互的、可视化的Python框架，实现了PPO（Proximal Policy Optimization）和ACO（Ant Colony Optimization）的混合算法，用于训练智能体在包含障碍物的二维连续空间中导航到目标位置。

## 🚀 核心特性

- **PPO算法**: 深度强化学习算法，训练智能体进行实时局部决策
- **ACO系统**: 蚁群优化算法，提供基于历史经验的全局路径引导
- **朗之万动力学**: 实现真实的微观粒子物理运动模型，包含噪声
- **障碍物避障**: 支持用户自定义的圆形障碍物
- **实时可视化**: 信息素热力图、轨迹动画、训练统计图表
- **模块化设计**: 清晰的代码结构，易于扩展和修改

## 📁 项目结构

```
/ppo_aco_project/
├── main.py             # 主执行文件，负责训练和测试流程的调度
├── config.py           # 配置文件，用于用户交互，设置所有超参数和环境参数
├── environment.py      # 定义 ActiveParticleEnv 环境类
├── network.py          # 定义 Actor 和 Critic 神经网络结构
├── ppo_agent.py        # 实现 PPO 算法的核心逻辑，包含与ACO的融合点
├── aco_system.py       # 实现蚁群系统，负责信息素地图的维护和更新
├── visualizer.py       # 可视化工具，用于实时渲染或生成训练/测试结果动画
├── models/             # 存储训练好的模型
├── logs/               # 存储训练日志
└── visualizations/     # 存储可视化结果
```

## 🛠️ 安装依赖

```bash
pip install torch torchvision
pip install gymnasium
pip install matplotlib
pip install numpy
pip install pillow  # 用于生成GIF动画
```

## ⚙️ 配置说明

所有参数都可以在 `config.py` 中进行配置：

### 环境参数
- `GRID_SIZE`: 信息素网格大小 (默认: 50)
- `ENV_BOUNDS`: 环境边界 (默认: 10.0)
- `MAX_STEPS_PER_EPISODE`: 每回合最大步数 (默认: 250)

### 障碍物设置
```python
OBSTACLES = [
    (2.0, 3.0, 1.0),     # (x, y, radius)
    (-5.0, -5.0, 1.5),
    (0.0, -7.0, 0.8)
]
```

### 噪声控制
```python
ENABLE_NOISE = True                    # 是否启用噪声
TRANSLATIONAL_NOISE_STD = 0.1         # 平移噪声标准差
ROTATIONAL_NOISE_STD = 0.2            # 旋转噪声标准差
```

### PPO超参数
- `LEARNING_RATE`: 学习率 (默认: 3e-4)
- `GAMMA`: 折扣因子 (默认: 0.99)
- `CLIP`: PPO剪切参数 (默认: 0.2)
- `TIMESTEPS_PER_BATCH`: 批次大小 (默认: 2048)

### ACO超参数
- `EVAPORATION_RATE`: 信息素蒸发率 (默认: 0.05)
- `DEPOSIT_AMOUNT`: 信息素沉积量 (默认: 100.0)

### 混合算法参数
- `ALPHA_Q_VALUE`: PPO策略权重 (默认: 0.7)
- `BETA_PHEROMONE`: 信息素权重 (默认: 0.3)

## 🚀 使用方法

### 训练模式

基础训练命令：
```bash
python main.py --mode train
```

带可视化的训练：
```bash
python main.py --mode train --render
```

自定义训练参数：
```bash
python main.py --mode train --iterations 500 --device cuda --render --save_freq 50
```

### 测试模式

基础测试命令：
```bash
python main.py --mode test --model_path models/ppo_aco_model.pth
```

带可视化的测试：
```bash
python main.py --mode test --model_path models/ppo_aco_model.pth --render --test_episodes 20
```

### 命令行参数说明

- `--mode`: 运行模式 (`train` 或 `test`)
- `--model_path`: 模型保存/加载路径
- `--iterations`: 训练迭代次数
- `--device`: 计算设备 (`cpu` 或 `cuda`)
- `--render`: 启用可视化
- `--save_freq`: 模型保存频率
- `--test_episodes`: 测试回合数
- `--seed`: 随机种子

## 🧠 算法原理

### PPO (局部决策专家)
智能体通过PPO学习一个实时的、局部的控制策略。它根据眼前的状态（与目标的相对位置、自身朝向）做出即时动作（改变转向）。

### ACO (全局历史向导)
整个环境被一个"信息素场"覆盖。成功的路径会增强其轨迹上的信息素，形成一个"经验地图"。这个地图为PPO的决策提供全局的、基于历史经验的引导。

### 融合机制
在PPO智能体进行动作选择时，它不仅要考虑自己神经网络的输出，还要参考其下一步可能到达位置的信息素浓度，综合两者做出最终决策：

```python
final_score = ALPHA_Q_VALUE * ppo_preference + BETA_PHEROMONE * pheromone_value
```

### 反馈回路
PPO智能体每完成一次成功的任务，其走过的路径将被用来更新和加强全局的信息素地图，从而在下一次迭代中更好地指导智能体。

## 📊 可视化功能

### 训练过程可视化
- 信息素热力图演化
- 实时训练统计图表
- 成功率和奖励曲线

### 测试结果可视化
- 轨迹动画生成
- 单回合可视化结果
- 成功/失败路径对比

## 🔬 实验自定义

### 修改环境设置
在 `config.py` 中修改：
```python
# 添加更多障碍物
OBSTACLES = [
    (1.0, 2.0, 0.8),
    (-3.0, -1.0, 1.2),
    (4.0, -4.0, 1.0),
    # 添加更多...
]

# 调整环境大小
ENV_BOUNDS = 15.0  # 更大的环境

# 修改噪声强度
TRANSLATIONAL_NOISE_STD = 0.05  # 更小的噪声
```

### 调整算法参数
```python
# 更偏向PPO的决策
ALPHA_Q_VALUE = 0.8
BETA_PHEROMONE = 0.2

# 更快的信息素蒸发
EVAPORATION_RATE = 0.1
```

## 🏃‍♂️ 快速开始示例

1. **基础训练**：
```bash
# 训练1000轮，每100轮保存一次模型
python main.py --mode train --iterations 1000 --save_freq 100 --render
```

2. **测试训练好的模型**：
```bash
# 测试20个回合并生成可视化
python main.py --mode test --test_episodes 20 --render
```

3. **自定义实验**：
   - 修改 `config.py` 中的障碍物配置
   - 调整PPO和ACO的权重比例
   - 重新训练并比较结果

## 📈 预期结果

训练成功后，你应该观察到：
- 成功率逐渐提高（通常在几百轮后达到70%以上）
- 平均奖励增加
- 信息素地图形成从起始区域到目标区域的"高速公路"
- 智能体能够有效避开障碍物并找到最优路径

## 🐛 常见问题

### Q: 训练不收敛怎么办？
A: 尝试：
- 降低学习率：`LEARNING_RATE = 1e-4`
- 增加批次大小：`TIMESTEPS_PER_BATCH = 4096`
- 调整PPO-ACO权重比例

### Q: 智能体总是撞到障碍物？
A: 可能需要：
- 增加碰撞惩罚：`COLLISION_PENALTY = -100.0`
- 降低噪声强度
- 增加训练轮数

### Q: 可视化图片保存在哪里？
A: 所有可视化结果保存在 `visualizations/` 目录中

## 📚 扩展建议

- 添加更复杂的障碍物形状（矩形、多边形）
- 实现多智能体协作
- 添加动态障碍物
- 引入更复杂的奖励函数
- 实现在线学习能力

## 📄 License

本项目基于MIT License开源。
