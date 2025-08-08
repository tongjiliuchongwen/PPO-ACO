# PPO-ACO Technical Documentation

## 算法架构

### 核心组件

#### 1. PPO (Proximal Policy Optimization)
- **Actor网络**: 输出动作的均值和标准差
- **Critic网络**: 输出状态价值函数V(s)
- **观测空间**: [dx, dy, cos(theta), sin(theta), distance_to_target] (5维)
- **动作空间**: [omega] (1维角速度控制)

#### 2. ACO (Ant Colony Optimization)
- **信息素地图**: 50x50网格，覆盖整个环境
- **坐标映射**: 连续世界坐标 ↔ 离散网格坐标
- **信息素更新**: 成功路径强化 + 定期蒸发

#### 3. 环境物理
- **朗之万动力学**: 自驱动粒子运动模型
- **噪声模型**: 平移噪声 + 旋转噪声
- **碰撞检测**: 圆形障碍物碰撞判定

### 算法融合机制

#### 动作选择融合
```python
# 候选动作评估
for candidate_action in candidates:
    # PPO偏好项
    ppo_preference = -(candidate_action - base_action)^2
    
    # ACO引导项
    next_position = predict_next_position(candidate_action)
    aco_guidance = get_pheromone_value(next_position)
    
    # 综合得分
    total_score = α * ppo_preference + β * aco_guidance
    
# 选择最高得分的动作
final_action = argmax(total_score)
```

#### 信息素更新机制
```python
# 成功轨迹筛选
successful_paths = filter_successful_episodes()
top_paths = select_top_quality_paths(successful_paths, top_k=30%)

# 信息素沉积
for path in top_paths:
    path_quality = 1.0 / path_length
    deposit_pheromone(path.trajectory, path_quality)

# 信息素蒸发
pheromone_map *= (1 - evaporation_rate)
```

### 关键参数说明

#### PPO参数
- `LEARNING_RATE = 3e-4`: Adam优化器学习率
- `GAMMA = 0.99`: 折扣因子
- `CLIP = 0.2`: PPO裁剪参数
- `TIMESTEPS_PER_BATCH = 2048`: 每次更新的样本数

#### ACO参数
- `EVAPORATION_RATE = 0.05`: 信息素蒸发率 (5%/轮)
- `DEPOSIT_AMOUNT = 100.0`: 基础信息素沉积量
- `GRID_SIZE = 50`: 信息素网格分辨率

#### 融合参数
- `ALPHA_Q_VALUE = 0.7`: PPO权重 (70%)
- `BETA_PHEROMONE = 0.3`: ACO权重 (30%)

### 网络架构

#### Actor网络
```
输入(5) -> 全连接(128) -> ReLU -> 全连接(128) -> ReLU -> 分支:
                                                    ├── 均值输出(1)
                                                    └── 对数标准差输出(1)
```

#### Critic网络
```
输入(5) -> 全连接(128) -> ReLU -> 全连接(128) -> ReLU -> 价值输出(1)
```

### 训练流程

1. **数据收集阶段**
   - 使用当前策略收集经验轨迹
   - 应用PPO-ACO融合的动作选择机制
   - 记录成功轨迹用于ACO更新

2. **PPO更新阶段**
   - 计算优势函数(GAE)
   - 更新Actor网络(裁剪目标函数)
   - 更新Critic网络(均方误差损失)

3. **ACO更新阶段**
   - 筛选高质量成功轨迹
   - 在轨迹上沉积信息素
   - 应用信息素蒸发

4. **性能评估**
   - 记录成功率、平均奖励
   - 生成可视化结果
   - 保存模型检查点

### 奖励函数设计

```python
def compute_reward(state, action, next_state, info):
    reward = 0
    
    # 时间惩罚 (鼓励快速到达)
    reward += STEP_PENALTY  # -0.01
    
    # 到达目标奖励
    if distance_to_target < TARGET_RADIUS:
        reward += TARGET_REWARD  # +100.0
        done = True
    
    # 碰撞惩罚
    if collision_detected:
        reward += COLLISION_PENALTY  # -50.0
        done = True
    
    return reward, done
```

### 性能指标

#### 主要指标
- **成功率**: 到达目标的回合百分比
- **平均奖励**: 每个回合的累积奖励
- **平均步数**: 成功到达目标的平均步数
- **信息素浓度**: ACO系统的活跃程度

#### 期望性能
- 训练300-500轮后成功率应达到60%以上
- 平均奖励应从负值逐渐提升至正值
- 信息素地图应形成明显的路径模式

### 超参数调优建议

#### 提高成功率
- 增加`COLLISION_PENALTY`强度
- 降低噪声强度`TRANSLATIONAL_NOISE_STD`
- 增加训练轮数

#### 加快收敛
- 调整PPO-ACO权重比例`ALPHA_Q_VALUE:BETA_PHEROMONE`
- 增加批次大小`TIMESTEPS_PER_BATCH`
- 调整信息素蒸发率`EVAPORATION_RATE`

#### 改善探索
- 增加噪声强度
- 降低信息素权重`BETA_PHEROMONE`
- 增加网络熵正则化

## 代码模块说明

### 文件职责
- `main.py`: 主程序入口，训练/测试流程控制
- `config.py`: 所有超参数和环境配置
- `environment.py`: 粒子物理模拟和环境交互
- `network.py`: 神经网络架构定义
- `ppo_agent.py`: PPO算法实现和ACO集成
- `aco_system.py`: 信息素地图管理
- `visualizer.py`: 可视化和结果分析
- `demo.py`: 功能演示脚本
- `quick_train.py`: 快速训练示例

### 扩展建议
1. 添加多智能体支持
2. 实现动态障碍物
3. 引入更复杂的奖励塑形
4. 支持不同形状的障碍物
5. 添加在线学习能力