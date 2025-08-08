#!/usr/bin/env python3
# demo.py

"""
PPO-ACO混合算法演示脚本
快速测试框架功能和生成示例可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import ActiveParticleEnv
from aco_system import ACOSystem
from ppo_agent import PPO
from visualizer import Visualizer
import config

def test_environment():
    """测试环境基本功能"""
    print("=== 测试环境功能 ===")
    env = ActiveParticleEnv()
    
    obs, _ = env.reset()
    print(f"初始观测: {obs}")
    print(f"智能体位置: {env.get_agent_position()}")
    print(f"目标位置: {env.get_target_position()}")
    print(f"智能体朝向: {env.get_agent_orientation():.2f} 弧度")
    
    # 测试几步动作
    for i in range(5):
        action = np.random.uniform(-config.OMEGA_MAX, config.OMEGA_MAX, size=1)
        obs, reward, done, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 动作={action[0]:.2f}, 奖励={reward:.2f}, 结束={done or truncated}")
        if done or truncated:
            break
    
    print()

def test_aco_system():
    """测试ACO系统功能"""
    print("=== 测试ACO系统功能 ===")
    aco = ACOSystem()
    
    print(f"网格大小: {aco.grid_size}x{aco.grid_size}")
    print(f"环境边界: ±{aco.env_bounds}")
    
    # 测试坐标转换
    test_positions = [[0, 0], [5, -3], [-8, 7]]
    for pos in test_positions:
        grid_pos = aco.world_to_grid(pos)
        back_pos = aco.grid_to_world(grid_pos)
        pheromone = aco.get_pheromone_value(pos)
        print(f"世界坐标{pos} -> 网格{grid_pos} -> 世界{back_pos} (信息素: {pheromone:.3f})")
    
    # 测试信息素沉积
    test_trajectory = [[0, 0], [1, 1], [2, 2], [3, 3]]
    aco.deposit(test_trajectory, path_quality=1.0)
    print(f"沉积后统计: {aco.get_statistics()}")
    
    print()

def test_visualization():
    """测试可视化功能"""
    print("=== 测试可视化功能 ===")
    
    env = ActiveParticleEnv()
    aco = ACOSystem()
    
    # 重置环境
    env.reset()
    
    # 添加一些信息素轨迹
    test_trajectories = [
        [[0, 0], [2, 2], [4, 4]],
        [[-2, -2], [0, 0], [2, 2]],
        [[3, -3], [1, -1], [0, 1]]
    ]
    
    for i, traj in enumerate(test_trajectories):
        aco.deposit(traj, path_quality=1.0/(i+1))
    
    # 创建可视化器
    visualizer = Visualizer(env, aco)
    
    # 生成静态图像
    agent_pos = env.get_agent_position()
    target_pos = env.get_target_position()
    agent_theta = env.get_agent_orientation()
    
    fig = visualizer.render_frame(
        agent_pos, target_pos, agent_theta,
        title="PPO-ACO Demo Visualization",
        save_path="visualizations/demo_visualization.png"
    )
    
    print("可视化图像已保存到: visualizations/demo_visualization.png")
    plt.close(fig)
    
    print()

def test_ppo_aco_integration():
    """测试PPO-ACO集成功能"""
    print("=== 测试PPO-ACO集成功能 ===")
    
    env = ActiveParticleEnv()
    aco = ACOSystem()
    ppo_agent = PPO(env)
    
    # 添加一些历史信息素
    success_trajectory = [
        [-5, -5], [-3, -3], [-1, -1], [1, 1], [3, 3], [5, 5]
    ]
    aco.deposit(success_trajectory, path_quality=2.0)
    
    # 测试智能体在有无ACO引导下的动作选择
    obs, _ = env.reset()
    
    print("测试动作选择（无ACO引导）:")
    for i in range(3):
        action_no_aco, log_prob = ppo_agent.get_action(obs, aco_system=None)
        print(f"  步骤 {i+1}: 动作 = {action_no_aco[0]:.3f}")
    
    print("测试动作选择（有ACO引导）:")
    for i in range(3):
        action_with_aco, log_prob = ppo_agent.get_action(obs, aco_system=aco)
        print(f"  步骤 {i+1}: 动作 = {action_with_aco[0]:.3f}")
    
    # 测试短期训练
    print("运行简短训练测试...")
    batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = \
        ppo_agent.rollout(aco_system=aco)
    
    print(f"收集数据: {len(batch_obs)} 步, {len(batch_lens)} 回合")
    print(f"平均回合长度: {np.mean(batch_lens):.1f}")
    print(f"平均奖励: {np.mean(batch_rews):.2f}")
    
    print()

def main():
    """主演示函数"""
    print("PPO-ACO混合算法框架演示")
    print("=" * 50)
    
    # 创建输出目录
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # 运行各项测试
    test_environment()
    test_aco_system()
    test_visualization()
    test_ppo_aco_integration()
    
    print("演示完成！")
    print("你可以运行以下命令来训练和测试完整模型:")
    print("  训练: python main.py --mode train --iterations 100 --render")
    print("  测试: python main.py --mode test --render")

if __name__ == "__main__":
    main()