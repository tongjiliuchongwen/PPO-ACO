# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:39:03 2025

@author: Lenovo
"""

# debug_test.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from environment import ActiveParticleEnv
from network import ActorNetwork
import config

def run_test():
    """
    加载一个模型并在环境中进行测试，不涉及任何训练逻辑。
    """
    print("--- 启动纯测试模式 ---")

    # --- 1. 设置 ---
    # 确保可视化文件夹存在
    os.makedirs('visualizations', exist_ok=True)
    
    device = torch.device('cpu')
    model_path = 'models/dummy_actor.pth'
    num_episodes = 5 # 我们跑5个回合看看

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'")
        print("请先运行 create_dummy_model.py 来创建一个。")
        return

    # --- 2. 初始化 ---
    # 创建环境
    env = ActiveParticleEnv()

    # 创建和加载 Actor 网络
    actor = ActorNetwork(env.observation_space.shape[0], env.action_space.shape[0])
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.to(device)
    actor.eval() # 设置为评估模式，这很重要

    print(f"成功加载模型: {model_path}")
    print(f"将运行 {num_episodes} 个回合进行测试...")

    # --- 3. 运行模拟 ---
    for i in range(num_episodes):
        print(f"\n--- 回合 #{i+1} ---")
        
        obs, _ = env.reset()
        done = False
        truncated = False
        
        agent_trajectory = [env.get_agent_position().copy()]
        step_count = 0

        while not (done or truncated):
            step_count += 1
            
            # 将观测转换为张量
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # 使用 torch.no_grad() 来确保不计算梯度
            with torch.no_grad():
                # 从 Actor 网络获取动作的均值和标准差
                mean, std = actor(obs_tensor)
            
            # 我们只使用均值作为确定性动作，不进行随机采样，方便观察
            action = mean.cpu().numpy()[0]
            
            # 在环境中执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            # 记录轨迹
            agent_trajectory.append(env.get_agent_position().copy())

        print(f"回合结束于 {step_count} 步。成功: {info.get('distance_to_target', 99) < config.TARGET_RADIUS}, 碰撞: {info.get('collision', False)}")

        # --- 4. 可视化 ---
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # 绘制障碍物
        for (ox, oy, r) in config.OBSTACLES:
            circle = plt.Circle((ox, oy), r, color='black', alpha=0.7)
            ax.add_artist(circle)
            
        # 绘制轨迹
        trajectory_arr = np.array(agent_trajectory)
        plt.plot(trajectory_arr[:, 0], trajectory_arr[:, 1], 'b-o', label='Agent Trajectory', markersize=3, linewidth=1.5)
        
        # 标记起点和终点
        plt.plot(trajectory_arr[0, 0], trajectory_arr[0, 1], 'go', markersize=10, label='Start')
        plt.plot(env.get_target_position()[0], env.get_target_position()[1], 'r*', markersize=15, label='Target')
        
        plt.title(f'Debug Test - Episode #{i+1} Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.xlim(-config.ENV_BOUNDS, config.ENV_BOUNDS)
        plt.ylim(-config.ENV_BOUNDS, config.ENV_BOUNDS)
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        
        save_path = f'visualizations/debug_trajectory_ep_{i+1}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"轨迹图已保存到: {save_path}")

if __name__ == '__main__':
    run_test()