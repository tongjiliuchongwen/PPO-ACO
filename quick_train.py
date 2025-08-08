#!/usr/bin/env python3
# quick_train.py

"""
快速训练示例脚本
用于演示PPO-ACO混合算法的训练过程
"""

import torch
import numpy as np
import config
from environment import ActiveParticleEnv
from aco_system import ACOSystem
from ppo_agent import PPO
from visualizer import Visualizer

def quick_training_demo(iterations=50, save_visualizations=True):
    """
    运行快速训练演示
    
    Args:
        iterations: 训练迭代次数
        save_visualizations: 是否保存可视化结果
    """
    print(f"开始快速训练演示 - {iterations} 轮训练")
    print("=" * 50)
    
    # 创建环境和系统
    env = ActiveParticleEnv()
    aco_system = ACOSystem()
    ppo_agent = PPO(env, device='cpu')
    
    # 创建可视化器
    visualizer = None
    if save_visualizations:
        visualizer = Visualizer(env, aco_system)
    
    # 记录信息素历史
    pheromone_history = []
    
    print(f"环境配置:")
    print(f"  障碍物数量: {len(config.OBSTACLES)}")
    print(f"  环境边界: ±{config.ENV_BOUNDS}")
    print(f"  最大步数: {config.MAX_STEPS_PER_EPISODE}")
    print(f"  噪声启用: {config.ENABLE_NOISE}")
    print()
    
    # 训练循环
    best_success_rate = 0.0
    
    for iteration in range(iterations):
        # PPO收集数据并更新网络
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = \
            ppo_agent.rollout(aco_system)
        
        # 计算优势和回报
        batch_advantages, batch_returns = \
            ppo_agent.compute_advantages(batch_obs, batch_rews, batch_lens)
        
        # 更新PPO网络
        ppo_agent.update_networks(batch_obs, batch_acts, batch_log_probs,
                                 batch_advantages, batch_returns)
        
        # 更新ACO系统
        successful_trajectories = ppo_agent.successful_trajectories
        
        # 从成功轨迹中选择最好的几条进行信息素沉积
        if successful_trajectories:
            # 按奖励排序，选择top 30%
            successful_trajectories.sort(key=lambda x: x['reward'], reverse=True)
            top_trajectories = successful_trajectories[:max(1, len(successful_trajectories) // 3)]
            
            for traj_data in top_trajectories:
                aco_system.deposit(traj_data['trajectory'], traj_data['quality'])
        
        # 信息素蒸发
        aco_system.evaporate()
        
        # 记录信息素地图
        if iteration % 5 == 0:
            pheromone_history.append(aco_system.pheromone_map.copy())
        
        # 计算训练统计
        avg_reward = np.mean(ppo_agent.logger['batch_rews'][-1])
        success_rate = ppo_agent.logger['batch_success_rate'][-1]
        episode_count = len(ppo_agent.logger['batch_rews'][-1])
        
        # 更新最佳成功率
        if success_rate > best_success_rate:
            best_success_rate = success_rate
        
        # 打印训练信息
        if iteration % 5 == 0 or iteration == iterations - 1:
            print(f"迭代 {iteration:3d}/{iterations}: "
                  f"平均奖励 = {avg_reward:7.2f}, "
                  f"成功率 = {success_rate:6.1%}, "
                  f"回合数 = {episode_count:3d}")
            
            # 打印ACO统计
            aco_stats = aco_system.get_statistics()
            print(f"            ACO: "
                  f"平均信息素 = {aco_stats['mean_pheromone']:.3f}, "
                  f"最大值 = {aco_stats['max_pheromone']:.3f}, "
                  f"沉积次数 = {aco_stats['total_deposits']}")
    
    print()
    print("训练完成！")
    print(f"最佳成功率: {best_success_rate:.1%}")
    
    # 保存模型
    model_path = "models/quick_demo_model.pth"
    ppo_agent.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 生成可视化
    if visualizer and save_visualizations:
        print("生成训练统计图...")
        visualizer.plot_training_statistics(ppo_agent, 
                                           save_path="visualizations/quick_training_stats.png")
        
        if pheromone_history:
            print("生成信息素演化图...")
            visualizer.plot_pheromone_evolution(pheromone_history, 
                                               save_path="visualizations/quick_pheromone_evolution.png")
        
        # 生成最终状态可视化
        print("生成最终状态可视化...")
        env.reset()
        agent_pos = env.get_agent_position()
        target_pos = env.get_target_position()
        agent_theta = env.get_agent_orientation()
        
        visualizer.render_frame(agent_pos, target_pos, agent_theta,
                              title=f"训练后状态 (最佳成功率: {best_success_rate:.1%})",
                              save_path="visualizations/quick_final_state.png")
    
    print("\n演示完成! 可视化结果已保存到 visualizations/ 目录")
    
    return ppo_agent, aco_system


def test_trained_model(model_path="models/quick_demo_model.pth", test_episodes=5):
    """
    测试训练好的模型
    
    Args:
        model_path: 模型文件路径
        test_episodes: 测试回合数
    """
    import os
    
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请先运行训练")
        return
    
    print(f"测试训练好的模型 - {test_episodes} 个回合")
    print("=" * 50)
    
    # 创建环境和系统
    env = ActiveParticleEnv()
    aco_system = ACOSystem()
    ppo_agent = PPO(env, device='cpu')
    
    # 加载模型
    ppo_agent.load_model(model_path)
    print(f"模型已从 {model_path} 加载")
    
    # 测试统计
    successful_episodes = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # 获取动作（使用ACO引导）
            action, _ = ppo_agent.get_action(obs, aco_system)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
        
        # 统计结果
        success = done and not info.get('collision', False)
        if success:
            successful_episodes += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"回合 {episode+1}: "
              f"奖励 = {episode_reward:7.2f}, "
              f"步数 = {step_count:3d}, "
              f"成功 = {'是' if success else '否'}")
    
    # 测试结果统计
    success_rate = successful_episodes / test_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print("\n测试结果:")
    print(f"成功率: {success_rate:.1%}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")


def main():
    """主函数"""
    import os
    
    # 创建必要目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    print("PPO-ACO快速训练演示")
    print("这个脚本将演示混合算法的训练过程")
    print()
    
    # 运行训练
    ppo_agent, aco_system = quick_training_demo(iterations=30, save_visualizations=True)
    
    print()
    
    # 测试模型
    test_trained_model(test_episodes=3)
    
    print("\n完整演示结束!")
    print("查看 visualizations/ 目录中的图像了解训练过程")


if __name__ == "__main__":
    main()