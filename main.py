# main.py

import argparse
import torch
import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

# 导入项目模块
import config
from environment import ActiveParticleEnv
from aco_system import ACOSystem
from ppo_agent import PPO
from visualizer import Visualizer

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO-ACO Hybrid Algorithm')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], 
                       default='train', help='运行模式: train 或 test')
    parser.add_argument('--model_path', type=str, default='models/ppo_aco_model.pth',
                       help='模型保存/加载路径')
    parser.add_argument('--iterations', type=int, default=None,
                       help='训练迭代次数（覆盖配置文件）')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--render', action='store_true',
                       help='是否启用可视化')
    parser.add_argument('--save_freq', type=int, default=None,
                       help='模型保存频率（覆盖配置文件）')
    parser.add_argument('--test_episodes', type=int, default=10,
                       help='测试模式下的回合数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()

def setup_directories():
    """创建必要的目录"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

def set_random_seeds(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("PPO-ACO Hybrid Algorithm Configuration")
    print("=" * 60)
    print(f"环境边界: {config.ENV_BOUNDS}")
    print(f"网格大小: {config.GRID_SIZE}")
    print(f"最大步数: {config.MAX_STEPS_PER_EPISODE}")
    print(f"障碍物数量: {len(config.OBSTACLES)}")
    print(f"噪声启用: {config.ENABLE_NOISE}")
    print("-" * 60)
    print(f"PPO学习率: {config.LEARNING_RATE}")
    print(f"PPO gamma: {config.GAMMA}")
    print(f"PPO clip: {config.CLIP}")
    print(f"批次大小: {config.TIMESTEPS_PER_BATCH}")
    print("-" * 60)
    print(f"ACO蒸发率: {config.EVAPORATION_RATE}")
    print(f"ACO沉积量: {config.DEPOSIT_AMOUNT}")
    print(f"PPO权重: {config.ALPHA_Q_VALUE}")
    print(f"ACO权重: {config.BETA_PHEROMONE}")
    print("=" * 60)

def train_ppo_aco(args):
    """训练PPO-ACO混合算法"""
    print("开始训练PPO-ACO混合算法...")
    
    # 创建环境和系统
    env = ActiveParticleEnv()
    aco_system = ACOSystem()
    ppo_agent = PPO(env, device=args.device)
    
    # 创建可视化器
    visualizer = None
    if args.render:
        visualizer = Visualizer(env, aco_system)
    
    # 训练参数
    total_iterations = args.iterations or config.TOTAL_ITERATIONS
    save_frequency = args.save_freq or config.SAVE_FREQUENCY
    render_frequency = config.RENDER_FREQUENCY
    
    print(f"总训练迭代次数: {total_iterations}")
    print(f"模型保存频率: {save_frequency}")
    
    # 记录信息素历史
    pheromone_history = []
    
    start_time = time.time()
    best_success_rate = 0.0
    
    try:
        for iteration in range(total_iterations):
            iteration_start = time.time()
            
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
            if iteration % 10 == 0:
                pheromone_history.append(aco_system.pheromone_map.copy())
            
            # 计算训练统计
            avg_reward = np.mean(ppo_agent.logger['batch_rews'][-1])
            success_rate = ppo_agent.logger['batch_success_rate'][-1]
            episode_count = len(ppo_agent.logger['batch_rews'][-1])
            
            iteration_time = time.time() - iteration_start
            
            # 打印训练信息
            if iteration % 10 == 0:
                print(f"Iteration {iteration:4d}/{total_iterations}: "
                      f"Avg Reward = {avg_reward:7.2f}, "
                      f"Success Rate = {success_rate:6.1%}, "
                      f"Episodes = {episode_count:3d}, "
                      f"Time = {iteration_time:.2f}s")
                
                # 打印ACO统计
                aco_stats = aco_system.get_statistics()
                print(f"              ACO: "
                      f"Mean Pheromone = {aco_stats['mean_pheromone']:.3f}, "
                      f"Max = {aco_stats['max_pheromone']:.3f}, "
                      f"Deposits = {aco_stats['total_deposits']}")
            
            # 可视化
            if args.render and iteration % render_frequency == 0 and visualizer:
                env.reset()
                agent_pos = env.get_agent_position()
                target_pos = env.get_target_position()
                agent_theta = env.get_agent_orientation()
                
                title = f"Training Iteration {iteration}"
                save_path = f"visualizations/training_iter_{iteration}.png"
                visualizer.render_frame(agent_pos, target_pos, agent_theta, 
                                      title=title, save_path=save_path)
                plt.close()
            
            # 保存最佳模型
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_model_path = args.model_path.replace('.pth', '_best.pth')
                ppo_agent.save_model(best_model_path)
            
            # 定期保存模型
            if iteration % save_frequency == 0 and iteration > 0:
                checkpoint_path = args.model_path.replace('.pth', f'_iter_{iteration}.pth')
                ppo_agent.save_model(checkpoint_path)
                
                # 保存ACO系统
                aco_path = checkpoint_path.replace('.pth', '_aco.npy')
                aco_system.save_pheromone_map(aco_path)
                
                print(f"模型已保存到: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    
    finally:
        # 保存最终模型
        ppo_agent.save_model(args.model_path)
        final_aco_path = args.model_path.replace('.pth', '_aco.npy')
        aco_system.save_pheromone_map(final_aco_path)
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.2f}秒")
        print(f"最佳成功率: {best_success_rate:.1%}")
        print(f"最终模型已保存到: {args.model_path}")
        
        # 生成训练统计图
        if visualizer:
            visualizer.plot_training_statistics(ppo_agent)
            if pheromone_history:
                visualizer.plot_pheromone_evolution(pheromone_history)

def test_ppo_aco(args):
    """测试PPO-ACO混合算法"""
    print("开始测试PPO-ACO混合算法...")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        print("请先训练模型或指定正确的模型路径")
        return
    
    # 创建环境和系统
    env = ActiveParticleEnv()
    aco_system = ACOSystem()
    ppo_agent = PPO(env, device=args.device)
    
    # 加载模型
    ppo_agent.load_model(args.model_path)
    print(f"模型已从 {args.model_path} 加载")
    
    # 尝试加载ACO系统
    aco_path = args.model_path.replace('.pth', '_aco.npy')
    if os.path.exists(aco_path):
        aco_system.load_pheromone_map(aco_path)
        print(f"ACO信息素地图已从 {aco_path} 加载")
    else:
        print("警告: 未找到ACO信息素地图文件，使用初始信息素地图")
    
    # 创建可视化器
    visualizer = Visualizer(env, aco_system)
    
    # 测试统计
    total_episodes = args.test_episodes
    successful_episodes = 0
    total_rewards = []
    episode_lengths = []
    test_trajectories = []
    
    print(f"开始测试 {total_episodes} 个回合...")
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_trajectory = []
        episode_reward = 0
        step_count = 0
        
        done = False
        truncated = False
        
        # 记录初始状态
        episode_trajectory.append(env.get_agent_position().copy())
        
        while not (done or truncated):
            # 获取动作（使用ACO引导）
            action, _ = ppo_agent.get_action(obs, aco_system)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            # 记录数据
            episode_trajectory.append(env.get_agent_position().copy())
            episode_reward += reward
            step_count += 1
            
            # 可视化（每5步或结束时）
            if args.render and (step_count % 5 == 0 or done or truncated):
                agent_pos = env.get_agent_position()
                target_pos = env.get_target_position()
                agent_theta = env.get_agent_orientation()
                
                title = f"Test Episode {episode+1}, Step {step_count}"
                save_path = f"visualizations/test_ep{episode+1}_step{step_count}.png"
                fig = visualizer.render_frame(agent_pos, target_pos, agent_theta, title=title)
                
                # 在环境图上绘制轨迹
                if len(fig.axes) >= 2:
                    ax = fig.axes[1]
                    traj_array = np.array(episode_trajectory)
                    if len(traj_array) > 1:
                        ax.plot(traj_array[:, 0], traj_array[:, 1], 'g-', linewidth=2, alpha=0.7)
                
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
        
        # 统计结果
        success = done and not info.get('collision', False)
        if success:
            successful_episodes += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        test_trajectories.append({
            'trajectory': episode_trajectory,
            'reward': episode_reward,
            'success': success,
            'length': step_count
        })
        
        print(f"Episode {episode+1:2d}: "
              f"Reward = {episode_reward:7.2f}, "
              f"Steps = {step_count:3d}, "
              f"Success = {'Yes' if success else 'No'}")
        
        # 保存回合可视化
        if args.render:
            visualizer.save_episode_visualization(
                episode_trajectory, episode+1, success
            )
    
    # 测试结果统计
    success_rate = successful_episodes / total_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print("\n" + "="*60)
    print("测试结果统计:")
    print(f"总回合数: {total_episodes}")
    print(f"成功回合数: {successful_episodes}")
    print(f"成功率: {success_rate:.1%}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_length:.1f}")
    print("="*60)
    
    # 生成测试动画
    if args.render and test_trajectories:
        print("生成测试动画...")
        # 选择一个成功的轨迹生成动画
        successful_trajs = [t for t in test_trajectories if t['success']]
        if successful_trajs:
            best_traj = max(successful_trajs, key=lambda x: x['reward'])
            
            # 准备动画数据
            animation_data = []
            for i, pos in enumerate(best_traj['trajectory']):
                animation_data.append({
                    'agent_pos': pos,
                    'target_pos': env.get_target_position(),
                    'agent_theta': 0,  # 简化处理
                    'trajectory': best_traj['trajectory'][:i+1]
                })
            
            visualizer.generate_animation(animation_data, "test_best_trajectory.gif")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 创建目录
    setup_directories()
    
    # 打印配置
    print_config()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    print(f"使用设备: {args.device}")
    print(f"运行模式: {args.mode}")
    print(f"随机种子: {args.seed}")
    
    # 根据模式运行
    if args.mode == 'train':
        train_ppo_aco(args)
    elif args.mode == 'test':
        test_ppo_aco(args)
    else:
        print(f"未知模式: {args.mode}")

if __name__ == "__main__":
    main()