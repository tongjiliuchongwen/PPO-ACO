# visualizer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import config
import os

class Visualizer:
    """
    可视化工具类，用于实时渲染和生成动画
    """
    
    def __init__(self, env, aco_system):
        """
        初始化可视化器
        
        Args:
            env: 环境实例
            aco_system: ACO系统实例
        """
        self.env = env
        self.aco_system = aco_system
        self.env_bounds = config.ENV_BOUNDS
        self.obstacles = config.OBSTACLES
        
        # 可视化参数
        self.figure_size = (12, 5)
        self.dpi = 100
        
        # 创建输出目录
        self.output_dir = "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def render_frame(self, agent_pos, target_pos, agent_theta=0, 
                    title="PPO-ACO Simulation", save_path=None):
        """
        渲染单帧图像
        
        Args:
            agent_pos: 智能体位置 [x, y]
            target_pos: 目标位置 [x, y]
            agent_theta: 智能体朝向角度
            title: 图像标题
            save_path: 保存路径（可选）
        
        Returns:
            fig: matplotlib图像对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # 左图：信息素热力图
        self._plot_pheromone_heatmap(ax1, agent_pos, target_pos, agent_theta)
        ax1.set_title("Pheromone Map")
        
        # 右图：环境和轨迹
        self._plot_environment(ax2, agent_pos, target_pos, agent_theta)
        ax2.set_title("Environment")
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_pheromone_heatmap(self, ax, agent_pos, target_pos, agent_theta):
        """绘制信息素热力图"""
        # 显示信息素地图
        im = ax.imshow(
            self.aco_system.pheromone_map, 
            extent=[-self.env_bounds, self.env_bounds, 
                   -self.env_bounds, self.env_bounds],
            origin='lower',
            cmap='hot',
            alpha=0.8
        )
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Pheromone Concentration')
        
        # 绘制障碍物
        for obs_x, obs_y, obs_radius in self.obstacles:
            circle = patches.Circle(
                (obs_x, obs_y), obs_radius, 
                linewidth=2, edgecolor='black', 
                facecolor='gray', alpha=0.8
            )
            ax.add_patch(circle)
        
        # 绘制智能体
        agent_marker = patches.Circle(
            agent_pos, 0.2, 
            linewidth=2, edgecolor='blue', 
            facecolor='lightblue'
        )
        ax.add_patch(agent_marker)
        
        # 绘制智能体朝向
        arrow_length = 0.5
        arrow_end = agent_pos + arrow_length * np.array([
            np.cos(agent_theta), np.sin(agent_theta)
        ])
        ax.arrow(
            agent_pos[0], agent_pos[1], 
            arrow_end[0] - agent_pos[0], arrow_end[1] - agent_pos[1],
            head_width=0.1, head_length=0.1, fc='blue', ec='blue'
        )
        
        # 绘制目标
        target_marker = patches.Circle(
            target_pos, config.TARGET_RADIUS, 
            linewidth=2, edgecolor='red', 
            facecolor='pink', alpha=0.8
        )
        ax.add_patch(target_marker)
        
        ax.set_xlim(-self.env_bounds, self.env_bounds)
        ax.set_ylim(-self.env_bounds, self.env_bounds)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_environment(self, ax, agent_pos, target_pos, agent_theta, trajectory=None):
        """绘制环境和轨迹"""
        # 设置背景
        ax.set_facecolor('lightgray')
        
        # 绘制障碍物
        for obs_x, obs_y, obs_radius in self.obstacles:
            circle = patches.Circle(
                (obs_x, obs_y), obs_radius, 
                linewidth=2, edgecolor='black', 
                facecolor='darkgray'
            )
            ax.add_patch(circle)
        
        # 绘制轨迹
        if trajectory is not None and len(trajectory) > 1:
            trajectory = np.array(trajectory)
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], 
                'g-', linewidth=2, alpha=0.7, label='Trajectory'
            )
        
        # 绘制智能体
        agent_marker = patches.Circle(
            agent_pos, 0.2, 
            linewidth=2, edgecolor='blue', 
            facecolor='lightblue'
        )
        ax.add_patch(agent_marker)
        
        # 绘制智能体朝向
        arrow_length = 0.5
        arrow_end = agent_pos + arrow_length * np.array([
            np.cos(agent_theta), np.sin(agent_theta)
        ])
        ax.arrow(
            agent_pos[0], agent_pos[1], 
            arrow_end[0] - agent_pos[0], arrow_end[1] - agent_pos[1],
            head_width=0.1, head_length=0.1, fc='blue', ec='blue'
        )
        
        # 绘制目标
        target_marker = patches.Circle(
            target_pos, config.TARGET_RADIUS, 
            linewidth=2, edgecolor='red', 
            facecolor='pink'
        )
        ax.add_patch(target_marker)
        
        ax.set_xlim(-self.env_bounds, self.env_bounds)
        ax.set_ylim(-self.env_bounds, self.env_bounds)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if trajectory is not None:
            ax.legend()
    
    def generate_animation(self, trajectory_data, filename="simulation.gif"):
        """
        生成轨迹动画
        
        Args:
            trajectory_data: 轨迹数据列表，每个元素包含位置、朝向等信息
            filename: 输出文件名
        """
        if not trajectory_data:
            print("没有轨迹数据可供动画生成")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # 初始化动画元素
        def init():
            ax1.clear()
            ax2.clear()
            return []
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # 获取当前帧数据
            if frame < len(trajectory_data):
                data = trajectory_data[frame]
                agent_pos = data['agent_pos']
                target_pos = data['target_pos']
                agent_theta = data.get('agent_theta', 0)
                trajectory = data.get('trajectory', None)
                
                # 绘制信息素热力图
                self._plot_pheromone_heatmap(ax1, agent_pos, target_pos, agent_theta)
                ax1.set_title(f"Pheromone Map (Step {frame})")
                
                # 绘制环境
                self._plot_environment(ax2, agent_pos, target_pos, agent_theta, trajectory)
                ax2.set_title(f"Environment (Step {frame})")
            
            return []
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(trajectory_data), interval=100, blit=False
        )
        
        # 保存动画
        save_path = os.path.join(self.output_dir, filename)
        anim.save(save_path, writer='pillow', fps=10)
        print(f"动画已保存到: {save_path}")
        
        plt.close(fig)
    
    def plot_training_statistics(self, ppo_agent, save_path=None):
        """
        绘制训练统计图表
        
        Args:
            ppo_agent: PPO智能体实例
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励曲线
        if ppo_agent.logger['batch_rews']:
            avg_rewards = [np.mean(rewards) for rewards in ppo_agent.logger['batch_rews']]
            axes[0, 0].plot(avg_rewards)
            axes[0, 0].set_title('Average Episode Reward')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # 成功率曲线
        if ppo_agent.logger['batch_success_rate']:
            axes[0, 1].plot(ppo_agent.logger['batch_success_rate'])
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        # Actor损失曲线
        if ppo_agent.logger['actor_losses']:
            axes[1, 0].plot(ppo_agent.logger['actor_losses'])
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Critic损失曲线
        if ppo_agent.logger['critic_losses']:
            axes[1, 1].plot(ppo_agent.logger['critic_losses'])
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "training_statistics.png")
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        print(f"训练统计图已保存到: {save_path}")
        plt.close(fig)
    
    def plot_pheromone_evolution(self, pheromone_history, save_path=None):
        """
        绘制信息素演化过程
        
        Args:
            pheromone_history: 信息素历史数据
            save_path: 保存路径（可选）
        """
        if not pheromone_history:
            print("没有信息素历史数据")
            return
        
        n_snapshots = min(len(pheromone_history), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_snapshots):
            idx = i * len(pheromone_history) // n_snapshots
            pheromone_map = pheromone_history[idx]
            
            im = axes[i].imshow(
                pheromone_map,
                extent=[-self.env_bounds, self.env_bounds, 
                       -self.env_bounds, self.env_bounds],
                origin='lower',
                cmap='hot'
            )
            
            # 绘制障碍物
            for obs_x, obs_y, obs_radius in self.obstacles:
                circle = patches.Circle(
                    (obs_x, obs_y), obs_radius, 
                    linewidth=1, edgecolor='black', 
                    facecolor='gray', alpha=0.5
                )
                axes[i].add_patch(circle)
            
            axes[i].set_title(f'Iteration {idx}')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i])
        
        plt.suptitle('Pheromone Map Evolution')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "pheromone_evolution.png")
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        print(f"信息素演化图已保存到: {save_path}")
        plt.close(fig)
    
    def save_episode_visualization(self, trajectory, episode_num, success=False):
        """
        保存单个回合的可视化结果
        
        Args:
            trajectory: 回合轨迹
            episode_num: 回合编号
            success: 是否成功到达目标
        """
        if not trajectory:
            return
        
        agent_pos = trajectory[-1]  # 最终位置
        target_pos = self.env.get_target_position()
        
        status = "success" if success else "failure"
        title = f"Episode {episode_num} ({status})"
        filename = f"episode_{episode_num}_{status}.png"
        save_path = os.path.join(self.output_dir, filename)
        
        fig = self.render_frame(agent_pos, target_pos, title=title)
        
        # 在环境图上绘制完整轨迹
        if len(fig.axes) >= 2:
            ax = fig.axes[1]  # 环境图
            trajectory = np.array(trajectory)
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], 
                'g-' if success else 'r--', 
                linewidth=2, alpha=0.7, label=f'Trajectory ({status})'
            )
            ax.legend()
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    # 测试可视化功能
    from environment import ActiveParticleEnv
    from aco_system import ACOSystem
    
    # 创建环境和ACO系统
    env = ActiveParticleEnv()
    aco_system = ACOSystem()
    
    # 创建可视化器
    visualizer = Visualizer(env, aco_system)
    
    # 测试静态渲染
    env.reset()
    agent_pos = env.get_agent_position()
    target_pos = env.get_target_position()
    agent_theta = env.get_agent_orientation()
    
    fig = visualizer.render_frame(agent_pos, target_pos, agent_theta, 
                                 title="Test Visualization")
    plt.show()
    
    print("可视化测试完成")