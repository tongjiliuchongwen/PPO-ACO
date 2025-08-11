# visualizer.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import config
import os
import matplotlib.lines as mlines

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
        
        # 标记是否有可供图例显示的内容
        has_legend_items = False
        
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
            # 确保轨迹线有一个明确的标签
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], 
                'g-', linewidth=2, alpha=0.7, label='Trajectory'
            )
            # 标记我们绘制了带标签的元素
            has_legend_items = True
        
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
        
        # 只有在需要时（即绘制了带标签的元素后）才调用 legend()
        if has_legend_items:
            ax.legend()
    
    def plot_training_statistics(self, ppo_agent, save_path=None):
        """
        绘制训练统计图表
        
        Args:
            ppo_agent: PPO智能体实例，其中包含了logger
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Statistics')
        
        # 奖励曲线
        if ppo_agent.logger['batch_rews']:
            # logger['batch_rews'] 是一个列表的列表，需要计算每个内部列表的均值
            avg_rewards = [np.mean(rewards) for rewards in ppo_agent.logger['batch_rews']]
            axes[0, 0].plot(avg_rewards)
            axes[0, 0].set_title('Average Episode Reward per Iteration')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        else:
            axes[0, 0].set_title('No Reward Data Available')

        # 成功率曲线
        if ppo_agent.logger['batch_success_rate']:
            axes[0, 1].plot(ppo_agent.logger['batch_success_rate'])
            axes[0, 1].set_title('Success Rate per Iteration')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        else:
            axes[0, 1].set_title('No Success Rate Data Available')

        # Actor损失曲线
        if ppo_agent.logger['actor_losses']:
            axes[1, 0].plot(ppo_agent.logger['actor_losses'])
            axes[1, 0].set_title('Average Actor Loss per Iteration')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].set_title('No Actor Loss Data Available')

        # Critic损失曲线
        if ppo_agent.logger['critic_losses']:
            axes[1, 1].plot(ppo_agent.logger['critic_losses'])
            axes[1, 1].set_title('Average Critic Loss per Iteration')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].set_title('No Critic Loss Data Available')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应总标题
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_statistics.png")

        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"训练统计图已保存到: {save_path}")
        plt.close(fig)    
    
    def generate_animation(self, trajectory_data, filename="simulation.gif"):
        """
        生成轨迹动画 (优化版：静态背景 + 动态前景)
        
        Args:
            trajectory_data: 轨迹数据列表，每个元素包含位置、朝向等信息
            filename: 输出文件名
        """
        if not trajectory_data:
            print("警告: 没有轨迹数据可供动画生成")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size, dpi=self.dpi)
        
        # --- 1. 绘制所有静态背景元素 ---
        
        # 绘制左侧静态的信息素热力图
        # 我们只需要第一帧的数据来获取目标位置
        initial_data = trajectory_data[0]
        self._plot_pheromone_heatmap(ax1, initial_data['agent_pos'], initial_data['target_pos'], 0)
        ax1.set_title("Pheromone Map (Static)")
        
        # 绘制右侧静态的环境背景（障碍物、目标）
        self._plot_environment(ax2, initial_data['agent_pos'], initial_data['target_pos'], 0)
        ax2.set_title("Environment")

        # --- 2. 定义需要动态更新的元素 ---

        # 在右侧环境图上，初始化一个空的轨迹线对象和一个智能体位置对象
        # 这些是我们每一帧需要更新的东西
        trajectory_line, = ax2.plot([], [], 'g-', linewidth=2, alpha=0.7)
        agent_marker = patches.Circle(initial_data['agent_pos'], 0.2, 
                                      linewidth=2, edgecolor='blue', facecolor='lightblue')
        ax2.add_patch(agent_marker)
        
        # --- 3. 定义动画的初始化和更新函数 ---
        
        def init():
            """初始化函数，设置空的动态元素"""
            trajectory_line.set_data([], [])
            agent_marker.center = initial_data['agent_pos']
            return [trajectory_line, agent_marker]

        def animate(frame):
            """动画更新函数，只更新动态元素"""
            if frame >= len(trajectory_data):
                return [trajectory_line, agent_marker]

            data = trajectory_data[frame]
            agent_pos = data['agent_pos']
            trajectory = np.array(data.get('trajectory', []))
            
            # 更新智能体的位置
            agent_marker.center = agent_pos
            
            # 更新轨迹线
            if trajectory.size > 0:
                trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
                
            # 更新标题
            ax2.set_title(f"Environment (Step {frame})")
            
            # 返回所有被修改过的 "artist" 对象
            return [trajectory_line, agent_marker]

        # --- 4. 创建并保存动画 ---
        
        # 使用 blit=True 可以显著提高渲染速度，因为它只重绘变化的部分
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(trajectory_data), interval=100, blit=True
        )
        
        # 保存动画
        save_path = os.path.join(self.output_dir, filename)
        try:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"动画已保存到: {save_path}")
        except Exception as e:
            print(f"!!! 错误：保存动画失败。错误信息: {e} !!!")
            print("这可能是由于matplotlib后端或Pillow库的问题。请尝试更新库：pip install --upgrade matplotlib pillow")

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
        
        # 渲染基础框架（信息素地图和环境背景）
        fig = self.render_frame(agent_pos, target_pos, title=title)
        
        # 在环境图（第二个子图）上绘制完整轨迹
        if len(fig.axes) >= 2:
            ax = fig.axes[1]
            trajectory = np.array(trajectory)

            # 根据成功与否定义线条样式和标签
            line_style = 'g-' if success else 'r--'
            line_label = f'Trajectory ({status})'
            
            # 绘制轨迹并赋予标签
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], 
                line_style, 
                linewidth=2, alpha=0.7, label=line_label
            )
            # 因为我们刚刚明确地添加了一个带标签的 plot, 所以这里调用 legend() 是安全的
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