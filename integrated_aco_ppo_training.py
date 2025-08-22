# integrated_aco_ppo_training.py - 集成修复ACO系统的完整训练环境
# 创建时间: 2025-08-22 03:13:50
# 作者: tongjiliuchongwen

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
from copy import deepcopy

import config_matrix as config
from lightweight_cnn_training import LightweightCNNNetwork


class FixedACOSystem:
    """修复的ACO系统：正确的空间扩散和梯度计算"""

    def __init__(self, grid_size=50, world_bounds=10.0):
        self.grid_size = grid_size
        self.world_bounds = world_bounds
        self.grid_resolution = (2 * world_bounds) / grid_size

        # 信息素地图
        self.nav_pheromone_map = np.full((grid_size, grid_size), 0.1, dtype=np.float32)
        self.exp_pheromone_map = np.full((grid_size, grid_size), 0.1, dtype=np.float32)

        # 扩散和蒸发参数
        self.nav_evaporation_rate = 0.03
        self.exp_evaporation_rate = 0.03
        self.diffusion_rate = 0.08
        self.kernel_size = 4  # 9x9区域的半径

        print(f"🔧 集成修复ACO系统:")
        print(f"   网格: {grid_size}x{grid_size}, 范围: [{-world_bounds}, {world_bounds}]")
        print(f"   分辨率: {self.grid_resolution:.3f}, 核大小: {2 * self.kernel_size + 1}x{2 * self.kernel_size + 1}")

    def world_to_grid(self, world_pos):
        """世界坐标转网格坐标"""
        x_normalized = (world_pos[0] + self.world_bounds) / (2 * self.world_bounds)
        y_normalized = (world_pos[1] + self.world_bounds) / (2 * self.world_bounds)

        x_idx = int(x_normalized * (self.grid_size - 1))
        y_idx = int(y_normalized * (self.grid_size - 1))

        x_idx = max(0, min(self.grid_size - 1, x_idx))
        y_idx = max(0, min(self.grid_size - 1, y_idx))

        return x_idx, y_idx

    def get_average_nav_pheromone(self, world_pos, kernel_size=None):
        """获取9x9区域的平均导航信息素浓度"""
        if kernel_size is None:
            kernel_size = self.kernel_size

        center_x, center_y = self.world_to_grid(world_pos)

        min_x = max(0, center_x - kernel_size)
        max_x = min(self.grid_size - 1, center_x + kernel_size)
        min_y = max(0, center_y - kernel_size)
        max_y = min(self.grid_size - 1, center_y + kernel_size)

        region = self.nav_pheromone_map[min_y:max_y + 1, min_x:max_x + 1]
        return float(np.mean(region))

    def get_average_exp_pheromone(self, world_pos, kernel_size=None):
        """获取9x9区域的平均探索信息素浓度"""
        if kernel_size is None:
            kernel_size = self.kernel_size

        center_x, center_y = self.world_to_grid(world_pos)

        min_x = max(0, center_x - kernel_size)
        max_x = min(self.grid_size - 1, center_x + kernel_size)
        min_y = max(0, center_y - kernel_size)
        max_y = min(self.grid_size - 1, center_y + kernel_size)

        region = self.exp_pheromone_map[min_y:max_y + 1, min_x:max_x + 1]
        return float(np.mean(region))

    def get_nav_gradient(self, world_pos):
        """计算导航信息素的空间梯度"""
        step_size = self.grid_resolution * 2

        pos_x_plus = world_pos + np.array([step_size, 0])
        pos_x_minus = world_pos - np.array([step_size, 0])
        pos_y_plus = world_pos + np.array([0, step_size])
        pos_y_minus = world_pos - np.array([0, step_size])

        pos_x_plus[0] = min(self.world_bounds, pos_x_plus[0])
        pos_x_minus[0] = max(-self.world_bounds, pos_x_minus[0])
        pos_y_plus[1] = min(self.world_bounds, pos_y_plus[1])
        pos_y_minus[1] = max(-self.world_bounds, pos_y_minus[1])

        grad_x = (self.get_average_nav_pheromone(pos_x_plus) -
                  self.get_average_nav_pheromone(pos_x_minus)) / (2 * step_size)
        grad_y = (self.get_average_nav_pheromone(pos_y_plus) -
                  self.get_average_nav_pheromone(pos_y_minus)) / (2 * step_size)

        return np.array([grad_x, grad_y])

    def deposit_exploration(self, world_pos, strength=1.0):
        """沉积探索信息素"""
        center_x, center_y = self.world_to_grid(world_pos)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x_idx = center_x + dx
                y_idx = center_y + dy

                if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                    distance_sq = dx * dx + dy * dy
                    weight = np.exp(-distance_sq / 2.0)
                    self.exp_pheromone_map[y_idx, x_idx] += strength * 0.01 * weight

    def deposit_navigation(self, trajectory, strength=1.0):
        """沉积导航信息素轨迹"""
        if len(trajectory) < 2:
            return

        path_quality = strength / max(1, len(trajectory))

        for pos in trajectory:
            center_x, center_y = self.world_to_grid(pos)

            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    x_idx = center_x + dx
                    y_idx = center_y + dy

                    if 0 <= x_idx < self.grid_size and 0 <= y_idx < self.grid_size:
                        distance_sq = dx * dx + dy * dy
                        weight = np.exp(-distance_sq / 4.0)
                        self.nav_pheromone_map[y_idx, x_idx] += path_quality * weight

    def evaporate(self):
        """信息素蒸发"""
        self.nav_pheromone_map *= (1 - self.nav_evaporation_rate)
        self.exp_pheromone_map *= (1 - self.exp_evaporation_rate)

        self.nav_pheromone_map = np.maximum(self.nav_pheromone_map, 0.05)
        self.exp_pheromone_map = np.maximum(self.exp_pheromone_map, 0.05)

    def diffuse_pheromones(self):
        """简化的信息素扩散"""
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1, 0.4, 0.1],
                           [0.05, 0.1, 0.05]])

        # 导航信息素扩散
        new_nav_map = np.copy(self.nav_pheromone_map)
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                region = self.nav_pheromone_map[i - 1:i + 2, j - 1:j + 2]
                new_nav_map[i, j] = np.sum(region * kernel)

        self.nav_pheromone_map = (1 - self.diffusion_rate) * self.nav_pheromone_map + self.diffusion_rate * new_nav_map


class IntegratedACOPPOEnv(gym.Env):
    """集成修复ACO系统的完整训练环境"""

    def __init__(self, use_cnn_features=True):
        super().__init__()

        # 环境参数
        self.env_bounds = config.ENV_BOUNDS
        self.max_steps = config.MAX_STEPS_PER_EPISODE
        self.obstacles = list(config.OBSTACLES)

        # 物理参数
        self.v0 = config.V0
        self.omega_max = config.OMEGA_MAX
        self.dt = config.DT
        self.target_radius = config.TARGET_RADIUS

        # 🔧 使用修复的ACO系统
        self.aco_system = FixedACOSystem(grid_size=50, world_bounds=self.env_bounds)

        # 🔧 修复的奖励参数
        self.step_penalty = -0.01
        self.target_reward = 50.0
        self.collision_penalty = -10.0
        self.boundary_penalty = -5.0

        # 信息素奖励参数
        self.pheromone_alpha = 3.0
        self.pheromone_beta = 5.0
        self.pheromone_threshold = 0.08

        # 观测空间设计
        self.use_cnn_features = use_cnn_features
        if use_cnn_features:
            # 基础观测: 位置+朝向+目标信息 = 6维
            # 信息素观测: 浓度+梯度 = 4维
            obs_dim = 10
            print(f"🔧 集成环境观测设计: 基础6维 + 信息素4维 = {obs_dim}维")
        else:
            # 简化观测: 只有基础信息
            obs_dim = 6
            print(f"🔧 集成环境观测设计: 仅基础观测 = {obs_dim}维")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.omega_max, high=self.omega_max, shape=(1,), dtype=np.float32)

        # 状态变量
        self.agent_pos = np.zeros(2, dtype=float)
        self.agent_theta = 0.0
        self.target_pos = np.zeros(2, dtype=float)
        self.current_step = 0
        self.target_found_this_episode = False

        print(f"🔧 集成ACO-PPO环境创建完成")

    def reset(self, seed=None, options=None, major_reset=False):
        super().reset(seed=seed)
        self.current_step = 0
        self.target_found_this_episode = False

        # 重置目标位置
        if major_reset:
            max_attempts = 100
            for _ in range(max_attempts):
                new_target = np.random.uniform(-self.env_bounds * 0.8, self.env_bounds * 0.8, size=2)
                if not self._is_collision(new_target):
                    self.target_pos = new_target
                    break

        # 重置智能体位置
        max_attempts = 100
        for _ in range(max_attempts):
            distance = np.random.uniform(3.0, 6.0)
            angle = np.random.uniform(0, 2 * np.pi)
            candidate_pos = self.target_pos + distance * np.array([np.cos(angle), np.sin(angle)])

            if (np.all(np.abs(candidate_pos) < self.env_bounds) and
                    not self._is_collision(candidate_pos)):
                self.agent_pos = candidate_pos
                break
        else:
            self.agent_pos = np.random.uniform(-self.env_bounds * 0.5, self.env_bounds * 0.5, size=2)

        self.agent_theta = np.random.uniform(0, 2 * np.pi)

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        omega = float(np.clip(action[0], -self.omega_max, self.omega_max))

        # 物理模拟
        self.agent_theta = (self.agent_theta + omega * self.dt) % (2 * np.pi)
        velocity = np.array([self.v0 * np.cos(self.agent_theta), self.v0 * np.sin(self.agent_theta)])

        new_pos_unclamped = self.agent_pos + velocity * self.dt
        new_pos = np.clip(new_pos_unclamped, -self.env_bounds, self.env_bounds)
        boundary_collision = not np.array_equal(new_pos, new_pos_unclamped)
        collision = self._is_collision(new_pos)

        done = truncated = False

        if collision:
            reward = self.collision_penalty
            done = True
        elif boundary_collision:
            reward = self.boundary_penalty
            done = True
        else:
            self.agent_pos = new_pos

            # 🔧 使用修复的ACO奖励计算
            reward = self._calculate_aco_reward()

            # 🔧 沉积探索信息素
            self.aco_system.deposit_exploration(self.agent_pos)

            # 成功检测
            distance_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
            if distance_to_target < self.target_radius:
                self.target_found_this_episode = True
                reward += self.target_reward
                done = True
            elif self.current_step >= self.max_steps:
                truncated = True

        # 🔧 每隔几步进行信息素扩散
        if self.current_step % 5 == 0:
            self.aco_system.diffuse_pheromones()

        # 蒸发
        self.aco_system.evaporate()

        obs = self._get_observation()
        info = {
            'collision': collision,
            'boundary_collision': boundary_collision,
            'target_found': self.target_found_this_episode,
            'current_step': self.current_step
        }

        return obs, reward, done, truncated, info

    def _calculate_aco_reward(self):
        """基于修复ACO系统的奖励计算"""
        # 基础步骤惩罚
        reward = self.step_penalty

        # 获取信息素信息
        nav_concentration = self.aco_system.get_average_nav_pheromone(self.agent_pos)
        nav_gradient = self.aco_system.get_nav_gradient(self.agent_pos)
        gradient_mag = np.linalg.norm(nav_gradient)

        # 浓度奖励
        if nav_concentration > self.pheromone_threshold:
            concentration_reward = self.pheromone_alpha * (nav_concentration - self.pheromone_threshold)
            reward += concentration_reward

        # 梯度奖励
        if gradient_mag > 0.001:
            gradient_reward = self.pheromone_beta * gradient_mag
            reward += gradient_reward

        return reward

    def _get_observation(self):
        """构造观测向量"""
        # 基础观测：智能体状态
        agent_pos_norm = self.agent_pos / self.env_bounds
        cos_theta = np.cos(self.agent_theta)
        sin_theta = np.sin(self.agent_theta)

        # 目标信息
        target_relative = self.target_pos - self.agent_pos
        target_distance = np.linalg.norm(target_relative) / (self.env_bounds * 2)  # 归一化
        target_angle = np.arctan2(target_relative[1], target_relative[0])
        target_cos = np.cos(target_angle)
        target_sin = np.sin(target_angle)

        # 基础观测
        base_obs = np.array([
            agent_pos_norm[0],  # 0: 智能体x位置
            agent_pos_norm[1],  # 1: 智能体y位置
            cos_theta,  # 2: 朝向余弦
            sin_theta,  # 3: 朝向正弦
            target_distance,  # 4: 目标距离
            target_cos,  # 5: 目标方向余弦
        ])

        if self.use_cnn_features:
            # 信息素观测
            nav_concentration = self.aco_system.get_average_nav_pheromone(self.agent_pos)
            exp_concentration = self.aco_system.get_average_exp_pheromone(self.agent_pos)
            nav_gradient = self.aco_system.get_nav_gradient(self.agent_pos)

            # 归一化信息素观测
            nav_conc_norm = (nav_concentration - 0.05) / 0.5
            exp_conc_norm = (exp_concentration - 0.05) / 0.5

            pheromone_obs = np.array([
                nav_conc_norm,  # 6: 导航信息素浓度
                exp_conc_norm,  # 7: 探索信息素浓度
                nav_gradient[0],  # 8: 导航信息素梯度x
                nav_gradient[1]  # 9: 导航信息素梯度y
            ])

            obs = np.concatenate([base_obs, pheromone_obs])
        else:
            obs = base_obs

        return obs.astype(np.float32)

    def _is_collision(self, position):
        """检测碰撞"""
        for ox, oy, r in self.obstacles:
            if np.linalg.norm(position - np.array([ox, oy])) < r:
                return True
        return False

    def process_successful_episode(self, trajectory):
        """处理成功episode：沉积导航信息素"""
        if len(trajectory) > 1:
            path_quality = 2.0 / max(1, len(trajectory))  # 路径质量
            self.aco_system.deposit_navigation(trajectory, strength=path_quality)

    def get_agent_position(self):
        return self.agent_pos.copy()

    def get_target_position(self):
        return self.target_pos.copy()

    def get_agent_orientation(self):
        return self.agent_theta


class IntegratedACOPPOAgent:
    """集成ACO系统的PPO智能体"""

    def __init__(self, env, device='cpu', learning_rate=3e-4):
        self.env = env
        self.device = device
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # 创建网络
        self.policy_network = LightweightCNNNetwork(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            lstm_hidden_size=64,
            max_action=config.OMEGA_MAX
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # PPO超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.1
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.3
        self.n_updates_per_iteration = 4

        # 统计记录
        self.logger = {
            'batch_success_rate': 0.0,
            'avg_reward': 0.0,
            'avg_episode_length': 0.0
        }
        self._lifetime_successes = 0
        self._global_step = 0

        print(f"🤖 集成ACO-PPO智能体:")
        print(f"   观测维度: {self.obs_dim}, 动作维度: {self.action_dim}")
        print(f"   网络参数: {sum(p.numel() for p in self.policy_network.parameters()):,}")

    def get_action(self, observation, hidden_state=None):
        """获取动作"""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            action, log_prob, value, new_hidden = self.policy_network.get_action_and_value(
                obs_tensor, hidden_state
            )
            return action.cpu().numpy()[0], log_prob.cpu().item(), new_hidden

    def train(self, num_iterations=50, episodes_per_iteration=20):
        """完整训练循环"""
        print(f"🚀 开始集成ACO-PPO训练")
        print(f"   迭代次数: {num_iterations}, 每次迭代episodes: {episodes_per_iteration}")
        print(f"   开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        training_history = {
            'success_rates': [],
            'avg_rewards': [],
            'avg_episode_lengths': []
        }

        for iteration in range(num_iterations):
            # 收集数据
            batch_data = self._collect_batch_data(episodes_per_iteration)

            # 更新网络
            if batch_data['observations']:
                self._update_network(batch_data)

            # 记录统计
            success_rate = batch_data['success_rate']
            avg_reward = batch_data['avg_reward']
            avg_length = batch_data['avg_episode_length']

            training_history['success_rates'].append(success_rate)
            training_history['avg_rewards'].append(avg_reward)
            training_history['avg_episode_lengths'].append(avg_length)

            self.logger['batch_success_rate'] = success_rate
            self.logger['avg_reward'] = avg_reward
            self.logger['avg_episode_length'] = avg_length

            # 输出进度
            print(f"迭代 {iteration + 1:3d} | 成功率={success_rate:5.1%} | "
                  f"平均奖励={avg_reward:7.2f} | 平均步数={avg_length:5.1f} | "
                  f"总成功={self._lifetime_successes}")

            # 定期保存和可视化
            if (iteration + 1) % 10 == 0:
                self._save_checkpoint(iteration + 1)
                self._visualize_progress(training_history, iteration + 1)

        print(f"🏁 训练完成! 总成功次数: {self._lifetime_successes}")
        return training_history

    def _collect_batch_data(self, num_episodes):
        """收集一批训练数据"""
        batch_obs = []
        batch_acts = []
        batch_logp = []
        batch_rews_list = []
        batch_lens = []
        batch_infos = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset(major_reset=True)
            hidden_state = None
            ep_obs = []
            ep_acts = []
            ep_logp = []
            ep_rews = []
            ep_traj = []
            done = truncated = False

            while not (done or truncated):
                self._global_step += 1
                ep_traj.append(self.env.get_agent_position().copy())

                action, log_prob, hidden_state = self.get_action(obs, hidden_state)
                next_obs, reward, done, truncated, info = self.env.step(action)

                ep_obs.append(obs)
                ep_acts.append(action)
                ep_logp.append(log_prob)
                ep_rews.append(reward)

                obs = next_obs

            # 处理episode结束
            ep_traj.append(self.env.get_agent_position().copy())

            if info.get('target_found', False):
                self._lifetime_successes += 1
                # 沉积成功轨迹的导航信息素
                self.env.process_successful_episode(ep_traj)

            # 添加到批次数据
            batch_obs.extend(ep_obs)
            batch_acts.extend(ep_acts)
            batch_logp.extend(ep_logp)
            batch_rews_list.append(ep_rews)
            batch_lens.append(len(ep_rews))
            batch_infos.append(info)

        # 计算统计
        successes = sum(1 for info in batch_infos if info.get('target_found', False))
        success_rate = successes / num_episodes
        avg_reward = np.mean([sum(rews) for rews in batch_rews_list])
        avg_length = np.mean(batch_lens)

        return {
            'observations': batch_obs,
            'actions': batch_acts,
            'log_probs': batch_logp,
            'rewards_list': batch_rews_list,
            'episode_lengths': batch_lens,
            'infos': batch_infos,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_episode_length': avg_length
        }

    def _update_network(self, batch_data):
        """PPO网络更新"""
        try:
            # 转换数据
            batch_obs = np.array(batch_data['observations'], dtype=np.float32)
            batch_acts = np.array(batch_data['actions'], dtype=np.float32)
            batch_logp = np.array(batch_data['log_probs'], dtype=np.float32)

            # 计算价值和优势
            with torch.no_grad():
                obs_tensor = torch.from_numpy(batch_obs).to(self.device)
                _, _, values, _ = self.policy_network.forward(obs_tensor, None)
                values = values.cpu().numpy()

            # GAE计算
            all_advantages = []
            all_returns = []

            start_idx = 0
            for ep_rewards in batch_data['rewards_list']:
                ep_len = len(ep_rewards)
                ep_values = values[start_idx:start_idx + ep_len]

                # GAE
                advantages = np.zeros_like(ep_rewards, dtype=np.float32)
                last_advantage = 0.0

                for t in reversed(range(ep_len)):
                    next_value = ep_values[t + 1] if t < ep_len - 1 else 0.0
                    delta = ep_rewards[t] + self.gamma * next_value - ep_values[t]
                    advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
                    last_advantage = advantages[t]

                returns = advantages + ep_values
                all_advantages.extend(advantages)
                all_returns.extend(returns)
                start_idx += ep_len

            if len(all_advantages) == 0:
                return

            # 归一化优势
            all_advantages = np.array(all_advantages, dtype=np.float32)
            if all_advantages.std() > 1e-8:
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

            # 转换到GPU
            obs_tensor = torch.from_numpy(batch_obs).float().to(self.device)
            acts_tensor = torch.from_numpy(batch_acts).float().to(self.device)
            old_log_probs = torch.from_numpy(batch_logp).float().to(self.device)
            advantages_tensor = torch.from_numpy(all_advantages).float().to(self.device)
            returns_tensor = torch.from_numpy(np.array(all_returns, dtype=np.float32)).float().to(self.device)

            # PPO更新
            for epoch in range(self.n_updates_per_iteration):
                log_probs, values, entropy = self.policy_network.evaluate_actions(obs_tensor, acts_tensor)

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_tensor
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = torch.nn.MSELoss()(values, returns_tensor)
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        except Exception as e:
            print(f"⚠️ 网络更新失败: {e}")

    def _save_checkpoint(self, iteration):
        """保存检查点"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lifetime_successes': self._lifetime_successes,
            'global_step': self._global_step
        }
        torch.save(checkpoint, f'integrated_aco_ppo_checkpoint_iter_{iteration}.pth')
        print(f"   💾 保存检查点: integrated_aco_ppo_checkpoint_iter_{iteration}.pth")

    def _visualize_progress(self, history, iteration):
        """可视化训练进度"""
        plt.figure(figsize=(15, 5))

        # 成功率
        plt.subplot(1, 3, 1)
        plt.plot(history['success_rates'], 'b-', linewidth=2)
        plt.title('Training Success Rate')
        plt.xlabel('Iteration')
        plt.ylabel('Success Rate')
        plt.grid(True, alpha=0.3)

        # 平均奖励
        plt.subplot(1, 3, 2)
        plt.plot(history['avg_rewards'], 'g-', linewidth=2)
        plt.title('Average Reward')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)

        # 平均episode长度
        plt.subplot(1, 3, 3)
        plt.plot(history['avg_episode_lengths'], 'r-', linewidth=2)
        plt.title('Average Episode Length')
        plt.xlabel('Iteration')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'integrated_aco_ppo_progress_iter_{iteration}.png', dpi=150, bbox_inches='tight')
        plt.close()


def run_integrated_training():
    """运行完整的集成训练"""
    print(f"🚀 启动集成ACO-PPO训练")
    print(f"   用户: tongjiliuchongwen")
    print(f"   时间: 2025-08-22 03:13:50")
    print(f"=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 创建环境和智能体
    env = IntegratedACOPPOEnv(use_cnn_features=True)
    agent = IntegratedACOPPOAgent(env, device=device, learning_rate=3e-4)

    # 开始训练
    start_time = time.time()
    training_history = agent.train(num_iterations=50, episodes_per_iteration=20)
    training_time = time.time() - start_time

    # 训练总结
    print(f"\n🏁 训练完成总结:")
    print(f"   训练时间: {training_time / 60:.1f} 分钟")
    print(f"   总成功次数: {agent._lifetime_successes}")
    print(f"   最终成功率: {training_history['success_rates'][-1]:.1%}")
    print(f"   最终平均奖励: {training_history['avg_rewards'][-1]:.2f}")

    # 最终测试
    print(f"\n🧪 进行最终测试...")
    test_success_rate = test_trained_agent(env, agent, num_test_episodes=50)
    print(f"   测试成功率: {test_success_rate:.1%}")

    return training_history, agent


def test_trained_agent(env, agent, num_test_episodes=50):
    """测试训练好的智能体"""
    success_count = 0
    episode_lengths = []

    for episode in range(num_test_episodes):
        obs, _ = env.reset(major_reset=True)
        hidden_state = None
        done = truncated = False
        steps = 0

        while not (done or truncated) and steps < 200:
            action, _, hidden_state = agent.get_action(obs, hidden_state)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

        episode_lengths.append(steps)
        if info.get('target_found', False):
            success_count += 1
            if success_count <= 5:  # 显示前几次成功
                print(f"   🎯 测试成功 {success_count}: {steps} 步")

    success_rate = success_count / num_test_episodes
    avg_steps = np.mean(episode_lengths)

    print(f"   测试统计: 成功率={success_rate:.1%}, 平均步数={avg_steps:.1f}")
    return success_rate


if __name__ == "__main__":
    training_history, trained_agent = run_integrated_training()