# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import config
from network import create_networks

class PPO:
    """
    PPO算法实现，集成ACO系统
    """
    
    def __init__(self, env, device='cpu'):
        """
        初始化PPO智能体
        
        Args:
            env: 环境实例
            device: 计算设备
        """
        self.env = env
        self.device = device
        
        # 网络参数
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # 创建Actor和Critic网络
        self.actor, self.critic = create_networks(self.obs_dim, self.action_dim)
        self.actor.to(device)
        self.critic.to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE)
        # 使用线性衰减，在总迭代次数内将学习率从初始值衰减到0
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.actor_optimizer, 
            lr_lambda=lambda iteration: 1.0 - (iteration / config.TOTAL_ITERATIONS)
            )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer,
            lr_lambda=lambda iteration: 1.0 - (iteration / config.TOTAL_ITERATIONS)
            )
        # PPO超参数
        self.gamma = config.GAMMA
        self.clip_ratio = config.CLIP
        self.n_updates_per_iteration = config.N_UPDATES_PER_ITERATION
        self.timesteps_per_batch = config.TIMESTEPS_PER_BATCH
        
        # ACO集成参数
        self.alpha_q_value = config.ALPHA_Q_VALUE
        self.beta_pheromone = config.BETA_PHEROMONE
        
        # 训练统计
        self.logger = {
            'batch_lens': [],
            'batch_rews': [],
            'batch_success_rate': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        # 存储轨迹用于ACO更新
        self.successful_trajectories = []
        
    def get_action(self, observation, aco_system=None):
        """
        根据观测获取动作，并区分探索动作和原始动作。
        
        Args:
            observation: 当前观测
            aco_system: ACO系统实例
        
        Returns:
            exploration_action: 最终执行的、可能被ACO引导的动作 (numpy array)
            original_action: Actor网络原始采样的动作 (Tensor)
            original_log_prob: 原始动作的对数概率 (Tensor)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # 1. 从Actor网络获取原始的动作分布、采样动作和对数概率
            mean, std = self.actor(obs_tensor)
            dist = torch.distributions.Normal(mean, std)
            
            original_action = dist.sample()  # 这是Actor真正想执行的动作（Tensor形式）
            original_log_prob = dist.log_prob(original_action).sum(dim=-1) # 这是用于训练的log_prob
            
            # 将原始动作转换为numpy数组，用于后续的ACO引导计算
            base_action_np = original_action.cpu().numpy()[0]
            
            # 2. 如果有ACO系统，则使用其引导来生成最终的探索动作
            if aco_system is not None:
                # 集成ACO信息素引导，得到最终要探索的动作
                exploration_action = self._integrate_aco_guidance(
                    observation, base_action_np, aco_system
                )
            else:
                # 如果没有ACO引导，探索动作就是原始动作
                exploration_action = base_action_np
            
            return exploration_action, original_action, original_log_prob
    
    def _integrate_aco_guidance(self, observation, base_action, aco_system):
        """
        集成ACO信息素引导到动作选择中
        
        Args:
            observation: 当前观测 [dx, dy, cos(theta), sin(theta), distance]
            base_action: PPO网络输出的基础动作
            aco_system: ACO系统实例
        
        Returns:
            final_action: 融合后的最终动作
        """
        # 获取当前智能体位置和朝向
        agent_pos = self.env.get_agent_position()
        agent_theta = self.env.get_agent_orientation()
        
        # 生成候选动作
        candidates = [
            base_action[0] - 0.5,  # 左转更多
            base_action[0] - 0.2,  # 左转一点
            base_action[0],        # 原动作
            base_action[0] + 0.2,  # 右转一点
            base_action[0] + 0.5   # 右转更多
        ]
        
        # 限制候选动作在有效范围内
        candidates = [np.clip(c, -config.OMEGA_MAX, config.OMEGA_MAX) for c in candidates]
        
        best_score = -np.inf
        best_action = base_action[0]
        
        # 评估每个候选动作
        for candidate_omega in candidates:
            # 预测执行该动作后的下一步位置
            next_theta = agent_theta + candidate_omega * config.DT
            next_pos = agent_pos + config.V0 * config.DT * np.array([
                np.cos(next_theta),
                np.sin(next_theta)
            ])
            
            # 确保位置在环境范围内
            next_pos = np.clip(next_pos, -config.ENV_BOUNDS, config.ENV_BOUNDS)
            
            # 获取该位置的信息素浓度
            pheromone_value = aco_system.get_pheromone_value(next_pos)
            
            # 计算综合得分
            # PPO偏好项：偏向于原始动作
            ppo_preference = -((candidate_omega - base_action[0]) ** 2)
            
            # ACO引导项：偏向于信息素浓度高的位置
            aco_guidance = pheromone_value
            
            # 综合得分
            total_score = (self.alpha_q_value * ppo_preference + 
                          self.beta_pheromone * aco_guidance)
            
            if total_score > best_score:
                best_score = total_score
                best_action = candidate_omega
        
        return np.array([best_action])
    
    def rollout(self, aco_system=None):
        """
        收集一批经验数据
        
        Args:
            aco_system: ACO系统实例
        
        Returns:
            batch_obs: 观测批次
            batch_acts: 动作批次
            batch_log_probs: 对数概率批次
            batch_rews: 奖励批次
            batch_lens: 回合长度
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        
        # 记录成功轨迹
        episode_trajectories = []
        episode_rewards = []
        
        t = 0
        while t < self.timesteps_per_batch:
            ep_obs = []
            ep_acts = []
            ep_log_probs = []
            ep_rews = []
            ep_trajectory = []
            
            obs, _ = self.env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                # 记录当前位置
                current_pos = self.env.get_agent_position()
                ep_trajectory.append(current_pos.copy())
                
                # 获取动作：同时获得用于探索和用于训练的动作/log_prob
                exploration_action, original_action, original_log_prob = self.get_action(obs, aco_system)
                
                # 在环境中执行的是被ACO引导过的探索动作
                next_obs, reward, done, truncated, info = self.env.step(exploration_action)
                
                # 存储经验时，存储的是Actor网络原始的输出，这是PPO训练所必需的
                ep_obs.append(obs)
                ep_acts.append(original_action.cpu().numpy()[0])      # 存储原始动作
                ep_log_probs.append(original_log_prob.cpu().numpy()[0]) # 存储原始log_prob
                ep_rews.append(reward)
                
                obs = next_obs
                t += 1
                
                if t >= self.timesteps_per_batch:
                    break
            
            # 记录最后位置
            ep_trajectory.append(self.env.get_agent_position().copy())
            
            # 存储回合数据
            batch_obs.extend(ep_obs)
            batch_acts.extend(ep_acts)
            batch_log_probs.extend(ep_log_probs)
            # 注意：这里的奖励 batch_rews 应该是 ep_rews，而不是 batch_rews
            batch_rews.append(ep_rews)
            batch_lens.append(len(ep_obs))
            
            # 检查是否成功（到达目标）
            if done and not info.get('collision', False):
                # 计算路径质量（路径长度的倒数）
                path_length = len(ep_trajectory)
                path_quality = 1.0 / path_length if path_length > 0 else 0.0
                
                episode_trajectories.append({
                    'trajectory': ep_trajectory,
                    'quality': path_quality,
                    'reward': sum(ep_rews)
                })
            
            episode_rewards.append(sum(ep_rews))
        
        # 存储成功轨迹用于ACO更新
        self.successful_trajectories = episode_trajectories
        
        # 计算成功率
        success_count = len(episode_trajectories)
        total_episodes = len(batch_lens)
        success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
        
        # 更新统计信息
        self.logger['batch_lens'].append(batch_lens)
        self.logger['batch_rews'].append(episode_rewards)
        self.logger['batch_success_rate'].append(success_rate)
        
        # 注意：batch_rews 的收集方式有误，已在上面更正。现在返回的是正确的 batch_rews
        # 这里需要将 batch_rews 从一个 list of lists 展平
        flat_batch_rews = [rew for ep_rews in batch_rews for rew in ep_rews]

        return (np.array(batch_obs), np.array(batch_acts), 
                np.array(batch_log_probs), np.array(flat_batch_rews), 
                batch_lens)
    
    def compute_advantages(self, batch_obs, batch_rews, batch_lens):
        """
        计算优势函数
        
        Args:
            batch_obs: 观测批次
            batch_rews: 奖励批次
            batch_lens: 回合长度
        
        Returns:
            batch_advantages: 优势函数
            batch_returns: 回报
        """
        batch_returns = []
        batch_advantages = []
        
        # 计算每个回合的回报和优势
        obs_idx = 0
        for ep_len in batch_lens:
            ep_obs = batch_obs[obs_idx:obs_idx+ep_len]
            ep_rews = batch_rews[obs_idx:obs_idx+ep_len]
            
            # 计算回报
            ep_returns = []
            discounted_return = 0
            for reward in reversed(ep_rews):
                discounted_return = reward + self.gamma * discounted_return
                ep_returns.append(discounted_return)
            ep_returns.reverse()
            
            # 计算价值函数
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(ep_obs).to(self.device)
                ep_values = self.critic(obs_tensor).cpu().numpy()
            
            # 计算优势
            ep_advantages = np.array(ep_returns) - ep_values
            
            batch_returns.extend(ep_returns)
            batch_advantages.extend(ep_advantages)
            
            obs_idx += ep_len
        
        # 标准化优势
        batch_advantages = np.array(batch_advantages)
        batch_advantages = (batch_advantages - np.mean(batch_advantages)) / (np.std(batch_advantages) + 1e-8)
        
        batch_returns = np.array(batch_returns)
        batch_returns = (batch_returns - np.mean(batch_returns)) / (np.std(batch_returns) + 1e-8)
        
        return batch_advantages, np.array(batch_returns)
    
    def update_networks(self, batch_obs, batch_acts, batch_log_probs, 
                       batch_advantages, batch_returns):
        """
        更新Actor和Critic网络
        """
        # 转换为张量
        obs_tensor = torch.FloatTensor(batch_obs).to(self.device)
        acts_tensor = torch.FloatTensor(batch_acts).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(batch_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(batch_advantages).to(self.device)
        returns_tensor = torch.FloatTensor(batch_returns).to(self.device)
        
        actor_losses = []
        critic_losses = []
        
        # 多次更新
        for _ in range(self.n_updates_per_iteration):
            # 计算当前策略的对数概率和熵
            current_log_probs, entropy = self.actor.evaluate_action(obs_tensor, acts_tensor)
            
            # >>>>> 核心修改点：在计算 exp 之前增加数值稳定性 <<<<<
            # 1. 计算对数概率的比率
            log_ratio = current_log_probs - old_log_probs_tensor
            
            # 2. 对 log_ratio 进行裁剪，防止其值过大或过小导致 exp 溢出
            # 这个范围可以根据需要调整，[-20, 20] 是一个比较安全的选择
            log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
            
            # 3. 计算比率
            ratio = torch.exp(log_ratio)
            
            # 计算clipped surrogate loss
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_tensor
            
            # (可选的调试步骤) 检查计算过程中是否已出现 nan
            if torch.isnan(surr1).any() or torch.isnan(surr2).any():
                print("!!! 警告: 在计算 surrogate loss 时检测到 NaN，跳过本次更新。 !!!")
                # 如果需要深入调试，可以在这里设置断点
                # import pdb; pdb.set_trace()
                continue  # 跳过此次更新，防止 nan 污染梯度

            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # 计算Critic损失
            current_values = self.critic(obs_tensor)
            critic_loss = nn.MSELoss()(current_values, returns_tensor)
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # 更新统计信息
        self.logger['actor_losses'].append(np.mean(actor_losses))
        self.logger['critic_losses'].append(np.mean(critic_losses))

    
    def learn(self, total_iterations, aco_system=None):
        """
        训练PPO智能体
        
        Args:
            total_iterations: 总训练迭代次数
            aco_system: ACO系统实例
        """
        for iteration in range(total_iterations):
            # 收集经验
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = \
                self.rollout(aco_system)
            
            # 计算优势和回报
            batch_advantages, batch_returns = \
                self.compute_advantages(batch_obs, batch_rews, batch_lens)
            
            # 更新网络
            self.update_networks(batch_obs, batch_acts, batch_log_probs,
                               batch_advantages, batch_returns)
            
            # 打印训练信息
            if iteration % 10 == 0:
                avg_reward = np.mean(self.logger['batch_rews'][-1])
                success_rate = self.logger['batch_success_rate'][-1]
                print(f"Iteration {iteration}: Avg Reward = {avg_reward:.2f}, "
                      f"Success Rate = {success_rate:.2%}")
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])