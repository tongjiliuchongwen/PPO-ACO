# stability_fixed_aco_ppo.py - 修复训练稳定性问题的ACO-PPO
# 解决后期性能崩溃问题

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from copy import deepcopy

import config_matrix as config
from integrated_aco_ppo_training import IntegratedACOPPOEnv, IntegratedACOPPOAgent


class StabilizedACOPPOAgent(IntegratedACOPPOAgent):
    """稳定化的ACO-PPO智能体 - 防止训练崩溃"""

    def __init__(self, env, device='cpu'):
        super().__init__(env, device, learning_rate=3e-4)  # 降低初始学习率

        # 🔧 稳定化超参数
        self.gamma = 0.99  # 标准折扣因子
        self.gae_lambda = 0.95  # 标准GAE参数
        self.clip_ratio = 0.1  # 保守的裁剪比例
        self.entropy_coef = 0.02  # 保持探索
        self.value_coef = 0.5  # 标准价值损失权重
        self.max_grad_norm = 0.2  # 严格的梯度裁剪
        self.n_updates_per_iteration = 3  # 减少更新次数

        # 🔧 适应性学习率调度（修复版本）
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=15
        )

        # 🔧 性能跟踪和保护机制
        self.best_model_state = None
        self.best_success_rate = 0.0
        self.performance_buffer = []
        self.buffer_size = 10
        self.performance_threshold = 0.05  # 性能下降阈值
        self.lr_reduced_count = 0  # 学习率降低计数

        print(f"🛡️ 稳定化智能体参数:")
        print(f"   保守clip_ratio={self.clip_ratio}")
        print(f"   严格grad_norm={self.max_grad_norm}")
        print(f"   减少更新次数={self.n_updates_per_iteration}")

    def train(self, num_iterations=100, episodes_per_iteration=25):
        """带稳定性保护的训练循环"""
        print(f"🛡️ 开始稳定化ACO-PPO训练")
        print(f"   迭代次数: {num_iterations}, 每次迭代episodes: {episodes_per_iteration}")
        print(f"   当前时间: 2025-08-22 04:29:28")
        print(f"   用户: tongjiliuchongwen")

        training_history = {
            'success_rates': [],
            'avg_rewards': [],
            'avg_episode_lengths': [],
            'learning_rates': [],
            'rollbacks': []
        }

        rollback_count = 0
        consecutive_poor_performance = 0

        for iteration in range(num_iterations):
            # 收集数据
            batch_data = self._collect_batch_data(episodes_per_iteration)
            success_rate = batch_data['success_rate']
            avg_reward = batch_data['avg_reward']

            # 🔧 性能监控
            self.performance_buffer.append(success_rate)
            if len(self.performance_buffer) > self.buffer_size:
                self.performance_buffer.pop(0)

            # 🔧 保存当前最佳模型
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                self.best_model_state = deepcopy(self.policy_network.state_dict())
                consecutive_poor_performance = 0
                print(f"   💎 新最佳模型: {success_rate:.1%}")

            # 🔧 检测性能崩溃
            performance_drop = False
            if len(self.performance_buffer) >= self.buffer_size:
                recent_avg = np.mean(self.performance_buffer[-5:])
                earlier_avg = np.mean(self.performance_buffer[:5])

                if earlier_avg > 0.1 and recent_avg < earlier_avg - self.performance_threshold:
                    performance_drop = True
                    consecutive_poor_performance += 1

            # 🔧 性能回滚机制
            if performance_drop and consecutive_poor_performance >= 3 and self.best_model_state is not None:
                print(f"   🔄 检测到性能崩溃！回滚到最佳模型...")
                self.policy_network.load_state_dict(self.best_model_state)
                rollback_count += 1
                consecutive_poor_performance = 0

                # 降低学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                self.lr_reduced_count += 1
                print(f"   📉 学习率降低到: {self.optimizer.param_groups[0]['lr']:.2e}")

            # 🔧 谨慎的网络更新
            if batch_data['observations'] and not performance_drop:
                self._stable_update_network(batch_data)

            # 学习率调度
            self.scheduler.step(success_rate)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录统计
            training_history['success_rates'].append(success_rate)
            training_history['avg_rewards'].append(avg_reward)
            training_history['avg_episode_lengths'].append(batch_data['avg_episode_length'])
            training_history['learning_rates'].append(current_lr)
            training_history['rollbacks'].append(rollback_count)

            # 输出进度
            if iteration % 5 == 0 or iteration < 15:
                status_icon = "🔄" if performance_drop else "✅"
                print(f"{status_icon} 迭代 {iteration + 1:3d} | 成功率={success_rate:5.1%} | "
                      f"奖励={avg_reward:7.2f} | 总成功={self._lifetime_successes} | "
                      f"LR={current_lr:.2e} | 回滚={rollback_count}")

            # 里程碑报告
            if (iteration + 1) % 20 == 0:
                recent_success_rate = np.mean(training_history['success_rates'][-10:])
                print(f"📊 迭代 {iteration + 1} 里程碑:")
                print(f"   最近10次平均: {recent_success_rate:.1%}")
                print(f"   历史最佳: {self.best_success_rate:.1%}")
                print(f"   总回滚次数: {rollback_count}")
                print(f"   学习率调整次数: {self.lr_reduced_count}")

        print(f"🏁 稳定化训练完成!")
        print(f"   最佳成功率: {self.best_success_rate:.1%}")
        print(f"   总回滚次数: {rollback_count}")
        print(f"   学习率调整次数: {self.lr_reduced_count}")

        # 最终加载最佳模型
        if self.best_model_state is not None:
            print(f"   🎯 加载最佳模型用于测试")
            self.policy_network.load_state_dict(self.best_model_state)

        return training_history

    def _stable_update_network(self, batch_data):
        """稳定的网络更新，增加额外保护"""
        try:
            # 标准PPO更新流程（减少更新次数）
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

            # 🔧 稳定的优势归一化
            all_advantages = np.array(all_advantages, dtype=np.float32)
            if all_advantages.std() > 1e-6:
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
                # 限制优势值范围
                all_advantages = np.clip(all_advantages, -5.0, 5.0)

            # 转换到GPU
            obs_tensor = torch.from_numpy(batch_obs).float().to(self.device)
            acts_tensor = torch.from_numpy(batch_acts).float().to(self.device)
            old_log_probs = torch.from_numpy(batch_logp).float().to(self.device)
            advantages_tensor = torch.from_numpy(all_advantages).float().to(self.device)
            returns_tensor = torch.from_numpy(np.array(all_returns, dtype=np.float32)).float().to(self.device)

            # 🔧 稳定的PPO更新（减少更新次数）
            for epoch in range(self.n_updates_per_iteration):
                log_probs, values, entropy = self.policy_network.evaluate_actions(obs_tensor, acts_tensor)

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_tensor
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = torch.nn.MSELoss()(values, returns_tensor)
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

                # 🔧 检查损失是否正常
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"   ⚠️ 检测到异常损失，跳过此次更新")
                    break

                self.optimizer.zero_grad()
                total_loss.backward()

                # 🔧 严格的梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)

                # 🔧 梯度异常检查
                if grad_norm > 5.0:
                    print(f"   ⚠️ 梯度过大 ({grad_norm:.2f})，跳过此次更新")
                    continue

                self.optimizer.step()

        except Exception as e:
            print(f"⚠️ 网络更新失败: {e}")


class StabilizedACOPPOEnv(IntegratedACOPPOEnv):
    """稳定化环境 - 减少奖励波动"""

    def __init__(self):
        super().__init__(use_cnn_features=True)

        # 🔧 更稳定的奖励参数
        self.step_penalty = -0.01  # 标准步骤惩罚
        self.target_reward = 50.0  # 适中的成功奖励
        self.collision_penalty = -15.0  # 适中的碰撞惩罚

        # 更保守的信息素奖励参数
        self.pheromone_alpha = 2.0
        self.pheromone_beta = 4.0
        self.pheromone_threshold = 0.08

        # 减少距离塑形奖励的影响
        self.distance_shaping_coef = 0.5
        self.prev_distance = None

        print(f"🛡️ 稳定化环境参数:")
        print(f"   保守奖励设计，减少波动")
        print(f"   信息素奖励系数: α={self.pheromone_alpha}, β={self.pheromone_beta}")

    def reset(self, seed=None, options=None, major_reset=False):
        obs, info = super().reset(seed, options, major_reset)
        self.prev_distance = np.linalg.norm(self.agent_pos - self.target_pos)
        return obs, info

    def _calculate_aco_reward(self):
        """稳定化的奖励计算"""
        # 基础ACO奖励
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

        # 🔧 添加适度的距离塑形奖励
        current_distance = np.linalg.norm(self.agent_pos - self.target_pos)
        if self.prev_distance is not None:
            distance_improvement = self.prev_distance - current_distance
            distance_reward = self.distance_shaping_coef * distance_improvement
            reward += distance_reward

        self.prev_distance = current_distance

        # 🔧 接近目标的适度奖励
        if current_distance < 2.0:
            proximity_reward = 3.0 * (2.0 - current_distance) / 2.0
            reward += proximity_reward

        return reward


def stability_test():
    """稳定性测试"""
    print(f"🛡️ ACO-PPO稳定性测试")
    print(f"   当前时间: 2025-08-22 04:29:28")
    print(f"   用户: tongjiliuchongwen")
    print(f"=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 使用稳定化版本
    env = StabilizedACOPPOEnv()
    agent = StabilizedACOPPOAgent(env, device)

    # 稳定化训练
    print(f"🛡️ 进行稳定化训练 (60次迭代)...")
    history = agent.train(num_iterations=60, episodes_per_iteration=25)

    # 测试稳定性
    print(f"\n🧪 稳定性测试...")
    test_results = []
    for i in range(8):
        success_rate = test_comprehensive(env, agent, 20)
        test_results.append(success_rate)
        print(f"   测试轮次{i + 1}: {success_rate:.1%}")

    avg_test_success = np.mean(test_results)
    test_std = np.std(test_results)
    stability_score = 1 - test_std / max(avg_test_success, 0.01)

    print(f"\n📊 稳定性测试结果:")
    print(f"   平均测试成功率: {avg_test_success:.1%} ± {test_std:.1%}")
    print(f"   测试稳定性评分: {stability_score:.1%}")
    print(f"   最终训练成功率: {history['success_rates'][-1]:.1%}")
    print(f"   最佳历史成功率: {agent.best_success_rate:.1%}")
    print(f"   训练回滚次数: {history['rollbacks'][-1]}")

    # 分析训练稳定性
    success_rates = history['success_rates']
    if len(success_rates) > 30:
        early_phase = np.mean(success_rates[10:20])
        mid_phase = np.mean(success_rates[25:35])
        late_phase = np.mean(success_rates[-10:])

        print(f"\n📈 训练阶段分析:")
        print(f"   早期阶段 (10-20): {early_phase:.1%}")
        print(f"   中期阶段 (25-35): {mid_phase:.1%}")
        print(f"   后期阶段 (最后10次): {late_phase:.1%}")

        # 判断稳定性
        if late_phase >= mid_phase * 0.8:
            print(f"   ✅ 训练稳定，无明显性能崩溃")
            stability_ok = True
        else:
            print(f"   ⚠️ 训练后期有性能下降")
            stability_ok = False
    else:
        stability_ok = True

    # 可视化稳定性
    plt.figure(figsize=(15, 8))

    # 成功率曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['success_rates'], 'b-', linewidth=2, label='Success Rate')
    if agent.best_success_rate > 0:
        plt.axhline(y=agent.best_success_rate, color='r', linestyle='--', alpha=0.7, label='Best Rate')
    plt.title('Training Success Rate (Stabilized)')
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 回滚次数
    plt.subplot(2, 3, 2)
    plt.plot(history['rollbacks'], 'orange', linewidth=2)
    plt.title('Model Rollbacks')
    plt.xlabel('Iteration')
    plt.ylabel('Rollback Count')
    plt.grid(True, alpha=0.3)

    # 学习率变化
    plt.subplot(2, 3, 3)
    plt.plot(history['learning_rates'], 'green', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 测试结果
    plt.subplot(2, 3, 4)
    plt.bar(range(1, len(test_results) + 1), test_results, alpha=0.7)
    plt.axhline(y=avg_test_success, color='r', linestyle='--', label=f'Avg: {avg_test_success:.1%}')
    plt.title('Test Stability')
    plt.xlabel('Test Round')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 奖励稳定性
    plt.subplot(2, 3, 5)
    plt.plot(history['avg_rewards'], 'purple', linewidth=2)
    plt.title('Reward Stability')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)

    # 性能分布
    plt.subplot(2, 3, 6)
    if len(set(history['success_rates'])) > 1:  # 检查是否有变化
        plt.hist(history['success_rates'], bins=max(5, len(set(history['success_rates']))), alpha=0.7,
                 edgecolor='black')
    else:
        plt.bar([0], [len(history['success_rates'])], alpha=0.7)
    plt.axvline(x=np.mean(history['success_rates']), color='r', linestyle='--',
                label=f'Mean: {np.mean(history["success_rates"]):.1%}')
    plt.title('Success Rate Distribution')
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stability_test_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 保存稳定性测试图: stability_test_results.png")

    return avg_test_success, stability_score


def test_comprehensive(env, agent, num_episodes):
    """测试函数"""
    successes = 0
    for episode in range(num_episodes):
        obs, _ = env.reset(major_reset=True)
        hidden_state = None
        done = truncated = False
        steps = 0

        while not (done or truncated) and steps < 200:
            action, _, hidden_state = agent.get_action(obs, hidden_state)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

        if info.get('target_found', False):
            successes += 1

    return successes / num_episodes


if __name__ == "__main__":
    start_time = time.time()

    avg_success, stability = stability_test()

    total_time = time.time() - start_time

    print(f"\n🏁 稳定性测试总结:")
    print(f"   用时: {total_time / 60:.1f} 分钟")
    print(f"   平均成功率: {avg_success:.1%}")
    print(f"   稳定性评分: {stability:.1%}")

    if stability > 0.8 and avg_success > 0.15:
        print(f"🎉 训练稳定且有效！")
    elif stability > 0.8:
        print(f"✅ 训练稳定，但成功率需提升")
    elif avg_success > 0.15:
        print(f"✅ 成功率良好，但稳定性需改善")
    else:
        print(f"⚠️ 仍有稳定性和性能问题需解决")

    print(f"\n💡 建议:")
    if avg_success < 0.1:
        print(f"   - 成功率太低，检查奖励函数设计")
        print(f"   - 考虑增加信息素引导强度")
    elif stability < 0.7:
        print(f"   - 稳定性不足，需要更保守的训练策略")
        print(f"   - 考虑进一步减小学习率和梯度裁剪")
    else:
        print(f"   - 表现良好，可考虑增加训练时间或优化网络架构")