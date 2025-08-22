# lightweight_cnn_training.py - 轻量版CNN训练（基于快速测试成功配置）

import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import config_matrix as config
from environment_cnn_pheromone import CNNPheromoneEnv
from aco_system_matrix import ACOSystemMatrix


class LightweightCNNNetwork(torch.nn.Module):
    """轻量版CNN-LSTM网络：基于快速测试的成功配置"""

    def __init__(self, input_dim=12, action_dim=1, lstm_hidden_size=64, max_action=1.0):
        super(LightweightCNNNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.max_action = max_action

        print(f"🪶 轻量版CNN-LSTM网络:")
        print(f"   输入维度: {input_dim}")
        print(f"   LSTM隐藏维度: {lstm_hidden_size} (简化)")
        print(f"   动作维度: {action_dim}")

        # 简化的输入预处理层
        self.input_processor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )

        # 更小的LSTM
        self.lstm = torch.nn.LSTM(
            input_size=16,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        # 简化的输出头
        self.actor_head = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, action_dim)
        )

        self.critic_head = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

        # 保守的对数标准差
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -1.0))

        # 轻量级权重初始化
        self._init_weights()

    def _init_weights(self):
        """轻量级权重初始化"""
        for module in [self.input_processor, self.actor_head, self.critic_head]:
            for layer in module:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, gain=0.5)
                    torch.nn.init.constant_(layer.bias, 0)

        # LSTM初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data, gain=0.5)
            elif 'bias' in name:
                torch.nn.init.constant_(param.data, 0)

    def forward(self, obs, hidden_state=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            single_step = True
        else:
            single_step = False

        batch_size, seq_len, _ = obs.shape

        # 输入预处理
        processed_input = self.input_processor(obs)

        # LSTM处理
        if hidden_state is None:
            h_0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=obs.device)
            c_0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=obs.device)
            hidden_state = (h_0, c_0)

        lstm_out, new_hidden_state = self.lstm(processed_input, hidden_state)

        # 输出头
        action_mean = self.actor_head(lstm_out)
        value = self.critic_head(lstm_out)

        # 处理输出维度
        if single_step:
            action_mean = action_mean.squeeze(1)
            value = value.squeeze(1).squeeze(1)
        else:
            value = value.squeeze(-1)

        # 保守的log_std限制
        log_std = torch.clamp(self.log_std, -2.0, 0.5)

        return action_mean, log_std, value, new_hidden_state

    def get_action_and_value(self, obs, hidden_state=None, deterministic=False):
        action_mean, log_std, value, new_hidden_state = self.forward(obs, hidden_state)

        if deterministic:
            action = torch.tanh(action_mean) * self.max_action
            log_prob = None
        else:
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_mean, std)
            raw_action = dist.sample()

            action = torch.tanh(raw_action) * self.max_action

            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.sum(torch.log(self.max_action * (1 - torch.tanh(raw_action).pow(2)) + 1e-6), dim=-1)

        return action, log_prob, value, new_hidden_state

    def evaluate_actions(self, obs, actions, hidden_states=None):
        action_mean, log_std, values, _ = self.forward(obs, hidden_states)

        # 反向计算raw_actions
        normalized_actions = actions / self.max_action
        normalized_actions = torch.clamp(normalized_actions, -1 + 1e-6, 1 - 1e-6)
        raw_actions = 0.5 * torch.log((1 + normalized_actions) / (1 - normalized_actions))

        # 计算log_prob
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(action_mean, std)
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)

        # Tanh校正
        log_probs -= torch.sum(torch.log(self.max_action * (1 - normalized_actions.pow(2)) + 1e-6), dim=-1)

        # 熵
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values, entropy


class LightweightCNNAgent:
    """轻量版CNN智能体"""

    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # 获取CNN特征提取器
        pheromone_extractor = env.get_pheromone_extractor()

        # 轻量版网络
        self.policy_network = LightweightCNNNetwork(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            lstm_hidden_size=64,  # 减少到64
            max_action=config.OMEGA_MAX
        ).to(self.device)

        # 保守的优化器设置
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=5e-5)  # 保守学习率

        # 简化的PPO超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.1  # 更保守的裁剪
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.3  # 更严格的梯度裁剪
        self.n_updates_per_iteration = 3  # 减少更新次数
        self.timesteps_per_batch = 2000  # 减少批次大小

        # 轻量版ACO引导
        self.alpha_pref = 0.9
        self.beta_aco = 0.02  # 更轻微的引导

        # 统计记录
        self.logger = {'batch_rews': [], 'batch_success_rate': []}
        self._lifetime_successes = 0
        self._global_step = 0

        # 早停监控
        self.success_history = []
        self.best_success_rate = 0.0
        self.patience_counter = 0
        self.patience_limit = 20  # 20次迭代无改善则早停

    def get_action(self, observation, hidden_state=None, aco_system=None):
        """轻量版动作选择"""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            action, log_prob, value, new_hidden = self.policy_network.get_action_and_value(
                obs_tensor, hidden_state
            )
            return action.cpu().numpy()[0], log_prob.cpu().item(), new_hidden

    def rollout(self, aco_system, episodes_per_batch=30):
        """轻量版rollout：更少的episodes，更快的反馈"""
        batch_obs = []
        batch_acts = []
        batch_logp = []
        batch_rews_list = []
        batch_lens = []
        batch_infos = []

        episode_count = 0
        with torch.no_grad():
            while episode_count < episodes_per_batch:
                try:
                    obs, _ = self.env.reset(major_reset=True)
                    hidden_state = None
                    ep_rews = []
                    ep_traj = []
                    done = truncated = False

                    while not (done or truncated):
                        self._global_step += 1
                        ep_traj.append(self.env.get_agent_position().copy())

                        action, log_prob, hidden_state = self.get_action(obs, hidden_state, aco_system)
                        next_obs, reward, done, truncated, info = self.env.step(action, aco_system)

                        batch_obs.append(obs)
                        batch_acts.append(action)
                        batch_logp.append(log_prob)
                        ep_rews.append(reward)

                        obs = next_obs

                    ep_traj.append(self.env.get_agent_position().copy())
                    batch_rews_list.append(ep_rews)
                    batch_lens.append(len(ep_rews))
                    batch_infos.append(info)

                    # 成功处理：更温和的信息素沉积
                    if info.get('target_found', False):
                        self._lifetime_successes += 1
                        path_quality = 1.5 / max(1, len(ep_traj))  # 适中的沉积强度
                        aco_system.deposit_navigation(ep_traj, path_quality)
                        print(f"🎯 轻量版成功! 步数: {len(ep_traj)}, 总成功: {self._lifetime_successes}")

                    # 温和的蒸发
                    aco_system.nav_evaporation_rate = 0.03
                    aco_system.evaporate()

                    episode_count += 1

                except Exception as e:
                    print(f"⚠️ Episode失败: {e}")
                    episode_count += 1
                    continue

        # 统计
        self.logger['batch_rews'] = [sum(rs) for rs in batch_rews_list]
        num_eps = len(batch_lens)
        succ = sum(1 for inf in batch_infos if inf.get('target_found', False))
        self.logger['batch_success_rate'] = (succ / num_eps) if num_eps > 0 else 0.0

        return {
            'batch_obs': np.array(batch_obs, dtype=np.float32),
            'batch_acts': np.array(batch_acts, dtype=np.float32),
            'batch_log_probs': np.array(batch_logp, dtype=np.float32),
            'episode_lengths': batch_lens,
            'episode_rewards': batch_rews_list
        }

    def compute_advantages(self, rollout_data):
        """简化的优势计算"""
        batch_obs = rollout_data['batch_obs']
        episode_lengths = rollout_data['episode_lengths']
        episode_rewards = rollout_data['episode_rewards']

        if len(batch_obs) == 0:
            return np.array([]), np.array([])

        with torch.no_grad():
            obs_tensor = torch.from_numpy(batch_obs).to(self.device)
            _, _, values, _ = self.policy_network.forward(obs_tensor, None)
            values = values.cpu().numpy()

        all_advantages = []
        all_returns = []

        start_idx = 0
        for i, ep_len in enumerate(episode_lengths):
            if i >= len(episode_rewards) or ep_len == 0:
                continue

            ep_values = values[start_idx:start_idx + ep_len]
            ep_rewards = episode_rewards[i]

            if len(ep_rewards) != ep_len:
                start_idx += ep_len
                continue

            # GAE计算
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
            return np.array([]), np.array([])

        all_advantages = np.array(all_advantages, dtype=np.float32)
        if all_advantages.std() > 1e-8:
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        return all_advantages, np.array(all_returns, dtype=np.float32)

    def update_networks(self, rollout_data, advantages, returns):
        """轻量版网络更新"""
        batch_obs = rollout_data['batch_obs']
        batch_acts = rollout_data['batch_acts']
        batch_log_probs = rollout_data['batch_log_probs']

        if len(batch_obs) == 0 or len(advantages) == 0:
            return

        obs_tensor = torch.from_numpy(batch_obs).float().to(self.device)
        acts_tensor = torch.from_numpy(batch_acts).float().to(self.device)
        old_log_probs = torch.from_numpy(batch_log_probs).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)

        # 更少的更新轮次
        for epoch in range(self.n_updates_per_iteration):
            try:
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
                print(f"⚠️ 更新失败: {e}")
                break

    def check_early_stopping(self, current_success_rate):
        """早停检查"""
        self.success_history.append(current_success_rate)

        # 检查是否有改善
        if current_success_rate > self.best_success_rate:
            self.best_success_rate = current_success_rate
            self.patience_counter = 0
            return False, f"📈 新最佳成功率: {current_success_rate:.1%}"
        else:
            self.patience_counter += 1

            # 检查是否达到早停条件
            if self.patience_counter >= self.patience_limit:
                return True, f"⏹️ 早停触发: {self.patience_limit}次迭代无改善"

            # 检查是否达到目标成功率
            if current_success_rate >= 0.60:
                return True, f"🎯 达到目标成功率: {current_success_rate:.1%}"

            return False, f"⏳ 等待改善: {self.patience_counter}/{self.patience_limit}"

    def save_model(self, path):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'pheromone_extractor': self.env.get_pheromone_extractor().state_dict(),
            'success_history': self.success_history,
            'best_success_rate': self.best_success_rate
        }, path)


def train_lightweight_cnn():
    """轻量版CNN训练主程序"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 启动轻量版CNN训练 - 设备: {device}")
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 使用8维CNN特征（与快速测试一致）
    env = CNNPheromoneEnv(cnn_feature_dim=8)
    aco = ACOSystemMatrix()
    agent = LightweightCNNAgent(env, device=device)

    # 计算网络参数
    total_params = sum(p.numel() for p in agent.policy_network.parameters() if p.requires_grad)
    cnn_params = sum(p.numel() for p in env.get_pheromone_extractor().parameters() if p.requires_grad)

    print(f"📊 轻量版配置:")
    print(f"   观测维度: {env.observation_space.shape[0]} (4基础 + 8CNN特征)")
    print(f"   网络总参数: {total_params:,} (轻量化)")
    print(f"   CNN参数: {cnn_params:,}")
    print(f"   LSTM隐藏层: 64 (vs 原来256)")
    print(f"   学习率: 5e-5 (保守)")
    print(f"   早停耐心: {agent.patience_limit} 迭代")
    print(f"   目标成功率: 60%")

    start_time = time.time()
    iteration = 0
    max_iterations = 100  # 最大迭代次数限制

    print(f"\n🎯 开始轻量版训练...")
    print(f"=" * 60)

    while iteration < max_iterations:
        iteration += 1

        try:
            # 执行rollout
            rollout_data = agent.rollout(aco, episodes_per_batch=30)
            advantages, returns = agent.compute_advantages(rollout_data)

            if len(advantages) > 0:
                agent.update_networks(rollout_data, advantages, returns)

            # 获取统计信息
            success_rate = agent.logger['batch_success_rate']
            avg_reward = np.mean(agent.logger['batch_rews']) if agent.logger['batch_rews'] else 0.0

            # 早停检查
            should_stop, stop_message = agent.check_early_stopping(success_rate)

            # 输出进度
            print(f"迭代 {iteration:3d} | 成功率={success_rate:5.1%} | "
                  f"奖励={avg_reward:6.2f} | 生涯成功={agent._lifetime_successes:3d} | {stop_message}")

            # 早停检查
            if should_stop:
                print(f"\n{stop_message}")
                break

            # 定期保存检查点
            if iteration % 20 == 0:
                checkpoint_path = f"models/lightweight_cnn_checkpoint_{iteration}.pth"
                agent.save_model(checkpoint_path)
                print(f"💾 保存检查点: {checkpoint_path}")

        except Exception as e:
            print(f"❌ 迭代 {iteration} 失败: {e}")
            continue

    # 训练完成
    total_time = time.time() - start_time
    final_success = agent.best_success_rate

    print(f"\n" + "=" * 60)
    print(f"🏁 轻量版训练完成!")
    print(f"   总迭代: {iteration}")
    print(f"   用时: {total_time / 60:.1f} 分钟")
    print(f"   最佳成功率: {final_success:.1%}")
    print(f"   总成功次数: {agent._lifetime_successes}")
    print(f"   结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 保存最终模型
    final_path = "models/lightweight_cnn_final.pth"
    agent.save_model(final_path)
    print(f"💾 保存最终模型: {final_path}")

    # 绘制学习曲线
    if len(agent.success_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(agent.success_history, 'b-', linewidth=2, label='成功率')
        plt.axhline(y=0.60, color='r', linestyle='--', label='目标成功率 (60%)')
        plt.axhline(y=agent.best_success_rate, color='g', linestyle='--',
                    label=f'最佳成功率 ({agent.best_success_rate:.1%})')
        plt.xlabel('迭代次数')
        plt.ylabel('成功率')
        plt.title(f'轻量版CNN训练进展 (最佳: {agent.best_success_rate:.1%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lightweight_cnn_learning_curve.png', dpi=150, bbox_inches='tight')
        print(f"📈 学习曲线保存: lightweight_cnn_learning_curve.png")

    return agent.success_history


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)

    try:
        success_history = train_lightweight_cnn()

        if len(success_history) > 0 and max(success_history) >= 0.60:
            print(f"\n🎉 轻量版CNN训练成功!")
            print(f"   证明CNN方法确实有效")
            print(f"   问题在于之前的训练策略过于复杂")
        elif len(success_history) > 0 and max(success_history) >= 0.30:
            print(f"\n⚠️ 轻量版CNN部分成功")
            print(f"   成功率有提升但未达到预期")
            print(f"   需要进一步调优")
        else:
            print(f"\n❌ 轻量版CNN仍然失败")
            print(f"   问题可能在于算法设计本身")

    except Exception as e:
        print(f"❌ 轻量版训练失败: {e}")
        import traceback

        traceback.print_exc()