# final_optimized_aco_ppo.py - 最终优化版ACO-PPO
# 解决泛化性问题，提升测试稳定性

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from copy import deepcopy

import config_matrix as config
from stability_fixed_aco_ppo import StabilizedACOPPOAgent, StabilizedACOPPOEnv


class FinalOptimizedAgent(StabilizedACOPPOAgent):
    """最终优化智能体 - 提升泛化能力"""

    def __init__(self, env, device='cpu'):
        super().__init__(env, device)

        # 🔧 提升泛化能力的参数调整
        self.entropy_coef = 0.03  # 增加探索
        self.n_updates_per_iteration = 4  # 适度增加更新
        self.clip_ratio = 0.12  # 稍微放松裁剪

        # 🔧 经验回放缓冲区（提升样本效率）
        self.experience_buffer = []
        self.buffer_capacity = 1000
        self.use_experience_replay = True

        print(f"🎯 最终优化智能体:")
        print(f"   增强探索: entropy_coef={self.entropy_coef}")
        print(f"   经验回放: 容量={self.buffer_capacity}")


class FinalOptimizedEnv(StabilizedACOPPOEnv):
    """最终优化环境 - 增强泛化训练"""

    def __init__(self):
        super().__init__()

        # 🔧 更强的泛化训练设置
        self.training_difficulty_levels = [
            {'max_distance': 6.0, 'min_distance': 3.0},  # 简单
            {'max_distance': 8.0, 'min_distance': 4.0},  # 中等
            {'max_distance': 10.0, 'min_distance': 5.0}  # 困难
        ]
        self.current_difficulty = 0
        self.episodes_per_difficulty = 100
        self.episode_count = 0

        print(f"🎯 最终优化环境:")
        print(f"   渐进难度训练，{len(self.training_difficulty_levels)}个难度级别")

    def reset(self, seed=None, options=None, major_reset=False):
        # 🔧 渐进难度调整
        if major_reset:
            self.episode_count += 1
            if self.episode_count % self.episodes_per_difficulty == 0:
                self.current_difficulty = min(
                    self.current_difficulty + 1,
                    len(self.training_difficulty_levels) - 1
                )
                if self.episode_count % (self.episodes_per_difficulty * 2) == 0:
                    print(f"   📈 难度提升到级别 {self.current_difficulty + 1}")

        obs, info = super().reset(seed, options, major_reset)

        # 🔧 根据难度调整目标距离
        if major_reset:
            difficulty = self.training_difficulty_levels[self.current_difficulty]
            max_attempts = 50

            for _ in range(max_attempts):
                distance = np.random.uniform(-self.env_bounds * 0.8, self.env_bounds * 0.8)
                angle = np.random.uniform(0, 2 * np.pi)

                # 根据难度设置距离
                target_distance = np.random.uniform(
                    difficulty['min_distance'],
                    difficulty['max_distance']
                )

                candidate_target = self.agent_pos + target_distance * np.array([
                    np.cos(angle), np.sin(angle)
                ])

                if (np.all(np.abs(candidate_target) < self.env_bounds) and
                        not self._is_collision(candidate_target)):
                    self.target_pos = candidate_target
                    break

        return obs, info


def comprehensive_final_test():
    """综合最终测试"""
    print(f"🎯 ACO-PPO最终综合测试")
    print(f"   当前时间: 2025-08-22 04:37:56")
    print(f"   用户: tongjiliuchongwen")
    print(f"=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 🔧 最终优化训练
    env = FinalOptimizedEnv()
    agent = FinalOptimizedAgent(env, device)

    print(f"🎯 最终优化训练 (80次迭代)...")
    history = agent.train(num_iterations=80, episodes_per_iteration=25)

    # 🔧 多轮次稳定性测试
    print(f"\n🧪 多轮次稳定性测试...")
    all_test_results = []

    test_configs = [
        {"episodes": 20, "name": "标准测试"},
        {"episodes": 30, "name": "长测试"},
        {"episodes": 15, "name": "快速测试"}
    ]

    for config_item in test_configs:
        print(f"\n📋 {config_item['name']} ({config_item['episodes']}个episodes):")
        round_results = []

        for round_num in range(5):
            success_rate = test_comprehensive_final(env, agent, config_item['episodes'])
            round_results.append(success_rate)
            all_test_results.append(success_rate)
            print(f"   轮次{round_num + 1}: {success_rate:.1%}")

        round_avg = np.mean(round_results)
        round_std = np.std(round_results)
        print(f"   {config_item['name']}平均: {round_avg:.1%} ± {round_std:.1%}")

    # 🔧 综合分析
    overall_avg = np.mean(all_test_results)
    overall_std = np.std(all_test_results)
    consistency_score = 1 - (overall_std / max(overall_avg, 0.01))

    print(f"\n📊 最终综合测试结果:")
    print(f"   总体平均成功率: {overall_avg:.1%} ± {overall_std:.1%}")
    print(f"   测试一致性评分: {consistency_score:.1%}")
    print(f"   最佳训练成功率: {agent.best_success_rate:.1%}")
    print(f"   最终训练成功率: {history['success_rates'][-1]:.1%}")

    # 🔧 性能稳定性判断
    training_final_10 = np.mean(history['success_rates'][-10:])
    if overall_avg >= training_final_10 * 0.7:
        generalization_quality = "优秀"
        generalization_icon = "🎉"
    elif overall_avg >= training_final_10 * 0.5:
        generalization_quality = "良好"
        generalization_icon = "✅"
    else:
        generalization_quality = "需改进"
        generalization_icon = "⚠️"

    print(f"   泛化能力评估: {generalization_icon} {generalization_quality}")
    print(f"   训练稳定性: {'✅ 无崩溃' if history['rollbacks'][-1] == 0 else '⚠️ 有回滚'}")

    # 🔧 详细可视化
    create_final_analysis_plots(history, all_test_results, agent)

    return overall_avg, consistency_score


def test_comprehensive_final(env, agent, num_episodes):
    """最终测试函数"""
    successes = 0

    for episode in range(num_episodes):
        obs, _ = env.reset(major_reset=True)
        hidden_state = None
        done = truncated = False
        steps = 0

        while not (done or truncated) and steps < 250:  # 增加测试步数
            action, _, hidden_state = agent.get_action(obs, hidden_state)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

        if info.get('target_found', False):
            successes += 1

    return successes / num_episodes


def create_final_analysis_plots(history, test_results, agent):
    """创建最终分析图表"""
    plt.figure(figsize=(18, 10))

    # 训练成功率曲线
    plt.subplot(2, 4, 1)
    plt.plot(history['success_rates'], 'b-', linewidth=2, label='Training')
    if agent.best_success_rate > 0:
        plt.axhline(y=agent.best_success_rate, color='r', linestyle='--', alpha=0.7,
                    label=f'Best: {agent.best_success_rate:.1%}')
    plt.title('Training Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 学习率变化
    plt.subplot(2, 4, 2)
    plt.plot(history['learning_rates'], 'green', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 奖励趋势
    plt.subplot(2, 4, 3)
    plt.plot(history['avg_rewards'], 'purple', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Reward Trend')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)

    # 测试结果分布
    plt.subplot(2, 4, 4)
    plt.hist(test_results, bins=max(5, len(set(test_results)) if len(set(test_results)) > 1 else 5),
             alpha=0.7, edgecolor='black')
    plt.axvline(x=np.mean(test_results), color='r', linestyle='--',
                label=f'Mean: {np.mean(test_results):.1%}')
    plt.title('Test Results Distribution')
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 训练vs测试对比
    plt.subplot(2, 4, 5)
    final_training = np.mean(history['success_rates'][-10:])
    test_avg = np.mean(test_results)

    categories = ['Training\n(Final)', 'Testing\n(Average)']
    values = [final_training, test_avg]
    colors = ['skyblue', 'lightcoral']

    bars = plt.bar(categories, values, color=colors, alpha=0.8)
    plt.title('Training vs Testing')
    plt.ylabel('Success Rate')

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.1%}', ha='center', va='bottom', fontweight='bold')

    # 稳定性分析
    plt.subplot(2, 4, 6)
    window_size = 10
    rolling_std = []
    for i in range(window_size, len(history['success_rates'])):
        window_data = history['success_rates'][i - window_size:i]
        rolling_std.append(np.std(window_data))

    plt.plot(range(window_size, len(history['success_rates'])), rolling_std, 'orange', linewidth=2)
    plt.title('Training Stability')
    plt.xlabel('Iteration')
    plt.ylabel('Rolling Std')
    plt.grid(True, alpha=0.3)

    # 学习阶段分析
    plt.subplot(2, 4, 7)
    n_phases = 4
    phase_size = len(history['success_rates']) // n_phases
    phase_avgs = []

    for i in range(n_phases):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < n_phases - 1 else len(history['success_rates'])
        phase_avg = np.mean(history['success_rates'][start_idx:end_idx])
        phase_avgs.append(phase_avg)

    plt.bar(range(1, n_phases + 1), phase_avgs, alpha=0.7, color='lightgreen')
    plt.title('Learning Phases')
    plt.xlabel('Phase')
    plt.ylabel('Average Success Rate')
    plt.xticks(range(1, n_phases + 1))
    plt.grid(True, alpha=0.3)

    # 综合评分雷达图
    plt.subplot(2, 4, 8)

    # 计算各维度评分
    final_success = min(final_training / 0.5, 1.0)  # 基于50%目标
    test_success = min(test_avg / 0.3, 1.0)  # 基于30%目标
    stability = min(1 - np.std(history['success_rates'][-20:]) / max(np.mean(history['success_rates'][-20:]), 0.01),
                    1.0)
    consistency = min(1 - np.std(test_results) / max(np.mean(test_results), 0.01), 1.0)

    scores = [final_success, test_success, stability, consistency]
    labels = ['Training\nSuccess', 'Test\nSuccess', 'Training\nStability', 'Test\nConsistency']

    angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
    scores += scores[:1]  # 闭合图形
    angles += angles[:1]

    ax = plt.subplot(2, 4, 8, projection='polar')
    ax.plot(angles, scores, 'o-', linewidth=2, color='blue')
    ax.fill(angles, scores, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar', pad=20)

    plt.tight_layout()
    plt.savefig('final_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 保存最终综合分析图: final_comprehensive_analysis.png")


if __name__ == "__main__":
    start_time = time.time()

    overall_success, consistency = comprehensive_final_test()

    total_time = time.time() - start_time

    print(f"\n🏁 最终综合测试总结:")
    print(f"   总用时: {total_time / 60:.1f} 分钟")
    print(f"   综合成功率: {overall_success:.1%}")
    print(f"   测试一致性: {consistency:.1%}")

    # 最终评价
    if overall_success >= 0.25 and consistency >= 0.7:
        final_grade = "A+ 优秀"
        final_icon = "🎉"
    elif overall_success >= 0.2 and consistency >= 0.6:
        final_grade = "A 良好"
        final_icon = "🎯"
    elif overall_success >= 0.15 and consistency >= 0.5:
        final_grade = "B+ 可接受"
        final_icon = "✅"
    elif overall_success >= 0.1:
        final_grade = "B 需改进"
        final_icon = "⚠️"
    else:
        final_grade = "C 不理想"
        final_icon = "❌"

    print(f"\n{final_icon} 最终评价: {final_grade}")
    print(f"   ACO-PPO系统 {'成功实现目标' if overall_success >= 0.2 else '仍需优化'}")