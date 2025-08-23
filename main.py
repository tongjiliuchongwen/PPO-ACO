# main.py - RL Navigation Platform V3 Main Training Script
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import config
from environment import ActiveParticleEnv
from ppo_agent import PPOAgent


def setup_gpu_optimizations():
    """Setup GPU optimizations for RTX 4080 SUPER"""
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuning for optimal convolution algorithms
        if config.ENABLE_CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
            print("🔧 cuDNN benchmark enabled")
        
        # Set device
        device = torch.device('cuda')
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Adjust batch size based on available memory if configured for large batches
        if config.TIMESTEPS_PER_BATCH >= config.MAX_BATCH_SIZE_GPU:
            print(f"⚡ Large batch training: {config.TIMESTEPS_PER_BATCH} timesteps/batch")
        
        return device
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device('cpu')


def create_save_directory():
    """Create directory for saving models and logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/{config.EXPERIMENT_MODE}_{config.AGENT_ARCHITECTURE}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    config_path = os.path.join(save_dir, "config.txt")
    with open(config_path, 'w') as f:
        for attr in dir(config):
            if not attr.startswith('_'):
                f.write(f"{attr} = {getattr(config, attr)}\n")
    
    print(f"📁 Experiment directory: {save_dir}")
    return save_dir


def print_experiment_info():
    """Print current experiment configuration"""
    print("=" * 80)
    print(f"🧪 RL NAVIGATION PLATFORM V3")
    print(f"   Experiment Mode: {config.EXPERIMENT_MODE}")
    print(f"   Agent Architecture: {config.AGENT_ARCHITECTURE}")
    print(f"   Reward Function: {config.REWARD_FUNCTION_TYPE}")
    print("=" * 80)
    
    # Explain the current experiment
    if config.EXPERIMENT_MODE == 'BASELINE':
        print("🎯 BASELINE EXPERIMENT:")
        print("   - Fixed target position")
        print("   - Testing basic navigation ability")
        print("   - Standard MLP should perform well")
        
    elif config.EXPERIMENT_MODE == 'SEARCH_RL':
        print("🔍 SEARCH_RL EXPERIMENT:")
        print("   - Random target positions (blind search)")
        print("   - Testing memory importance")
        print("   - LSTM should outperform MLP")
        
    elif config.EXPERIMENT_MODE == 'SEARCH_HYBRID':
        print("🐜 SEARCH_HYBRID EXPERIMENT:")
        print("   - Random targets + ACO pheromone guidance")
        print("   - Testing collective experience value")
        print("   - ACO should accelerate LSTM learning")
    
    print()


def train_baseline_mode(env, agent, save_dir):
    """Training loop for BASELINE mode"""
    print("🏃‍♂️ Starting BASELINE training...")
    
    total_iterations = config.ITERATIONS_PER_RESET
    training_stats = {
        'iterations': [],
        'mean_rewards': [],
        'success_rates': [],
        'episode_lengths': []
    }
    
    for iteration in range(total_iterations):
        start_time = time.time()
        
        # Collect rollout data
        batch_data = agent.rollout()
        
        # Update networks
        agent.update_networks(batch_data)
        
        # Get statistics
        stats = agent.get_statistics()
        
        # Record training progress
        training_stats['iterations'].append(iteration)
        training_stats['mean_rewards'].append(stats.get('mean_episode_reward', 0))
        training_stats['success_rates'].append(stats.get('success_rate', 0))
        training_stats['episode_lengths'].append(stats.get('mean_episode_length', 0))
        
        elapsed_time = time.time() - start_time
        
        # Logging
        if iteration % config.LOG_FREQUENCY == 0:
            print(f"Iteration {iteration:4d} | "
                  f"Reward: {stats.get('mean_episode_reward', 0):6.1f} | "
                  f"Success: {stats.get('success_rate', 0):5.1%} | "
                  f"Episodes: {stats.get('total_episodes', 0):4d} | "
                  f"Time: {elapsed_time:.2f}s")
        
        # Save model periodically
        if iteration % config.SAVE_FREQUENCY == 0 and config.SAVE_MODELS:
            model_path = os.path.join(save_dir, f"model_iteration_{iteration}.pth")
            agent.save_model(model_path)
    
    return training_stats


def train_search_mode(env, agent, save_dir):
    """Training loop for SEARCH_RL and SEARCH_HYBRID modes"""
    print(f"🔍 Starting {config.EXPERIMENT_MODE} training...")
    
    total_major_resets = config.TOTAL_MAJOR_RESETS
    iterations_per_reset = config.ITERATIONS_PER_RESET
    
    training_stats = {
        'major_resets': [],
        'iterations': [],
        'mean_rewards': [],
        'success_rates': [],
        'episode_lengths': []
    }
    
    global_iteration = 0
    
    for major_reset in range(total_major_resets):
        print(f"\n🔄 Major Reset {major_reset + 1}/{total_major_resets}")
        
        # Perform major reset (new random target)
        env.reset(major_reset=True)
        
        # Reset ACO pheromones if in hybrid mode
        if config.EXPERIMENT_MODE == 'SEARCH_HYBRID' and hasattr(env, 'aco_system'):
            env.aco_system.reset_pheromones()
        
        # Training iterations within this major reset
        for iteration in range(iterations_per_reset):
            start_time = time.time()
            
            # Collect rollout data
            batch_data = agent.rollout()
            
            # Update ACO system with successful trajectories (in SEARCH_HYBRID mode)
            if config.EXPERIMENT_MODE == 'SEARCH_HYBRID' and hasattr(env, 'aco_system'):
                update_aco_with_successful_trajectories(env.aco_system, batch_data)
            
            # Update networks
            agent.update_networks(batch_data)
            
            # Get statistics
            stats = agent.get_statistics()
            
            # Record training progress
            training_stats['major_resets'].append(major_reset)
            training_stats['iterations'].append(global_iteration)
            training_stats['mean_rewards'].append(stats.get('mean_episode_reward', 0))
            training_stats['success_rates'].append(stats.get('success_rate', 0))
            training_stats['episode_lengths'].append(stats.get('mean_episode_length', 0))
            
            elapsed_time = time.time() - start_time
            
            # Logging
            if global_iteration % config.LOG_FREQUENCY == 0:
                print(f"Reset {major_reset:2d} | Iter {iteration:3d} | "
                      f"Reward: {stats.get('mean_episode_reward', 0):6.1f} | "
                      f"Success: {stats.get('success_rate', 0):5.1%} | "
                      f"Episodes: {stats.get('total_episodes', 0):4d} | "
                      f"Time: {elapsed_time:.2f}s")
            
            # Save model periodically
            if global_iteration % config.SAVE_FREQUENCY == 0 and config.SAVE_MODELS:
                model_path = os.path.join(save_dir, f"model_iteration_{global_iteration}.pth")
                agent.save_model(model_path)
            
            global_iteration += 1
    
    return training_stats


def update_aco_with_successful_trajectories(aco_system, batch_data):
    """Update ACO system with successful trajectories from rollout"""
    # This is a simplified version - in practice, you'd track full trajectories
    # and identify successful episodes for pheromone deposition
    
    observations = batch_data['observations']
    rewards = batch_data['rewards']
    
    # Identify successful steps (positive rewards above threshold)
    successful_indices = np.where(rewards > config.TARGET_REWARD * 0.1)[0]
    
    if len(successful_indices) > 0:
        # Extract successful positions and deposit navigation pheromone
        successful_positions = []
        for idx in successful_indices:
            if idx < len(observations):
                obs = observations[idx]
                # Convert observation back to world position (simplified)
                # In practice, you'd need to track absolute positions
                position = np.array([obs[0], obs[1]], dtype=np.float32)  # dx, dy from observation
                successful_positions.append(position)
        
        if successful_positions:
            aco_system.deposit_navigation_pheromone(successful_positions, success_quality=1.0)


def create_training_plots(training_stats, save_dir):
    """Create and save training visualization plots"""
    if not config.VISUALIZE_TRAINING:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    iterations = training_stats['iterations']
    
    # Reward plot
    axes[0, 0].plot(iterations, training_stats['mean_rewards'])
    axes[0, 0].set_title('Mean Episode Reward')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Success rate plot
    axes[0, 1].plot(iterations, training_stats['success_rates'])
    axes[0, 1].set_title('Success Rate')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].grid(True)
    
    # Episode length plot
    axes[1, 0].plot(iterations, training_stats['episode_lengths'])
    axes[1, 0].set_title('Mean Episode Length')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True)
    
    # Learning curve (smoothed reward)
    if len(training_stats['mean_rewards']) > 10:
        window = min(50, len(training_stats['mean_rewards']) // 10)
        smoothed_rewards = np.convolve(training_stats['mean_rewards'], 
                                     np.ones(window)/window, mode='valid')
        smoothed_iterations = iterations[window-1:]
        axes[1, 1].plot(smoothed_iterations, smoothed_rewards)
        axes[1, 1].set_title(f'Smoothed Learning Curve (window={window})')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Smoothed Reward')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{config.EXPERIMENT_MODE}_{config.AGENT_ARCHITECTURE}_training.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to: {plot_path}")


def main():
    """Main training function"""
    # Print experiment information
    print_experiment_info()
    
    # Setup GPU optimizations
    device = setup_gpu_optimizations()
    
    # Create save directory
    save_dir = create_save_directory()
    
    # Create environment and agent
    env = ActiveParticleEnv()
    agent = PPOAgent(env, device=device)
    
    # Start training based on experiment mode
    start_time = time.time()
    
    if config.EXPERIMENT_MODE == 'BASELINE':
        training_stats = train_baseline_mode(env, agent, save_dir)
    else:  # SEARCH_RL or SEARCH_HYBRID
        training_stats = train_search_mode(env, agent, save_dir)
    
    total_time = time.time() - start_time
    
    # Save final model
    if config.SAVE_MODELS:
        final_model_path = os.path.join(save_dir, f"{config.EXPERIMENT_MODE}_{config.AGENT_ARCHITECTURE}_final.pth")
        agent.save_model(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
    
    # Create training plots
    create_training_plots(training_stats, save_dir)
    
    # Final statistics
    final_stats = agent.get_statistics()
    print("\n" + "=" * 80)
    print("🏁 TRAINING COMPLETED")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Final mean reward: {final_stats.get('mean_episode_reward', 0):.2f}")
    print(f"   Final success rate: {final_stats.get('success_rate', 0):.1%}")
    print(f"   Total episodes: {final_stats.get('total_episodes', 0)}")
    print("=" * 80)


if __name__ == "__main__":
    main()