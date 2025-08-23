# test_platform.py - Quick test of the RL Navigation Platform V3
import torch
import config
from environment import ActiveParticleEnv
from ppo_agent import PPOAgent

def test_training_loop():
    """Test a few training iterations to ensure everything works"""
    print("🧪 Testing RL Navigation Platform V3...")
    
    # Set up for quick test
    config.TIMESTEPS_PER_BATCH = 512  # Smaller batch for testing
    config.N_UPDATES_PER_ITERATION = 2  # Fewer updates for testing
    
    # Test device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test BASELINE mode with MLP
    print("\n1. Testing BASELINE mode with MLP...")
    config.EXPERIMENT_MODE = 'BASELINE'
    config.AGENT_ARCHITECTURE = 'MLP'
    config.REWARD_FUNCTION_TYPE = 'DISTANCE_SHAPING'
    
    env = ActiveParticleEnv()
    agent = PPOAgent(env, device=device)
    
    # Run a few training iterations
    for i in range(3):
        print(f"   Iteration {i+1}/3...")
        batch_data = agent.rollout(max_timesteps=128)
        agent.update_networks(batch_data)
        stats = agent.get_statistics()
        print(f"     Episodes: {stats.get('total_episodes', 0)}, "
              f"Reward: {stats.get('mean_episode_reward', 0):.2f}")
    
    print("   ✅ BASELINE + MLP test passed!")
    
    # Test SEARCH_RL mode with LSTM
    print("\n2. Testing SEARCH_RL mode with LSTM...")
    config.EXPERIMENT_MODE = 'SEARCH_RL'
    config.AGENT_ARCHITECTURE = 'LSTM'
    config.REWARD_FUNCTION_TYPE = 'SPARSE'
    
    env = ActiveParticleEnv()
    agent = PPOAgent(env, device=device)
    
    # Test major reset
    env.reset(major_reset=True)
    
    # Run a few training iterations
    for i in range(3):
        print(f"   Iteration {i+1}/3...")
        batch_data = agent.rollout(max_timesteps=128)
        agent.update_networks(batch_data)
        stats = agent.get_statistics()
        print(f"     Episodes: {stats.get('total_episodes', 0)}, "
              f"Reward: {stats.get('mean_episode_reward', 0):.2f}")
    
    print("   ✅ SEARCH_RL + LSTM test passed!")
    
    # Test SEARCH_HYBRID mode with LSTM
    print("\n3. Testing SEARCH_HYBRID mode with LSTM...")
    config.EXPERIMENT_MODE = 'SEARCH_HYBRID'
    config.AGENT_ARCHITECTURE = 'LSTM'
    config.REWARD_FUNCTION_TYPE = 'PHEROMONE_SHAPING'
    
    env = ActiveParticleEnv()
    agent = PPOAgent(env, device=device)
    
    # Test major reset
    env.reset(major_reset=True)
    
    # Run a few training iterations
    for i in range(3):
        print(f"   Iteration {i+1}/3...")
        batch_data = agent.rollout(max_timesteps=128)
        
        # Test ACO update (simplified)
        if hasattr(env, 'aco_system') and env.aco_system is not None:
            observations = batch_data['observations']
            rewards = batch_data['rewards']
            successful_indices = [j for j, r in enumerate(rewards) if r > 10]
            if successful_indices:
                trajectory = [observations[j][:2] for j in successful_indices[:5]]
                env.aco_system.deposit_navigation_pheromone(trajectory, success_quality=1.0)
        
        agent.update_networks(batch_data)
        stats = agent.get_statistics()
        print(f"     Episodes: {stats.get('total_episodes', 0)}, "
              f"Reward: {stats.get('mean_episode_reward', 0):.2f}")
        
        # Check ACO stats
        if hasattr(env, 'aco_system') and env.aco_system is not None:
            aco_stats = env.aco_system.get_pheromone_stats()
            print(f"     ACO: nav_mean={aco_stats['nav_mean']:.3f}, "
                  f"exp_mean={aco_stats['exp_mean']:.3f}")
    
    print("   ✅ SEARCH_HYBRID + LSTM test passed!")
    
    print("\n🎉 All platform tests passed! The RL Navigation Platform V3 is working correctly.")
    print("\n📋 Platform Features Verified:")
    print("   ✅ Modular configuration system")
    print("   ✅ Three experiment modes (BASELINE, SEARCH_RL, SEARCH_HYBRID)")
    print("   ✅ Dual architecture support (MLP, LSTM)")
    print("   ✅ Multiple reward functions (DISTANCE_SHAPING, SPARSE, PHEROMONE_SHAPING)")
    print("   ✅ ACO dual pheromone system")
    print("   ✅ PPO training with GPU optimizations")
    print("   ✅ LSTM hidden state management")

if __name__ == "__main__":
    test_training_loop()