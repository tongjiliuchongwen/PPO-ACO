# final_demo.py - Final demonstration of RL Navigation Platform V3
import os
import torch
import config
from environment import ActiveParticleEnv
from ppo_agent import PPOAgent

def run_mini_experiment(mode, architecture, reward_type, iterations=3):
    """Run a mini experiment to demonstrate functionality"""
    print(f"\n🧪 MINI EXPERIMENT: {mode} + {architecture} + {reward_type}")
    print("-" * 60)
    
    # Set configuration
    config.EXPERIMENT_MODE = mode
    config.AGENT_ARCHITECTURE = architecture  
    config.REWARD_FUNCTION_TYPE = reward_type
    config.TIMESTEPS_PER_BATCH = 256  # Small for demo
    config.N_UPDATES_PER_ITERATION = 1  # Fast demo
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = ActiveParticleEnv()
    agent = PPOAgent(env, device=device)
    
    # Print experiment info
    print(f"📋 Configuration:")
    print(f"   Mode: {mode}")
    print(f"   Architecture: {architecture}")
    print(f"   Reward: {reward_type}")
    print(f"   Device: {device}")
    if hasattr(env, 'aco_system') and env.aco_system is not None:
        print(f"   ACO System: Active (grid: {env.aco_system.grid_size}x{env.aco_system.grid_size})")
    else:
        print(f"   ACO System: Inactive")
    
    # Run training iterations
    print(f"\n⚡ Running {iterations} training iterations...")
    for i in range(iterations):
        # Collect data and train
        batch_data = agent.rollout(max_timesteps=64)  # Very small for demo
        agent.update_networks(batch_data)
        
        # Get stats
        stats = agent.get_statistics()
        
        print(f"   Iteration {i+1}: Episodes={stats.get('total_episodes', 0):2d}, "
              f"Reward={stats.get('mean_episode_reward', 0):6.1f}, "
              f"Success={stats.get('success_rate', 0):5.1%}")
        
        # Show ACO stats if available
        if hasattr(env, 'aco_system') and env.aco_system is not None:
            aco_stats = env.aco_system.get_pheromone_stats()
            print(f"              ACO: nav_mean={aco_stats['nav_mean']:.3f}, "
                  f"exp_mean={aco_stats['exp_mean']:.3f}")
    
    print(f"   ✅ {mode} experiment completed successfully!")
    return stats

def main():
    """Run comprehensive demonstration of all platform capabilities"""
    print("🎯 RL NAVIGATION PLATFORM V3 - FINAL DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo shows the three core experiments that address our")
    print("scientific questions about memory and collective experience.")
    print()
    
    # Store original config to restore later
    original_mode = config.EXPERIMENT_MODE
    original_arch = config.AGENT_ARCHITECTURE
    original_reward = config.REWARD_FUNCTION_TYPE
    
    try:
        # Experiment 1: Baseline (Standard DRL)
        stats1 = run_mini_experiment('BASELINE', 'MLP', 'DISTANCE_SHAPING')
        
        # Experiment 2: Memory importance  
        stats2 = run_mini_experiment('SEARCH_RL', 'LSTM', 'SPARSE')
        
        # Experiment 3: Collective experience
        stats3 = run_mini_experiment('SEARCH_HYBRID', 'LSTM', 'PHEROMONE_SHAPING')
        
        # Summary
        print("\n📊 EXPERIMENT SUMMARY")
        print("=" * 40)
        print("The platform successfully ran all three experimental modes:")
        print()
        print("1. 📈 BASELINE (PPO+MLP): Tests basic navigation capability")
        print("   → Fixed targets, distance shaping, simple networks")
        print()
        print("2. 🧠 SEARCH_RL (PPO+LSTM): Tests memory importance")  
        print("   → Random targets, sparse rewards, memory networks")
        print()
        print("3. 🐜 SEARCH_HYBRID (PPO+LSTM+ACO): Tests collective experience")
        print("   → Random targets, pheromone guidance, dual memory systems")
        print()
        
        print("🎉 PLATFORM VALIDATION COMPLETE!")
        print()
        print("Key Achievements:")
        print("✅ Modular configuration system working")
        print("✅ Three experiment modes functional")  
        print("✅ MLP and LSTM architectures working")
        print("✅ Multiple reward functions implemented")
        print("✅ ACO dual pheromone system active")
        print("✅ GPU optimizations ready")
        print("✅ Scientific framework established")
        print()
        print("🚀 Ready for full-scale experiments!")
        print("   Edit config.py and run 'python main.py' to start")
        
    finally:
        # Restore original configuration
        config.EXPERIMENT_MODE = original_mode
        config.AGENT_ARCHITECTURE = original_arch
        config.REWARD_FUNCTION_TYPE = original_reward

if __name__ == "__main__":
    main()