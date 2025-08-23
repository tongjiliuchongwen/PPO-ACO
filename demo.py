# demo.py - RL Navigation Platform V3 Configuration Demo
"""
This demo shows how easy it is to switch between different experimental configurations
using the modular RL Navigation Platform V3.
"""

import config

def demo_experiment_switching():
    """Demonstrate easy experiment switching through config modification"""
    
    print("🎯 RL NAVIGATION PLATFORM V3 - CONFIGURATION DEMO")
    print("=" * 60)
    print()
    
    # Show the three core scientific questions
    print("🔬 CORE SCIENTIFIC QUESTIONS:")
    print()
    print("1. 📊 BASELINE: What is the performance of a standard DRL agent (PPO+MLP)")
    print("   in fixed-target navigation tasks?")
    print()
    print("2. 🧠 ROLE OF MEMORY: In random 'blind search' tasks, how much does")
    print("   short-term memory (PPO+LSTM) improve sample efficiency and generalization")
    print("   compared to memoryless agents?")
    print()
    print("3. 🐜 VALUE OF COLLECTIVE EXPERIENCE: In blind search tasks, can a dual")
    print("   pheromone system (ACO) acting as external 'long-term memory' further")
    print("   accelerate learning for memory-enabled agents?")
    print()
    
    # Demo 1: BASELINE Experiment
    print("🧪 EXPERIMENT 1: BASELINE")
    print("-" * 30)
    print("To run baseline experiments, simply set in config.py:")
    print("  EXPERIMENT_MODE = 'BASELINE'")
    print("  AGENT_ARCHITECTURE = 'MLP'")
    print("  REWARD_FUNCTION_TYPE = 'DISTANCE_SHAPING'")
    print()
    print("This will:")
    print("  • Use fixed target positions")
    print("  • Use simple MLP networks")
    print("  • Use distance-based reward shaping")
    print("  • Run single-level training loop")
    print()
    
    # Demo 2: SEARCH_RL Experiment  
    print("🧪 EXPERIMENT 2: SEARCH_RL (Memory Importance)")
    print("-" * 50)
    print("To test the importance of memory, set in config.py:")
    print("  EXPERIMENT_MODE = 'SEARCH_RL'")
    print("  AGENT_ARCHITECTURE = 'LSTM'")
    print("  REWARD_FUNCTION_TYPE = 'SPARSE'")
    print()
    print("This will:")
    print("  • Use random target positions (blind search)")
    print("  • Use LSTM networks with hidden state management")
    print("  • Use sparse rewards (only terminal rewards)")
    print("  • Run dual-loop training (major resets + iterations)")
    print()
    
    # Demo 3: SEARCH_HYBRID Experiment
    print("🧪 EXPERIMENT 3: SEARCH_HYBRID (Collective Experience)")
    print("-" * 55)
    print("To test collective experience value, set in config.py:")
    print("  EXPERIMENT_MODE = 'SEARCH_HYBRID'")
    print("  AGENT_ARCHITECTURE = 'LSTM'")
    print("  REWARD_FUNCTION_TYPE = 'PHEROMONE_SHAPING'")
    print()
    print("This will:")
    print("  • Use random target positions (blind search)")
    print("  • Use LSTM networks + ACO dual pheromone system")
    print("  • Use pheromone-based reward shaping")
    print("  • Include collective experience from successful trajectories")
    print()
    
    # Show GPU optimizations
    print("🚀 RTX 4080 SUPER OPTIMIZATIONS")
    print("-" * 35)
    print("The platform includes several GPU optimizations:")
    print("  • Mixed precision training (torch.cuda.amp)")
    print("  • Large batch sizes (TIMESTEPS_PER_BATCH = 4096+)")
    print("  • cuDNN auto-tuning (torch.backends.cudnn.benchmark)")
    print("  • Pinned memory for faster CPU-GPU transfers")
    print()
    
    # Show current configuration
    print("⚙️ CURRENT CONFIGURATION")
    print("-" * 25)
    print(f"  EXPERIMENT_MODE = '{config.EXPERIMENT_MODE}'")
    print(f"  AGENT_ARCHITECTURE = '{config.AGENT_ARCHITECTURE}'")
    print(f"  REWARD_FUNCTION_TYPE = '{config.REWARD_FUNCTION_TYPE}'")
    print(f"  TIMESTEPS_PER_BATCH = {config.TIMESTEPS_PER_BATCH}")
    print(f"  USE_MIXED_PRECISION = {config.USE_MIXED_PRECISION}")
    print()
    
    # Show how to run
    print("🏃‍♂️ HOW TO RUN EXPERIMENTS")
    print("-" * 25)
    print("1. Edit config.py to set your desired experiment mode")
    print("2. Run: python main.py")
    print("3. Results will be saved in experiments/ directory")
    print("4. Training plots and models are automatically saved")
    print()
    
    print("🎯 The platform makes it trivial to compare:")
    print("   • MLP vs LSTM architectures")
    print("   • Different reward functions")
    print("   • With/without ACO guidance")
    print("   • Various hyperparameter settings")
    print()
    print("All through simple config file changes! 🔧")

if __name__ == "__main__":
    demo_experiment_switching()