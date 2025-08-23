# environment.py - Active Particle Navigation Environment
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import config
from aco_system import ACOSystem


class ActiveParticleEnv(gym.Env):
    """Active particle navigation environment with configurable modes and reward functions"""
    
    def __init__(self):
        super().__init__()
        
        # Read configuration from config.py
        self.env_bounds = config.ENV_BOUNDS
        self.max_steps = config.MAX_STEPS_PER_EPISODE
        self.obstacles = config.OBSTACLES
        self.v0 = config.V0
        self.omega_max = config.OMEGA_MAX
        self.dt = config.DT
        self.target_radius = config.TARGET_RADIUS
        self.enable_noise = config.ENABLE_NOISE
        self.trans_noise_std = config.TRANSLATIONAL_NOISE_STD
        self.rot_noise_std = config.ROTATIONAL_NOISE_STD
        
        # Initialize ACO system if needed for SEARCH_HYBRID mode
        if config.EXPERIMENT_MODE == 'SEARCH_HYBRID':
            self.aco_system = ACOSystem(
                grid_size=config.ACO_GRID_SIZE,
                world_bounds=config.ENV_BOUNDS,
                nav_evaporation_rate=config.NAV_EVAPORATION_RATE,
                exp_evaporation_rate=config.EXP_EVAPORATION_RATE
            )
        else:
            self.aco_system = None
            
        # Define observation and action spaces
        # Observation: [dx, dy, cos(theta), sin(theta)] - 4D relative coordinates
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Action: [omega] - 1D angular velocity control
        self.action_space = spaces.Box(
            low=-self.omega_max, high=self.omega_max, shape=(1,), dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
        print(f"🌍 ActiveParticleEnv initialized:")
        print(f"   Mode: {config.EXPERIMENT_MODE}")
        print(f"   Reward: {config.REWARD_FUNCTION_TYPE}")
        print(f"   Bounds: [{-self.env_bounds}, {self.env_bounds}]")
        print(f"   Max steps: {self.max_steps}")
        
    def reset(self, seed=None, options=None, major_reset=False):
        """Reset environment state"""
        super().reset(seed=seed)
        
        # Initialize agent position and orientation
        self.position = np.array([0.0, 0.0], dtype=np.float32)
        self.theta = 0.0
        self.current_step = 0
        
        # Set target position based on experiment mode
        if config.EXPERIMENT_MODE == 'BASELINE':
            # Fixed target for baseline experiments
            self.target_position = np.array([5.0, 5.0], dtype=np.float32)
        elif config.EXPERIMENT_MODE in ['SEARCH_RL', 'SEARCH_HYBRID']:
            if major_reset or not hasattr(self, 'target_position'):
                # Generate new random target safely away from obstacles
                self.target_position = self._generate_safe_target()
        
        # Reset distance tracking for distance shaping reward
        self.prev_distance_to_target = np.linalg.norm(self.target_position - self.position)
        
        # Reset episode flags
        self.done = False
        self.target_reached = False
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        # Extract angular velocity from action
        omega = np.clip(action[0], -self.omega_max, self.omega_max)
        
        # Update position and orientation (particle physics)
        # Move -> Get new state -> Calculate reward -> Update exploration pheromone
        
        # 1. Move agent
        new_theta = self.theta + omega * self.dt
        new_position = self.position + self.v0 * self.dt * np.array([
            np.cos(new_theta), np.sin(new_theta)
        ])
        
        # Add noise if enabled
        if self.enable_noise:
            noise = np.random.normal(0, self.trans_noise_std, 2)
            new_position += noise
            new_theta += np.random.normal(0, self.rot_noise_std)
        
        # 2. Check collisions and boundaries
        collision = self._check_collision(new_position)
        boundary_collision = self._check_boundary(new_position)
        
        # Update position if no collision
        if not collision and not boundary_collision:
            self.position = new_position
            self.theta = new_theta
        
        # 3. Get new state
        obs = self._get_observation()
        
        # 4. Calculate reward based on configuration
        reward = self._calculate_reward(collision, boundary_collision)
        
        # 5. Update exploration pheromone (in ACO modes)
        if self.aco_system is not None:
            self.aco_system.deposit_exploration_pheromone(self.position)
            # Diffuse and evaporate pheromones
            self.aco_system.diffuse_pheromones()
            self.aco_system.evaporate()
        
        # Check if episode is done
        self.current_step += 1
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        
        if distance_to_target <= self.target_radius:
            self.target_reached = True
            self.done = True
            # Deposit navigation pheromone if successful in ACO modes
            if self.aco_system is not None:
                # This would be called from the training loop with full trajectory
                pass
        elif self.current_step >= self.max_steps or collision or boundary_collision:
            self.done = True
        
        info = {
            'collision': collision,
            'boundary_collision': boundary_collision,
            'target_reached': self.target_reached,
            'distance_to_target': distance_to_target,
            'current_step': self.current_step
        }
        
        return obs, reward, self.done, False, info
    
    def _get_observation(self):
        """Get 4D relative coordinate observation"""
        # Relative position to target
        dx = self.target_position[0] - self.position[0]
        dy = self.target_position[1] - self.position[1]
        
        # Agent orientation as cos/sin
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        return np.array([dx, dy, cos_theta, sin_theta], dtype=np.float32)
    
    def _calculate_reward(self, collision, boundary_collision):
        """Calculate reward based on configured reward function type"""
        reward = 0.0
        
        # Base penalties and rewards
        if self.target_reached:
            reward += config.TARGET_REWARD
        elif collision:
            reward += config.COLLISION_PENALTY
        elif boundary_collision:
            reward += config.BOUNDARY_PENALTY
        else:
            reward += config.STEP_PENALTY
        
        # Additional reward shaping based on type
        if config.REWARD_FUNCTION_TYPE == 'DISTANCE_SHAPING':
            # Distance-based reward shaping
            current_distance = np.linalg.norm(self.target_position - self.position)
            distance_improvement = self.prev_distance_to_target - current_distance
            reward += distance_improvement * config.DISTANCE_SHAPING_SCALING
            self.prev_distance_to_target = current_distance
            
        elif config.REWARD_FUNCTION_TYPE == 'PHEROMONE_SHAPING' and self.aco_system is not None:
            # Pheromone-based reward shaping
            nav_pheromone = self.aco_system.get_navigation_pheromone(self.position)
            exp_pheromone = self.aco_system.get_exploration_pheromone(self.position)
            
            reward += nav_pheromone * config.NAV_PHEROMONE_SCALING
            reward += exp_pheromone * config.EXP_PHEROMONE_SCALING
            
        # SPARSE reward type adds no additional shaping
        
        return reward
    
    def _check_collision(self, position):
        """Check collision with obstacles"""
        for obs_x, obs_y, obs_r in self.obstacles:
            distance = np.linalg.norm(position - np.array([obs_x, obs_y]))
            if distance <= obs_r:
                return True
        return False
    
    def _check_boundary(self, position):
        """Check boundary collision"""
        return (abs(position[0]) > self.env_bounds or 
                abs(position[1]) > self.env_bounds)
    
    def _generate_safe_target(self):
        """Generate a safe random target position away from obstacles"""
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random position within bounds
            target = np.random.uniform(
                -self.env_bounds + 1.0, 
                self.env_bounds - 1.0, 
                size=2
            )
            
            # Check if target is safe (away from obstacles and agent start)
            safe = True
            
            # Check distance from obstacles
            for obs_x, obs_y, obs_r in self.obstacles:
                if np.linalg.norm(target - np.array([obs_x, obs_y])) <= obs_r + 1.0:
                    safe = False
                    break
            
            # Check distance from starting position
            if safe and np.linalg.norm(target - np.array([0.0, 0.0])) < 2.0:
                safe = False
            
            if safe:
                return target.astype(np.float32)
        
        # Fallback to a safe default position
        return np.array([self.env_bounds - 2.0, self.env_bounds - 2.0], dtype=np.float32)
    
    def render(self, mode='human'):
        """Render the environment (placeholder)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass