# environment.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import config
# 导入 ACOSystem 只是为了类型提示，避免循环导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from aco_system import ACOSystem

class ActiveParticleEnv(gym.Env):
    """
    智能粒子环境类，实现朗之万动力学和障碍物碰撞检测
    """
    
    def __init__(self):
        super(ActiveParticleEnv, self).__init__()
        
        # 从配置文件加载参数
        self.env_bounds = config.ENV_BOUNDS
        self.max_steps = config.MAX_STEPS_PER_EPISODE
        self.obstacles = config.OBSTACLES
        self.enable_noise = config.ENABLE_NOISE
        self.trans_noise_std = config.TRANSLATIONAL_NOISE_STD
        self.rot_noise_std = config.ROTATIONAL_NOISE_STD
        
        # 物理参数
        self.v0 = config.V0
        self.omega_max = config.OMEGA_MAX
        self.dt = config.DT
        self.target_radius = config.TARGET_RADIUS
        
        # 奖励参数
        self.step_penalty = config.STEP_PENALTY
        self.target_reward = config.TARGET_REWARD
        self.collision_penalty = config.COLLISION_PENALTY
        
        # 动作空间
        self.action_space = spaces.Box(
            low=-self.omega_max, 
            high=self.omega_max, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )
        
        # 状态变量
        self.agent_pos = np.zeros(2)
        self.agent_theta = 0.0
        self.target_pos = np.zeros(2)
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        
        # 固定目标位置
        self.target_pos = np.array([8.0, 8.0])
        
        # 随机化智能体初始位置
        while True:
            self.agent_pos = np.random.uniform(
                -self.env_bounds + 1, 
                self.env_bounds - 1, 
                size=2
            )
            if (not self._is_collision(self.agent_pos) and
                np.linalg.norm(self.target_pos - self.agent_pos) > 3.0):
                break
                
        # 随机初始朝向
        self.agent_theta = np.random.uniform(0, 2 * np.pi)
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action, aco_system: 'ACOSystem' = None):
        """
        执行一步动作，奖励函数基于区域平均信息素浓度变化
        """
        self.current_step += 1
        
        omega = np.clip(action[0], -self.omega_max, self.omega_max)
        
        # 在移动前，获取当前位置的区域平均信息素浓度
        if aco_system is not None:
            previous_pheromone = aco_system.get_average_pheromone(self.agent_pos, kernel_size=config.ACO_KERNEL_SIZE)
        else:
            previous_pheromone = 0.0

        # 更新朝向
        if self.enable_noise:
            rot_noise = np.random.normal(0, self.rot_noise_std * np.sqrt(self.dt))
        else:
            rot_noise = 0.0
        self.agent_theta = (self.agent_theta + (omega + rot_noise) * self.dt) % (2 * np.pi)
        
        # 计算位移
        velocity = np.array([self.v0 * np.cos(self.agent_theta), self.v0 * np.sin(self.agent_theta)])
        if self.enable_noise:
            trans_noise = np.random.normal(0, self.trans_noise_std * np.sqrt(self.dt), size=2)
        else:
            trans_noise = np.zeros(2)
        
        # 预测新位置并检查边界
        new_pos = self.agent_pos + (velocity + trans_noise) * self.dt
        new_pos = np.clip(new_pos, -self.env_bounds, self.env_bounds)
        
        # 碰撞检测
        collision = self._is_collision(new_pos)
        
        if collision:
            reward = self.collision_penalty
            done = True
            truncated = False
        else:
            self.agent_pos = new_pos
            
            # 在移动后，获取新位置的区域平均信息素浓度
            if aco_system is not None:
                current_pheromone = aco_system.get_average_pheromone(self.agent_pos, kernel_size=config.ACO_KERNEL_SIZE)
            else:
                current_pheromone = 0.0

            distance_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
            
            if distance_to_target < self.target_radius:
                reward = self.target_reward
                done = True
                truncated = False
            elif self.current_step >= self.max_steps:
                reward = self.step_penalty
                done = False
                truncated = True
            else:
                # 基于信息素的奖励塑形
                pheromone_reward = current_pheromone - previous_pheromone
                shaping_scaling_factor = 500.0 # 这个值可能需要调试
                reward = self.step_penalty + (shaping_scaling_factor * pheromone_reward)
                done = False
                truncated = False
        
        observation = self._get_observation()
        info = {
            'collision': collision,
            'distance_to_target': np.linalg.norm(self.agent_pos - self.target_pos),
            'agent_pos': self.agent_pos.copy(),
            'target_pos': self.target_pos.copy()
        }
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self):
        """获取当前观测"""
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        
        cos_theta = np.cos(self.agent_theta)
        sin_theta = np.sin(self.agent_theta)
        
        distance_to_target = np.linalg.norm(self.target_pos - self.agent_pos)
        
        # 归一化观测值
        dx_norm = dx / self.env_bounds
        dy_norm = dy / self.env_bounds
        dist_norm = distance_to_target / (np.sqrt(2) * self.env_bounds)

        return np.array([dx_norm, dy_norm, cos_theta, sin_theta, dist_norm], dtype=np.float32)

    def _is_collision(self, position):
        """检查给定位置是否与障碍物碰撞"""
        for obs_x, obs_y, obs_radius in self.obstacles:
            if np.linalg.norm(position - np.array([obs_x, obs_y])) < obs_radius:
                return True
        return False
    
    def get_agent_position(self):
        """获取智能体当前位置"""
        return self.agent_pos.copy()
    
    def get_target_position(self):
        """获取目标位置"""
        return self.target_pos.copy()
    
    def get_agent_orientation(self):
        """获取智能体朝向"""
        return self.agent_theta