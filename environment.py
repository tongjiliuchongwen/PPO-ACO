# environment.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import config

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
        self.v0 = config.V0  # 基础速度
        self.omega_max = config.OMEGA_MAX  # 最大角速度
        self.dt = config.DT  # 时间步长
        self.target_radius = config.TARGET_RADIUS
        
        # 奖励参数
        self.step_penalty = config.STEP_PENALTY
        self.target_reward = config.TARGET_REWARD
        self.collision_penalty = config.COLLISION_PENALTY
        
        # 动作空间：角速度 omega，范围 [-omega_max, omega_max]
        self.action_space = spaces.Box(
            low=-self.omega_max, 
            high=self.omega_max, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # 观测空间：[dx, dy, cos(theta), sin(theta), distance_to_target]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )
        
        # 状态变量
        self.agent_pos = np.zeros(2)  # [x, y]
        self.agent_theta = 0.0  # 朝向角度
        self.target_pos = np.zeros(2)  # 目标位置
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        
        # 随机生成智能体和目标位置，确保不在障碍物内部
        while True:
            self.agent_pos = np.random.uniform(
                -self.env_bounds + 1, 
                self.env_bounds - 1, 
                size=2
            )
            if not self._is_collision(self.agent_pos):
                break
                
        while True:
            self.target_pos = np.random.uniform(
                -self.env_bounds + 1, 
                self.env_bounds - 1, 
                size=2
            )
            # 确保目标不在障碍物内部，且距离智能体足够远
            if (not self._is_collision(self.target_pos) and 
                np.linalg.norm(self.target_pos - self.agent_pos) > 3.0):
                break
        
        # 随机初始朝向
        self.agent_theta = np.random.uniform(0, 2 * np.pi)
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """执行一步动作"""
        self.current_step += 1
        
        # 限制动作范围
        omega = np.clip(action[0], -self.omega_max, self.omega_max)
        
        # 更新朝向（加入旋转噪声）
        if self.enable_noise:
            rot_noise = np.random.normal(0, self.rot_noise_std * np.sqrt(self.dt))
        else:
            rot_noise = 0.0
        
        self.agent_theta += (omega + rot_noise) * self.dt
        self.agent_theta = self.agent_theta % (2 * np.pi)  # 保持在 [0, 2π] 范围内
        
        # 计算位移（朗之万动力学）
        velocity = np.array([
            self.v0 * np.cos(self.agent_theta),
            self.v0 * np.sin(self.agent_theta)
        ])
        
        # 加入平动噪声
        if self.enable_noise:
            trans_noise = np.random.normal(
                0, self.trans_noise_std * np.sqrt(self.dt), size=2
            )
        else:
            trans_noise = np.zeros(2)
        
        # 更新位置
        new_pos = self.agent_pos + (velocity + trans_noise) * self.dt
        
        # 边界检查
        new_pos = np.clip(new_pos, -self.env_bounds, self.env_bounds)
        
        # 碰撞检测
        collision = self._is_collision(new_pos)
        
        if collision:
            # 碰撞时不更新位置，给予负奖励并结束回合
            reward = self.collision_penalty
            done = True
            truncated = False
        else:
            # 更新位置
            self.agent_pos = new_pos
            
            # 检查是否到达目标
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
                # 基础时间惩罚，可以加入距离相关的奖励塑形
                reward = self.step_penalty
                # 可选：加入距离奖励塑形
                # reward += -0.001 * distance_to_target
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
        # 相对位置
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        
        # 朝向的三角函数表示
        cos_theta = np.cos(self.agent_theta)
        sin_theta = np.sin(self.agent_theta)
        
        # 到目标的距离
        distance_to_target = np.linalg.norm(self.target_pos - self.agent_pos)
        
        return np.array([dx, dy, cos_theta, sin_theta, distance_to_target], dtype=np.float32)
    
    def _is_collision(self, position):
        """检查给定位置是否与障碍物碰撞"""
        for obs_x, obs_y, obs_radius in self.obstacles:
            distance = np.linalg.norm(position - np.array([obs_x, obs_y]))
            if distance < obs_radius:
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