# network.py

import torch
import torch.nn as nn
import numpy as np

class FeedForwardNN(nn.Module):
    """
    前馈神经网络基类，用于Actor和Critic网络
    """
    
    def __init__(self, input_dim, output_dim, activation=nn.ReLU):
        super(FeedForwardNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            activation(),
            nn.Linear(128, 128),
            activation(),
            nn.Linear(128, output_dim)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ActorNetwork(nn.Module):
    """
    Actor网络，输出动作的均值和标准差
    """
    
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        
        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 动作均值输出层
        self.mean_layer = nn.Linear(128, output_dim)
        
        # 动作标准差输出层（使用对数标准差）
        self.log_std_layer = nn.Linear(128, output_dim)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # 初始化均值层为较小的值
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        
        # 初始化标准差层
        nn.init.constant_(self.log_std_layer.weight, 0)
        nn.init.constant_(self.log_std_layer.bias, -0.5)
    
    def forward(self, x):
        """
        前向传播
        返回动作均值和对数标准差
        """
        shared_features = self.shared_layers(x)
        
        mean = self.mean_layer(shared_features)
        log_std = self.log_std_layer(shared_features)
        
        # 限制标准差范围
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return mean, std
    
    def get_action_and_log_prob(self, x):
        """
        获取动作和对数概率
        """
        mean, std = self.forward(x)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 采样动作
        action = dist.sample()
        
        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_action(self, x, action):
        """
        评估给定动作的对数概率和熵
        """
        mean, std = self.forward(x)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # 计算熵
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic网络，输出状态价值函数V(s)
    """
    
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # 最后一层使用较小的初始化
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)
    
    def forward(self, x):
        """前向传播，返回状态价值"""
        return self.network(x).squeeze(-1)


def create_networks(obs_dim, action_dim):
    """
    创建Actor和Critic网络的工厂函数
    
    Args:
        obs_dim: 观测空间维度
        action_dim: 动作空间维度
    
    Returns:
        actor: Actor网络
        critic: Critic网络
    """
    actor = ActorNetwork(obs_dim, action_dim)
    critic = CriticNetwork(obs_dim)
    
    return actor, critic


def count_parameters(model):
    """计算模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试网络创建
    obs_dim = 5  # [dx, dy, cos(theta), sin(theta), distance_to_target]
    action_dim = 1  # [omega]
    
    actor, critic = create_networks(obs_dim, action_dim)
    
    print(f"Actor网络参数数量: {count_parameters(actor)}")
    print(f"Critic网络参数数量: {count_parameters(critic)}")
    
    # 测试前向传播
    batch_size = 10
    obs = torch.randn(batch_size, obs_dim)
    
    # Actor测试
    action, log_prob = actor.get_action_and_log_prob(obs)
    print(f"动作形状: {action.shape}")
    print(f"对数概率形状: {log_prob.shape}")
    
    # Critic测试
    value = critic(obs)
    print(f"价值函数形状: {value.shape}")