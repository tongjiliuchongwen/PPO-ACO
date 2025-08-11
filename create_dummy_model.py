# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:38:05 2025

@author: Lenovo
"""

# create_dummy_model.py

import torch
import os
from network import ActorNetwork
import config

print("正在创建一个未经训练的 '哑巴' Actor 模型...")

# 确保 models 文件夹存在
os.makedirs('models', exist_ok=True)

# 根据环境定义模型的输入输出维度
# 观测空间: [dx, dy, cos(theta), sin(theta), distance_to_target] -> 5维
# 动作空间: [omega] -> 1维
obs_dim = 5
action_dim = 1

# 实例化 Actor 网络
# 它的权重是随机初始化的
dummy_actor = ActorNetwork(obs_dim, action_dim)

# 定义模型保存路径
model_path = 'models/dummy_actor.pth'

# 保存模型的状态字典
torch.save(dummy_actor.state_dict(), model_path)

print(f"成功！模型已保存到: {model_path}")
print("现在你可以用这个模型文件来进行测试了。")