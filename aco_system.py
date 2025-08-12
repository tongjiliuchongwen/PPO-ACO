# aco_system.py

import numpy as np
import config

class ACOSystem:
    """
    蚁群优化系统，管理信息素地图的更新和查询
    """
    
    def __init__(self, grid_size=None, env_bounds=None):
        """
        初始化ACO系统
        
        Args:
            grid_size: 信息素网格大小，默认从config读取
            env_bounds: 环境边界，默认从config读取
        """
        self.grid_size = grid_size or config.GRID_SIZE
        self.env_bounds = env_bounds or config.ENV_BOUNDS
        
        # 信息素地图初始化为小的正值
        self.pheromone_map = np.ones((self.grid_size, self.grid_size)) * 0.1
        
        # 从配置文件读取参数
        self.evaporation_rate = config.EVAPORATION_RATE
        self.deposit_amount = config.DEPOSIT_AMOUNT
        
        # 计算网格分辨率
        self.grid_resolution = (2 * self.env_bounds) / self.grid_size
        
        # 历史统计
        self.total_deposits = 0
        self.total_evaporation_steps = 0
    
    def world_to_grid(self, position):
        """
        将世界坐标转换为网格坐标
        
        Args:
            position: 世界坐标 [x, y]
        
        Returns:
            grid_coords: 网格坐标 [grid_x, grid_y]
        """
        # 将世界坐标从 [-env_bounds, env_bounds] 映射到 [0, grid_size-1]
        normalized_pos = (np.array(position) + self.env_bounds) / (2 * self.env_bounds)
        grid_coords = (normalized_pos * (self.grid_size - 1)).astype(int)
        
        # 确保在有效范围内
        grid_coords = np.clip(grid_coords, 0, self.grid_size - 1)
        
        return grid_coords
    
    def grid_to_world(self, grid_coords):
        """
        将网格坐标转换为世界坐标
        
        Args:
            grid_coords: 网格坐标 [grid_x, grid_y]
        
        Returns:
            position: 世界坐标 [x, y]
        """
        # 将网格坐标从 [0, grid_size-1] 映射到 [-env_bounds, env_bounds]
        normalized_pos = np.array(grid_coords) / (self.grid_size - 1)
        position = normalized_pos * (2 * self.env_bounds) - self.env_bounds
        
        return position
    
    def get_pheromone_value(self, position):
        """
        获取指定位置的信息素浓度
        
        Args:
            position: 世界坐标 [x, y]
        
        Returns:
            pheromone_value: 该位置的信息素浓度
        """
        grid_coords = self.world_to_grid(position)
        return self.pheromone_map[grid_coords[1], grid_coords[0]]  # 注意y,x顺序
    
    # >>>>> 新增的核心函数 <<<<<
    def get_average_pheromone(self, position, kernel_size=3):
        """
        获取指定位置周围一个区域内的平均信息素浓度。

        Args:
            position: 世界坐标 [x, y]
            kernel_size: 感知区域的边长（必须是奇数，如3, 5, 7）

        Returns:
            avg_pheromone: 该区域的平均信息素浓度
        """
        if kernel_size % 2 == 0:
            # 确保kernel_size是奇数，以便有明确的中心点
            kernel_size += 1
            # raise ValueError("Kernel size must be an odd number.")

        # 获取中心点的网格坐标
        center_gx, center_gy = self.world_to_grid(position)

        # 计算感知区域的边界
        radius = kernel_size // 2
        min_x = max(0, center_gx - radius)
        max_x = min(self.grid_size - 1, center_gx + radius)
        min_y = max(0, center_gy - radius)
        max_y = min(self.grid_size - 1, center_gy + radius)

        # 提取该区域的信息素值
        pheromone_patch = self.pheromone_map[min_y : max_y + 1, min_x : max_x + 1]
        
        # 计算平均值
        if pheromone_patch.size == 0:
            return 0.1 # 如果区域无效（不太可能发生），返回一个默认值
        
        return np.mean(pheromone_patch)

    def get_pheromone_gradient(self, position):
        """
        获取指定位置的信息素梯度（用于梯度引导）
        
        Args:
            position: 世界坐标 [x, y]
        
        Returns:
            gradient: 信息素梯度 [grad_x, grad_y]
        """
        grid_coords = self.world_to_grid(position)
        gx, gy = grid_coords[0], grid_coords[1]
        
        # 计算梯度（使用有限差分）
        grad_x = 0.0
        grad_y = 0.0
        
        if gx > 0 and gx < self.grid_size - 1:
            grad_x = (self.pheromone_map[gy, gx + 1] - self.pheromone_map[gy, gx - 1]) / 2.0
        elif gx == 0:
            grad_x = self.pheromone_map[gy, gx + 1] - self.pheromone_map[gy, gx]
        else:
            grad_x = self.pheromone_map[gy, gx] - self.pheromone_map[gy, gx - 1]
        
        if gy > 0 and gy < self.grid_size - 1:
            grad_y = (self.pheromone_map[gy + 1, gx] - self.pheromone_map[gy - 1, gx]) / 2.0
        elif gy == 0:
            grad_y = self.pheromone_map[gy + 1, gx] - self.pheromone_map[gy, gx]
        else:
            grad_y = self.pheromone_map[gy, gx] - self.pheromone_map[gy - 1, gx]
        
        return np.array([grad_x, grad_y])
    
    def deposit(self, trajectory, path_quality=None):
        """
        在成功路径上沉积信息素
        
        Args:
            trajectory: 路径轨迹
            path_quality: 路径质量
        """
        if len(trajectory) == 0:
            return
        
        if path_quality is None:
            path_length = sum(np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1])) for i in range(1, len(trajectory)))
            path_quality = 1.0 / (path_length + 1e-6)
        
        deposit_per_point = self.deposit_amount * path_quality / len(trajectory)
        
        for position in trajectory:
            gx, gy = self.world_to_grid(position)
            self._deposit_gaussian(gx, gy, deposit_per_point)
        
        self.total_deposits += 1
    
    def _deposit_gaussian(self, center_x, center_y, amount, sigma=1.0):
        """
        在指定中心周围以高斯分布沉积信息素
        """
        radius = int(3 * sigma)
        
        x_range = range(max(0, center_x - radius), min(self.grid_size, center_x + radius + 1))
        y_range = range(max(0, center_y - radius), min(self.grid_size, center_y + radius + 1))
        
        for i in x_range:
            for j in y_range:
                dist_sq = (i - center_x)**2 + (j - center_y)**2
                weight = np.exp(-dist_sq / (2 * sigma**2))
                self.pheromone_map[j, i] += amount * weight
    
    def evaporate(self):
        """
        信息素蒸发
        """
        self.pheromone_map *= (1 - self.evaporation_rate)
        self.pheromone_map = np.maximum(self.pheromone_map, 0.01)
        self.total_evaporation_steps += 1
    
    def reset(self):
        """
        重置信息素地图
        """
        self.pheromone_map = np.ones((self.grid_size, self.grid_size)) * 0.1
        self.total_deposits = 0
        self.total_evaporation_steps = 0
    
    def get_statistics(self):
        """
        获取ACO系统统计信息
        """
        return {
            'mean_pheromone': np.mean(self.pheromone_map),
            'max_pheromone': np.max(self.pheromone_map),
            'min_pheromone': np.min(self.pheromone_map),
            'std_pheromone': np.std(self.pheromone_map),
            'total_deposits': self.total_deposits,
            'total_evaporation_steps': self.total_evaporation_steps
        }
    
    def save_pheromone_map(self, filename):
        """保存信息素地图到文件"""
        np.save(filename, self.pheromone_map)
    
    def load_pheromone_map(self, filename):
        """从文件加载信息素地图"""
        self.pheromone_map = np.load(filename)

if __name__ == "__main__":
    # 测试ACO系统
    aco = ACOSystem()
    
    print(f"网格大小: {aco.grid_size}")
    print(f"环境边界: {aco.env_bounds}")
    print(f"网格分辨率: {aco.grid_resolution}")
    
    # 测试坐标转换
    world_pos = [5.0, -3.0]
    grid_pos = aco.world_to_grid(world_pos)
    world_pos_back = aco.grid_to_world(grid_pos)
    
    print(f"世界坐标: {world_pos}")
    print(f"网格坐标: {grid_pos}")
    print(f"转换回世界坐标: {world_pos_back}")
    
    # 测试信息素查询
    pheromone_value = aco.get_pheromone_value(world_pos)
    print(f"信息素浓度: {pheromone_value}")
    
    # 测试轨迹沉积
    trajectory = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ]
    aco.deposit(trajectory)
    
    # 测试蒸发
    aco.evaporate()
    
    # 显示统计信息
    stats = aco.get_statistics()
    print(f"ACO统计信息: {stats}")