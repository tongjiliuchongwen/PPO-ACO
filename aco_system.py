# aco_system.py - Dual Pheromone ACO System for Navigation
import numpy as np
import torch
import config


class ACOSystem:
    """Dual pheromone system for ACO-guided navigation"""
    
    def __init__(self, grid_size=50, world_bounds=10.0, 
                 nav_evaporation_rate=0.05, exp_evaporation_rate=0.3):
        self.grid_size = grid_size
        self.world_bounds = world_bounds
        self.grid_resolution = (2 * world_bounds) / grid_size
        
        # Dual pheromone maps
        self.nav_pheromone_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.exp_pheromone_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # ACO parameters
        self.nav_evaporation_rate = nav_evaporation_rate
        self.exp_evaporation_rate = exp_evaporation_rate
        self.nav_deposit_amount = config.NAV_DEPOSIT_AMOUNT
        self.exp_deposit_amount = config.EXP_DEPOSIT_AMOUNT
        self.kernel_size = config.ACO_KERNEL_SIZE
        self.diffusion_rate = config.ACO_DIFFUSION_RATE
        
        print(f"🐜 ACO System initialized:")
        print(f"   Grid: {grid_size}x{grid_size}")
        print(f"   World bounds: [{-world_bounds}, {world_bounds}]")
        print(f"   Resolution: {self.grid_resolution:.3f}")
        print(f"   Kernel size: {self.kernel_size}")
        
    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        x_normalized = (world_pos[0] + self.world_bounds) / (2 * self.world_bounds)
        y_normalized = (world_pos[1] + self.world_bounds) / (2 * self.world_bounds)
        
        x_idx = int(x_normalized * (self.grid_size - 1))
        y_idx = int(y_normalized * (self.grid_size - 1))
        
        # Clamp to valid grid range
        x_idx = max(0, min(self.grid_size - 1, x_idx))
        y_idx = max(0, min(self.grid_size - 1, y_idx))
        
        return x_idx, y_idx
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x_normalized = grid_x / (self.grid_size - 1)
        y_normalized = grid_y / (self.grid_size - 1)
        
        world_x = x_normalized * (2 * self.world_bounds) - self.world_bounds
        world_y = y_normalized * (2 * self.world_bounds) - self.world_bounds
        
        return world_x, world_y
    
    def deposit_navigation_pheromone(self, trajectory, success_quality=1.0):
        """Deposit navigation pheromone along successful trajectory"""
        if len(trajectory) == 0:
            return
            
        deposit_amount = self.nav_deposit_amount * success_quality
        
        for position in trajectory:
            grid_x, grid_y = self.world_to_grid(position)
            self.nav_pheromone_map[grid_y, grid_x] += deposit_amount
    
    def deposit_exploration_pheromone(self, position):
        """Deposit exploration pheromone at current position"""
        grid_x, grid_y = self.world_to_grid(position)
        self.exp_pheromone_map[grid_y, grid_x] += self.exp_deposit_amount
    
    def get_navigation_pheromone(self, position, kernel_size=None):
        """Get average navigation pheromone in kernel region around position"""
        if kernel_size is None:
            kernel_size = self.kernel_size
            
        center_x, center_y = self.world_to_grid(position)
        
        min_x = max(0, center_x - kernel_size)
        max_x = min(self.grid_size - 1, center_x + kernel_size)
        min_y = max(0, center_y - kernel_size)
        max_y = min(self.grid_size - 1, center_y + kernel_size)
        
        region = self.nav_pheromone_map[min_y:max_y + 1, min_x:max_x + 1]
        return float(np.mean(region))
    
    def get_exploration_pheromone(self, position, kernel_size=None):
        """Get average exploration pheromone in kernel region around position"""
        if kernel_size is None:
            kernel_size = self.kernel_size
            
        center_x, center_y = self.world_to_grid(position)
        
        min_x = max(0, center_x - kernel_size)
        max_x = min(self.grid_size - 1, center_x + kernel_size)
        min_y = max(0, center_y - kernel_size)
        max_y = min(self.grid_size - 1, center_y + kernel_size)
        
        region = self.exp_pheromone_map[min_y:max_y + 1, min_x:max_x + 1]
        return float(np.mean(region))
    
    def get_pheromone_gradient(self, position):
        """Get pheromone gradient for ACO guidance"""
        center_x, center_y = self.world_to_grid(position)
        
        # Calculate gradients using finite differences
        nav_grad_x = 0.0
        nav_grad_y = 0.0
        
        if center_x > 0 and center_x < self.grid_size - 1:
            nav_grad_x = (self.nav_pheromone_map[center_y, center_x + 1] - 
                         self.nav_pheromone_map[center_y, center_x - 1]) / (2 * self.grid_resolution)
        
        if center_y > 0 and center_y < self.grid_size - 1:
            nav_grad_y = (self.nav_pheromone_map[center_y + 1, center_x] - 
                         self.nav_pheromone_map[center_y - 1, center_x]) / (2 * self.grid_resolution)
        
        return np.array([nav_grad_x, nav_grad_y], dtype=np.float32)
    
    def diffuse_pheromones(self):
        """Apply spatial diffusion to pheromone maps"""
        # Simple diffusion using convolution with averaging kernel
        kernel = np.ones((3, 3)) / 9.0
        
        # Apply diffusion to navigation pheromones
        from scipy import ndimage
        diffused_nav = ndimage.convolve(self.nav_pheromone_map, kernel, mode='constant')
        self.nav_pheromone_map = ((1 - self.diffusion_rate) * self.nav_pheromone_map + 
                                 self.diffusion_rate * diffused_nav)
        
        # Apply diffusion to exploration pheromones  
        diffused_exp = ndimage.convolve(self.exp_pheromone_map, kernel, mode='constant')
        self.exp_pheromone_map = ((1 - self.diffusion_rate) * self.exp_pheromone_map + 
                                 self.diffusion_rate * diffused_exp)
    
    def evaporate(self):
        """Apply evaporation to both pheromone maps"""
        self.nav_pheromone_map *= (1 - self.nav_evaporation_rate)
        self.exp_pheromone_map *= (1 - self.exp_evaporation_rate)
        
        # Ensure non-negative values
        self.nav_pheromone_map = np.maximum(self.nav_pheromone_map, 0.0)
        self.exp_pheromone_map = np.maximum(self.exp_pheromone_map, 0.0)
    
    def reset_pheromones(self):
        """Reset all pheromone maps to zero"""
        self.nav_pheromone_map.fill(0.0)
        self.exp_pheromone_map.fill(0.0)
    
    def get_pheromone_stats(self):
        """Get statistics about pheromone maps for monitoring"""
        return {
            'nav_mean': float(np.mean(self.nav_pheromone_map)),
            'nav_max': float(np.max(self.nav_pheromone_map)),
            'nav_std': float(np.std(self.nav_pheromone_map)),
            'exp_mean': float(np.mean(self.exp_pheromone_map)),
            'exp_max': float(np.max(self.exp_pheromone_map)),
            'exp_std': float(np.std(self.exp_pheromone_map))
        }
    
    def save_pheromone_maps(self, filepath):
        """Save pheromone maps to file"""
        np.savez(filepath, 
                nav_pheromone=self.nav_pheromone_map,
                exp_pheromone=self.exp_pheromone_map)
    
    def load_pheromone_maps(self, filepath):
        """Load pheromone maps from file"""
        data = np.load(filepath)
        self.nav_pheromone_map = data['nav_pheromone']
        self.exp_pheromone_map = data['exp_pheromone']