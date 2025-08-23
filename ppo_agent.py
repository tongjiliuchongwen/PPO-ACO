# ppo_agent.py - PPO Agent with LSTM Support and GPU Optimizations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import config
from network import ActorCriticNetwork


class PPOAgent:
    """PPO Agent with support for both MLP and LSTM architectures"""
    
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        
        # Environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Create network based on configuration
        self.network = ActorCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            architecture=config.AGENT_ARCHITECTURE
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)
        
        # PPO hyperparameters
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.clip_ratio = config.CLIP
        self.n_updates_per_iteration = config.N_UPDATES_PER_ITERATION
        self.timesteps_per_batch = config.TIMESTEPS_PER_BATCH
        
        # GPU optimization: Mixed precision training
        self.use_mixed_precision = config.USE_MIXED_PRECISION and device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("🚀 Mixed precision training enabled")
        
        # Statistics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
        print(f"🤖 PPO Agent initialized:")
        print(f"   Architecture: {config.AGENT_ARCHITECTURE}")
        print(f"   Device: {device}")
        print(f"   Batch size: {self.timesteps_per_batch}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
    
    def rollout(self, max_timesteps=None):
        """
        Collect rollout data for training
        
        Returns:
            batch_data: Dictionary containing rollout data
        """
        if max_timesteps is None:
            max_timesteps = self.timesteps_per_batch
        
        # Storage for rollout data
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_log_probs = []
        batch_values = []
        batch_dones = []
        
        # For LSTM: track hidden states
        if config.AGENT_ARCHITECTURE == 'LSTM':
            batch_hidden_states = []
        
        # Episode-wise data for GAE computation
        episode_rewards = []
        episode_values = []
        episode_dones = []
        
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        timesteps_collected = 0
        
        # Initialize LSTM hidden state if needed
        if config.AGENT_ARCHITECTURE == 'LSTM':
            hidden_state = self.network.init_hidden_state(1, self.device)
        else:
            hidden_state = None
        
        while timesteps_collected < max_timesteps:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value, new_hidden_state = self.network.get_action(
                    obs_tensor, hidden_state
                )
            
            # Store data
            batch_obs.append(obs)
            batch_actions.append(action.cpu().numpy().flatten())
            batch_log_probs.append(log_prob.cpu().numpy().item() if log_prob is not None else 0.0)
            batch_values.append(value.cpu().numpy().item())
            
            if config.AGENT_ARCHITECTURE == 'LSTM':
                batch_hidden_states.append(hidden_state)
                hidden_state = new_hidden_state
            
            # Episode tracking
            episode_rewards.append(0.0)  # Will be filled after step
            episode_values.append(value.cpu().numpy().item())
            episode_dones.append(done)
            
            # Take environment step
            action_np = action.cpu().numpy().flatten()
            next_obs, reward, done, truncated, info = self.env.step(action_np)
            
            # Update reward in storage
            episode_rewards[-1] = reward
            batch_rewards.append(reward)
            batch_dones.append(done or truncated)
            
            episode_reward += reward
            episode_length += 1
            timesteps_collected += 1
            
            # Handle episode termination
            if done or truncated:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Reset environment and hidden state
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                
                if config.AGENT_ARCHITECTURE == 'LSTM':
                    hidden_state = self.network.init_hidden_state(1, self.device)
            else:
                obs = next_obs
        
        # Convert lists to numpy arrays
        batch_data = {
            'observations': np.array(batch_obs, dtype=np.float32),
            'actions': np.array(batch_actions, dtype=np.float32),
            'rewards': np.array(batch_rewards, dtype=np.float32),
            'log_probs': np.array(batch_log_probs, dtype=np.float32),
            'values': np.array(batch_values, dtype=np.float32),
            'dones': np.array(batch_dones, dtype=bool)
        }
        
        if config.AGENT_ARCHITECTURE == 'LSTM':
            batch_data['hidden_states'] = batch_hidden_states
        
        return batch_data
    
    def compute_gae(self, rewards, values, dones, next_value=0.0):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_networks(self, batch_data):
        """Update policy and value networks using PPO"""
        # Convert data to tensors
        obs = torch.FloatTensor(batch_data['observations']).to(self.device)
        actions = torch.FloatTensor(batch_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch_data['log_probs']).to(self.device)
        
        # Compute advantages and returns
        values = batch_data['values']
        rewards = batch_data['rewards']
        dones = batch_data['dones']
        
        # Get final value for GAE computation
        with torch.no_grad():
            if len(obs) > 0:
                final_obs = obs[-1:] 
                _, _, final_value, _ = self.network.forward(final_obs)
                final_value = final_value.cpu().numpy().item()
            else:
                final_value = 0.0
        
        advantages, returns = self.compute_gae(rewards, values, dones, final_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # PPO updates
        for _ in range(self.n_updates_per_iteration):
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs, values, entropy = self.network.evaluate_actions(obs, actions)
                    
                    # PPO loss computation
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(values, returns)
                    entropy_loss = -entropy.mean()
                    
                    total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_probs, values, entropy = self.network.evaluate_actions(obs, actions)
                
                # PPO loss computation
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns)
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
    
    def get_statistics(self):
        """Get training statistics"""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards[-100:]),
            'mean_episode_length': np.mean(self.episode_lengths[-100:]),
            'total_episodes': len(self.episode_rewards),
            'success_rate': sum(1 for r in self.episode_rewards[-100:] if r > 50) / min(len(self.episode_rewards), 100)
        }
    
    def save_model(self, filepath):
        """Save model and optimizer state"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
    
    def load_model(self, filepath):
        """Load model and optimizer state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])