# network.py - Unified Actor-Critic Network Architecture
import torch
import torch.nn as nn
import numpy as np
import config


class ActorCriticNetwork(nn.Module):
    """Unified Actor-Critic network supporting both MLP and LSTM architectures"""
    
    def __init__(self, obs_dim, action_dim, architecture='MLP'):
        super(ActorCriticNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.architecture = architecture
        
        if architecture == 'MLP':
            self._build_mlp_network()
        elif architecture == 'LSTM':
            self._build_lstm_network()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        print(f"🧠 ActorCriticNetwork ({architecture}):")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Action dim: {action_dim}")
        if architecture == 'LSTM':
            print(f"   LSTM hidden size: {config.LSTM_HIDDEN_SIZE}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_mlp_network(self):
        """Build MLP-based network"""
        hidden_sizes = config.MLP_HIDDEN_SIZES
        
        # Shared feature extraction layers
        layers = []
        prev_size = self.obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy network)
        self.actor_mean = nn.Linear(prev_size, self.action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        # Critic head (value network)
        self.critic = nn.Linear(prev_size, 1)
        
        # LSTM-specific components (not used in MLP)
        self.lstm = None
        self.lstm_hidden_size = 0
        
    def _build_lstm_network(self):
        """Build LSTM-based network"""
        self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        self.lstm_num_layers = config.LSTM_NUM_LAYERS
        
        # Input processing layer
        self.input_layer = nn.Linear(self.obs_dim, self.lstm_hidden_size)
        
        # LSTM core
        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )
        
        # Actor head (policy network)
        self.actor_mean = nn.Linear(self.lstm_hidden_size, self.action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        # Critic head (value network)
        self.critic = nn.Linear(self.lstm_hidden_size, 1)
        
        # MLP-specific components (not used in LSTM)
        self.shared_layers = None
        
    def forward(self, obs, hidden_state=None):
        """
        Forward pass through the network
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or [batch_size, seq_len, obs_dim]
            hidden_state: LSTM hidden state (only used for LSTM architecture)
            
        Returns:
            action_mean: Mean of action distribution
            action_log_std: Log standard deviation of action distribution  
            value: State value estimate
            new_hidden_state: Updated LSTM hidden state (None for MLP)
        """
        if self.architecture == 'MLP':
            return self._forward_mlp(obs)
        elif self.architecture == 'LSTM':
            return self._forward_lstm(obs, hidden_state)
    
    def _forward_mlp(self, obs):
        """Forward pass for MLP architecture"""
        features = self.shared_layers(obs)
        
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        value = self.critic(features)
        
        return action_mean, action_log_std, value, None
    
    def _forward_lstm(self, obs, hidden_state):
        """Forward pass for LSTM architecture"""
        # Process input
        x = torch.relu(self.input_layer(obs))
        
        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # LSTM forward pass
        if hidden_state is not None:
            lstm_out, new_hidden_state = self.lstm(x, hidden_state)
        else:
            lstm_out, new_hidden_state = self.lstm(x)
        
        # Use last timestep output
        features = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        value = self.critic(features)
        
        return action_mean, action_log_std, value, new_hidden_state
    
    def get_action(self, obs, hidden_state=None, deterministic=False):
        """
        Sample action from policy
        
        Args:
            obs: Observation tensor
            hidden_state: LSTM hidden state (if applicable)
            deterministic: Whether to sample deterministically
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
            new_hidden_state: Updated hidden state
        """
        action_mean, action_log_std, value, new_hidden_state = self.forward(obs, hidden_state)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            action_std = torch.exp(action_log_std)
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value.squeeze(-1), new_hidden_state
    
    def evaluate_actions(self, obs, actions, hidden_state=None):
        """
        Evaluate actions for PPO training
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            actions: Action tensor [batch_size, action_dim]
            hidden_state: LSTM hidden state (if applicable)
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        action_mean, action_log_std, values, _ = self.forward(obs, hidden_state)
        
        action_std = torch.exp(action_log_std)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy
    
    def init_hidden_state(self, batch_size, device):
        """Initialize LSTM hidden state"""
        if self.architecture != 'LSTM':
            return None
        
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        return (h0, c0)
    
    def save_model(self, filepath):
        """Save model parameters"""
        torch.save({
            'state_dict': self.state_dict(),
            'architecture': self.architecture,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }, filepath)
    
    def load_model(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint