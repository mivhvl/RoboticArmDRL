import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from collections import deque
import torch.nn.functional as F
import os

class Hyperparameters:
    def __init__(self):
        # Observation dimensions 
        self.obs_dim = 21  
        # Action dimensions 6 DoF arm + 1 DoF gripper
        self.action_dim = 7 
        # Training parameters
        self.gamma = 0.99
        self.lr = 1e-6
        self.batch_size = 64
        self.n_epochs = 10
        self.clip = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.hidden_size = 64
        self.buffer_size = 2048
        self.max_episodes = 10000

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Policy mean and log std (log std as parameter)
        self.policy_mean = nn.Linear(hidden_size, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

        # Value function head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        mean = self.policy_mean(x)
        std = torch.exp(self.policy_log_std)
        value = self.value_head(x).squeeze(-1)  # flatten value to shape (batch,)

        return mean, std, value
    

import numpy as np

class PPOAgent:
    def __init__(self, obs_dim, action_dim, device='cpu', **kwargs):
        self.device = device
        self.network = PPONetwork(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)

        # Hyperparameters
        self.clip = kwargs.get('clip', 0.2)
        self.gamma = kwargs.get('gamma', 0.99)
        self.lam = kwargs.get('lam', 0.95)  # for GAE
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.batch_size = kwargs.get('batch_size', 64)
        self.n_epochs = kwargs.get('n_epochs', 10)

        # Experience buffers
        self.memory = []

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        mean, std, value = self.network(obs_tensor)
        dist = MultivariateNormal(mean, torch.diag(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.item(), value.item()

    def store_transition(self, obs, action, log_prob, value, reward, done):
        self.memory.append((obs, action, log_prob, value, reward, done))

    def compute_gae(self, rewards, values, dones):
        """Compute GAE advantages and returns."""
        advantages = []
        gae = 0
        values = values + [0]  # add dummy for last value
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def preprocess_batch(self):
        obs, actions, log_probs, values, rewards, dones = zip(*self.memory)
        advantages, returns = self.compute_gae(list(rewards), list(values), list(dones))

        obs = torch.FloatTensor(np.array([self.preprocess_obs(o) for o in obs])).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        return obs, actions, old_log_probs, returns, advantages

    def preprocess_obs(self, raw_obs):
        # You can put your preprocessing here or outside and just feed ready obs
        # For now, assume raw_obs is already a numpy vector of obs_dim size
        return raw_obs

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        obs, actions, old_log_probs, returns, advantages = self.preprocess_batch()

        total_loss = 0
        for _ in range(self.n_epochs):
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)

            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                mean, std, values = self.network(batch_obs)
                dist = MultivariateNormal(mean, torch.diag(std))
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        self.memory = []  # Clear buffer after training
        return total_loss / (self.n_epochs * (len(obs) // self.batch_size + 1))
    
    def save_model(self, path):
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'clip': self.clip,
                'gamma': self.gamma,
                'lam': self.lam,
                'vf_coef': self.vf_coef,
                'ent_coef': self.ent_coef,
                'max_grad_norm': self.max_grad_norm,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'obs_dim': self.network.fc1.in_features,
                'action_dim': self.network.policy_mean.out_features
            }
        }
        torch.save(checkpoint, path)
        print(f"[INFO] Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        hyp = checkpoint.get('hyperparameters', {})
        self.clip = hyp.get('clip', self.clip)
        self.gamma = hyp.get('gamma', self.gamma)
        self.lam = hyp.get('lam', self.lam)
        self.vf_coef = hyp.get('vf_coef', self.vf_coef)
        self.ent_coef = hyp.get('ent_coef', self.ent_coef)
        self.max_grad_norm = hyp.get('max_grad_norm', self.max_grad_norm)
        self.batch_size = hyp.get('batch_size', self.batch_size)
        self.n_epochs = hyp.get('n_epochs', self.n_epochs)
        print(f"[INFO] Model loaded from {path}")


