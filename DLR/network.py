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
        self.lr = 3e-4
        self.batch_size = 64
        self.n_epochs = 15
        self.clip = 0.2
        self.ent_coef = 0.015
        self.vf_coef = 0.4
        self.max_grad_norm = .5
        self.hidden_size = 128
        self.buffer_size = 2048
        self.max_episodes = 1000

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super(PPONetwork, self).__init__()
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
    def __init__(self, obs_dim, action_dim, device='cpu', kwargs=Hyperparameters()):
        self.device = device
        self.network = PPONetwork(obs_dim, action_dim).to(device)

        # Hyperparameters
        self.clip = kwargs.clip
        self.gamma = kwargs.gamma
        self.lam = .95  # for GAE
        self.vf_coef = kwargs.vf_coef
        self.ent_coef = kwargs.ent_coef
        self.max_grad_norm = kwargs.max_grad_norm
        self.batch_size = kwargs.batch_size
        self.n_epochs = kwargs.n_epochs
        self.optimizer = optim.Adam(self.network.parameters(), lr=kwargs.lr)

        # Experience buffers
        self.memory = []

        # Plots
        self.value_trace  = []   # average critic prediction per minibatch
        self.return_trace = []   # average empirical return per minibatch

    def select_action(self, raw_obs):
        obs = self.preprocess_obs(raw_obs)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mean, std, value = self.network(obs_tensor)
        dist = MultivariateNormal(mean, torch.diag(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Automatic scaling
        action = action.squeeze(0).cpu().numpy()
        #action[:6] *= .1  # Smaller steps for position/orientation
        action[-1] = np.clip(action[-1], -1, 1) # Clip -1 to 1

        return action, log_prob.item(), value.item()

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

    # In your PPOAgent class (revert preprocess_obs)
    def preprocess_obs(self, raw_obs):
        return np.concatenate([
            np.array(raw_obs['robot0_eef_pos'], dtype=np.float32),
            np.array(raw_obs['robot0_eef_quat'], dtype=np.float32),
            np.array(raw_obs['robot0_gripper_qpos'], dtype=np.float32),
            np.array(raw_obs['robot0_gripper_qvel'], dtype=np.float32),
            np.array(raw_obs['object-state'], dtype=np.float32),
        ])

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        obs, actions, old_log_probs, returns, advantages = self.preprocess_batch()

        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                avg_value_loss = values.mean().item()
                avg_batch_returns = batch_returns.mean().item()
                self.value_trace.append(avg_value_loss)
                self.return_trace.append(avg_batch_returns)

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


