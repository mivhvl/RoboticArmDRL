import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from collections import deque
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
        self.ent_coef = 0.005
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.hidden_size = 128
        self.buffer_size = 2048
        self.max_episodes = 10000

class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.shared_fc1 = nn.Linear(obs_dim, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Policy head
        self.policy_mean = nn.Linear(hidden_size, action_dim)
        self.policy_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head
        self.value_out = nn.Linear(hidden_size, 1)
        
        self.activation = nn.Tanh()

    def forward(self, x):
        # Shared features
        x = self.activation(self.shared_fc1(x))
        x = self.activation(self.shared_fc2(x))
        
        # Policy
        mean = self.policy_mean(x)
        std = torch.exp(self.policy_std)
        
        # Value
        value = self.value_out(x)
        
        return mean, std, value

class PPOAgent:
    def __init__(self, params):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = PPONetwork(params.obs_dim, params.action_dim, params.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=params.lr)
        self.memory = deque(maxlen=params.buffer_size)

    def preprocess_obs(self, raw_obs):
        obs = np.concatenate([
            np.array(raw_obs['robot0_eef_pos'], dtype=np.float32),
            np.array(raw_obs['robot0_eef_quat'], dtype=np.float32),
            np.array(raw_obs['robot0_gripper_qpos'], dtype=np.float32),
            np.array(raw_obs['robot0_gripper_qvel'], dtype=np.float32),
            np.array(raw_obs['object-state'], dtype=np.float32),
        ]).astype(np.float32)

        norm = np.linalg.norm(obs)
        return obs / (norm + 1e-8)

    def get_action(self, raw_obs):
        obs = self.preprocess_obs(raw_obs)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, std, value = self.network(obs_tensor)
        
        # Scale position/orientation outputs differently
        dist = MultivariateNormal(mean, torch.diag(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Automatic scaling
        action = action.squeeze(0).cpu().numpy()
        action[:6] *= 0.1  # Smaller steps for position/orientation
        action[-1] = np.clip(action[-1], -1, 1) # Clip -1 to 1

    
        return action, log_prob.item(), value.item()
    
    def remember(self, obs, action, log_prob, value, reward, done):
        self.memory.append((obs, action, log_prob, value, reward, done))

    def compute_returns(self, rewards, values, dones):
        returns = []
        last_value = 0
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                last_value = 0
            last_value = reward + self.params.gamma * last_value
            returns.insert(0, last_value)
        return torch.FloatTensor(returns).to(self.device)
    
    def train(self):
        if len(self.memory) < self.params.batch_size:
            print("[DEBUG] Skipping training — not enough samples in memory.")
            return 0.0

        # Prepare training data
        obs, actions, old_log_probs, old_values, rewards, dones = zip(*self.memory)

        # Convert to numpy arrays
        obs_array = np.array([self.preprocess_obs(o) for o in obs])
        actions_array = np.array(actions, dtype=np.float32)
        old_log_probs_array = np.array(old_log_probs, dtype=np.float32)
        old_values_array = np.array(old_values, dtype=np.float32)
        rewards_array = np.array(rewards, dtype=np.float32)
        dones_array = np.array(dones, dtype=np.float32)

        print(f"[DEBUG] Observation shape: {obs_array.shape}")
        print(f"[DEBUG] Action shape: {actions_array.shape}")
        print(f"[DEBUG] Reward stats - mean: {rewards_array.mean():.3f}, std: {rewards_array.std():.3f}, min: {rewards_array.min():.3f}, max: {rewards_array.max():.3f}")

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        actions_tensor = torch.FloatTensor(actions_array).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs_array).to(self.device)
        old_values_tensor = torch.FloatTensor(old_values_array).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards_array).to(self.device)
        dones_tensor = torch.FloatTensor(dones_array).to(self.device)

        # Compute returns and advantages
        returns = self.compute_returns(rewards_tensor, old_values_tensor, dones_tensor)
        advantages = returns - old_values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        print(f"[DEBUG] Advantages - mean: {advantages.mean().item():.3f}, std: {advantages.std().item():.3f}")
        print(f"[DEBUG] Returns - mean: {returns.mean().item():.3f}, std: {returns.std().item():.3f}")

        total_loss = 0.0
        minibatch_count = 0

        for epoch in range(self.params.n_epochs):
            print(f"[DEBUG] PPO epoch {epoch+1}/{self.params.n_epochs}")
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)

            for start in range(0, len(self.memory), self.params.batch_size):
                end = start + self.params.batch_size
                batch_indices = indices[start:end].tolist()

                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                mean, std, values = self.network(batch_obs)
                dist = MultivariateNormal(mean, torch.diag(std))
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.params.clip, 1 + self.params.clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()
                print(f"[DEBUG] Returns={batch_returns.mean().item():.3f}, Values={values.mean().item():.3f}, Advantages={batch_advantages.mean().item():.3f}")

                loss = (
                    policy_loss
                    + self.params.vf_coef * value_loss
                    - self.params.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.params.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                minibatch_count += 1

                # Print per mini-batch loss breakdown
                print(f"[DEBUG][Minibatch {minibatch_count}] Loss: {loss.item():.3f}, "
                    f"Policy Loss: {policy_loss.item():.3f}, "
                    f"Value Loss: {value_loss.item():.3f}, "
                    f"Entropy: {entropy.item():.3f}")

        self.memory.clear()
        avg_loss = total_loss / max(minibatch_count, 1)
        print(f"[DEBUG] PPO Training Complete. Avg Loss: {avg_loss:.3f}")
        return avg_loss
  # Prevent division by 0

    
    def save_model(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': vars(self.params)
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])