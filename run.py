import numpy as np
import robosuite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from env.PickMove import PickMove
from DLR.network import Hyperparameters, PPOAgent
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import gym
from collections import deque


class ObservationNormalizer(gym.ObservationWrapper):
    def __init__(self, env, clip_range=None):
        super().__init__(env)
        # Make sure obs_rms shape is correct
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.clip_range = clip_range if clip_range is not None else (-10., 10.)
        # Initialize with a dummy update or by absorbing the first observation
        # Not strictly necessary if RunningMeanStd handles count=0 well, but can add robustness
        self.obs_rms.update(env.observation_space.sample()) # Not ideal, but an option

    def observation(self, observation):
        self.obs_rms.update(observation)
        normalized_obs = (observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized_obs, self.clip_range[0], self.clip_range[1])

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32) # Initialize var to ones to avoid division by zero
        self.count = 0.0

    def update(self, x):
        # Ensure x is a numpy array for consistency, and potentially add batch dimension if needed
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1: # If a single observation, add a batch dimension for calculation
            x = x[np.newaxis, :]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta_mean_old = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta_mean_old * (batch_count / total_count)
        
        # Calculate new M2 (sum of squared differences from the mean)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        
        # Note: If self.count is 0, m_a is 0. If batch_count is 0, m_b is 0.
        # This formula handles new data correctly.
        m2_new = m_a + m_b + np.square(delta_mean_old) * (self.count * batch_count / total_count)
        
        # New variance. Add a small epsilon to prevent division by zero in case total_count is 0 or 1
        # The var should not be 0 unless all observations are identical
        new_var = m2_new / (total_count if total_count > 0 else 1.0) # Avoid division by zero if count is 0
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


def main():
    # Initialize environment with BASIC controller
    robots = "Panda"
    config = load_composite_controller_config(controller="BASIC")
    

    my_obs_keys = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'robot0_gripper_qvel',
    'object-state', # This is usually a combined state for the objects
    ]
    
    # Create environment
    base_env = robosuite.make(
        'PickMove',
        robots,
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=40,
    )

    print(f"DEBUG: 1. Base Robosuite env created. Initial obs spec:")
    print(base_env.observation_spec()) # Print the raw robosuite observation spec

    gym_env = GymWrapper(base_env, keys=my_obs_keys)
    print(f"DEBUG: 2. GymWrapper applied. gym_wrapped_env.observation_space.shape: {gym_env.observation_space.shape}")

    env = ObservationNormalizer(gym_env)
    print(f"DEBUG: 3. ObservationNormalizer applied. env.observation_space.shape: {env.observation_space.shape}")

    current_obs = env.reset()

    # Initialize agent with proper dimensions
    params = Hyperparameters()
    agent = PPOAgent(params.obs_dim, params.action_dim)

    print(f"DEBUG: Agent expected obs_dim: {params.obs_dim}")

    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Training statistics
    stats = defaultdict(list)
    best_reward = -float('inf')
    start_time = time.time()

    for episode in range(params.max_episodes):
        episode_reward = 0
        done = False
        step_count = 0
        print(f"DEBUG: Before reset in episode {episode}. Type of 'env' object: {type(env)}") # NEW PRINT
        current_obs, _ = env.reset()
        print(f"DEBUG: Episode {episode}, after env.reset(): type={type(current_obs)}, shape={current_obs.shape}, content={current_obs}")
        # In the training loop:
        while not done:
            action = None
            try:
                action, log_prob, value = agent.select_action(current_obs)
                print(f"DEBUG: Episode {episode}, Step {step_count}, after select_action. Action: {action.shape}, Value: {value:.2f}")
                
                # Direct mapping - agent learns proper scaling
                env_action = np.array(action, dtype=np.float32)
                
                # Step the environment
                next_obs, reward, terminated, truncated, _ = env.step(env_action)
                done = terminated or truncated
                print(f"DEBUG: Episode {episode}, Step {step_count}, after env.step(): type={type(next_obs)}, shape={next_obs.shape}, reward={reward:.2f}, done={done}")

                
                # Store experience with proper type conversion
                agent.store_transition(
                    current_obs,
                    action.astype(np.float32),
                    float(log_prob),
                    float(value),
                    float(reward),
                    bool(done)
                )
                
                current_obs = next_obs
                episode_reward += reward
                step_count += 1
                
                # Train if enough samples
                if len(agent.memory) >= params.buffer_size:
                    loss = agent.train()
                    print(loss)
                    stats['losses'].append(loss)
                
                # Render if needed
                env.render()

            except Exception as e:
                print(f"Error in episode {episode}, step {step_count}:")
                print(f"Action: {action}")
                print(f"Observation keys: {current_obs.keys() if isinstance(current_obs, dict) else 'N/A'}")
                print(f"Error: {str(e)}")
                raise

        # Update statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(step_count)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model('models/best_model.pth')
        
        avg_reward = np.mean(stats['episode_rewards'][-10:])
        print(f"Episode: {episode:4d}, "
                f"Reward: {episode_reward:7.2f}, "
                f"Avg Reward (10): {avg_reward:7.2f}, "
                f"Best: {best_reward:7.2f}, "
                f"Steps: {step_count:4d}, "
                f"Time: {time.time()-start_time:.1f}s")
        
        # Periodic saves
        if episode % 100 == 0:
            agent.save_model(f'models/checkpoint_{episode}.pth')
    
    # Save final model
    agent.save_model('models/final_model.pth')

    plt.figure(figsize=(8,4))
    plt.plot(agent.value_trace,  label='Value (pred)')
    plt.plot(agent.return_trace, label='Return (target)')
    plt.title('Critic Prediction vs. Empirical Return')
    plt.xlabel('Minibatch'); plt.ylabel('Average')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('value_vs_return.png')   # also saved to disk
    plt.show()

    if stats['episode_rewards']:
        plt.figure(figsize=(10, 5))
        plt.plot(stats['episode_rewards'], label='Training Rewards')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.title('PPO Training Reward over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reward_curve.png')  # Save to file
        plt.show() 

    if stats['losses']:
        plt.figure(figsize=(10, 5))
        plt.plot(stats['losses'], label='Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('PPO Training Loss over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss_curve.png')  # Save to file
        plt.show() 

    env.close()

if __name__ == '__main__':
    main()