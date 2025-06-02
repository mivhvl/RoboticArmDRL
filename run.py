import numpy as np
import robosuite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from env.PickMove import PickMove
from DLR.network import Hyperparameters, PPOAgent
import os
import time
from collections import defaultdict

def main():
    # Initialize environment with BASIC controller
    robots = "Panda"
    config = load_composite_controller_config(controller="BASIC")
    
    # Create environment
    env = robosuite.make(
        'PickMove',
        robots,
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    # Initialize agent with proper dimensions
    params = Hyperparameters()
    agent = PPOAgent(params)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Training statistics
    stats = defaultdict(list)
    best_reward = -float('inf')
    start_time = time.time()

    for episode in range(params.max_episodes):
        raw_obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        # In the training loop:
        while not done:
            action = None
            try:
                action, log_prob, value = agent.get_action(raw_obs)
                
                # Direct mapping - agent learns proper scaling
                env_action = np.array(action, dtype=np.float32)
                
                # Step the environment
                next_raw_obs, reward, done, _ = env.step(env_action)
                
                # Store experience with proper type conversion
                agent.remember(
                    raw_obs,
                    action.astype(np.float32),
                    float(log_prob),
                    float(value),
                    float(reward),
                    bool(done)
                )
                
                raw_obs = next_raw_obs
                episode_reward += reward
                step_count += 1
                
                # Train if enough samples
                if len(agent.memory) >= params.buffer_size:
                    agent.train()
                
                # Render if needed
                env.render()

            except Exception as e:
                print(f"Error in episode {episode}, step {step_count}:")
                print(f"Action: {action}")
                print(f"Observation keys: {raw_obs.keys() if isinstance(raw_obs, dict) else 'N/A'}")
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
    env.close()

if __name__ == '__main__':
    main()