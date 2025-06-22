import numpy as np
import robosuite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from env.PickMove import PickMove
from DLR.network import Hyperparameters, PPOAgent
import os
import time

def main():
    # Initialize environment with BASIC controller
    robots = "Panda"
    config = load_composite_controller_config(controller="BASIC")

    # (Optional) Reduce movement size as in training
    for part in config["body_parts"]:
        if "output_max" not in config["body_parts"][part] or "output_min" not in config["body_parts"][part]:
            continue
        if type(config["body_parts"][part]["output_max"]) is not list:
            config["body_parts"][part]["output_max"] /= 5
            config["body_parts"][part]["output_min"] /= 5
            continue
        new_min = []
        new_max = []
        for val in config["body_parts"][part]["output_max"]:
            new_max.append(val / 5) 
        for val in config["body_parts"][part]["output_min"]:
            new_min.append(val / 5)
        config["body_parts"][part]["output_min"] = new_min
        config["body_parts"][part]["output_max"] = new_max

    # Create environment
    env = robosuite.make(
        'PickMove',
        robots,
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        ignore_done=False, 
    )

    # Initialize agent and load model
    params = Hyperparameters()
    agent = PPOAgent(params.obs_dim, params.action_dim, kwargs=params)
    model_path = 'models/best_model.pth'  # Change to your model path if needed
    agent.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Run a few episodes to sample actions
    num_episodes = 5
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        while not done:
            action, log_prob, value = agent.select_action(obs)
            env_action = np.array(action, dtype=np.float32)
            next_obs, reward, done, _ = env.step(env_action)
            obs = next_obs
            episode_reward += reward
            step_count += 1
            env.render()
            time.sleep(0.05)  # Slow down for visualization
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {step_count}")

    env.close()

if __name__ == '__main__':
    main()