import numpy as np
import robosuite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from env import NewLift

if __name__ == '__main__':

    debug = True

    # Initialize environment
    robots = "Panda"
    config = load_composite_controller_config(controller="BASIC")
   
    # Create the robosuite environment
    env = robosuite.make(
        'NewLift',
        robots,
        controller_configs=config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    # Wrap the environment with GymWrapper
    env = GymWrapper(env)

    # Reset the environment
    obs = env.reset()


    for i in range(10):
        action = env.action_space.sample() * 0.1  
        obs, reward, done, info, _ = env.step(action)
        original_obs_dict = env.unwrapped._get_observations()

        # Print all observation data to terminal
        if debug:
            print("Observations:")
            print(f"EEF Position: {original_obs_dict['robot0_eef_pos']}")
            print(f"EEF Quaternion: {original_obs_dict['robot0_eef_quat']}")
            print(f"Gripper Position: {original_obs_dict['robot0_gripper_qpos']}")
            print(f"Gripper Velocity: {original_obs_dict['robot0_gripper_qvel']}")
            print(f"Object State: {original_obs_dict['object-state']}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Info: {info}")

        env.render()

        if done:
            break
