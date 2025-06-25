import numpy as np
import robosuite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from env.PickMove import PickMove
from DLR.network import Hyperparameters, PPOAgent
import os
import time
import torch

def main():
    # Initialize environment with BASIC controller
    robots = "Panda"
    config = load_composite_controller_config(controller="BASIC")

    # (Optional) Reduce movement size as in training
    for part in config["body_parts"]:
        if "output_max" not in config["body_parts"][part] or "output_min" not in config["body_parts"][part]:
            continue
        if type(config["body_parts"][part]["output_max"]) is not list:
            config["body_parts"][part]["output_max"] /= 10
            config["body_parts"][part]["output_min"] /= 10
            continue
        new_min = []
        new_max = []
        for val in config["body_parts"][part]["output_max"]:
            new_max.append(val / 10)
        for val in config["body_parts"][part]["output_min"]:
            new_min.append(val / 10)
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
    model_path = 'run_com_so_close/consistent.pth'  # Change to your model path if needed
    agent.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Export model to ONNX
    agent.network.eval()  # Ensure model is in eval mode

    # Create dummy input with correct observation shape
    dummy_input = torch.randn(1, params.obs_dim)  # Batch size of 1

    onnx_export_path = "ppo_agent.onnx"
    torch.onnx.export(
        agent.network,             # Your model
        dummy_input,               # Dummy input
        onnx_export_path,          # Output path
        export_params=True,        # Store trained weights
        opset_version=11,          # ONNX version
        do_constant_folding=True,  # Run constant folding optimizations
        input_names=['input'],     # Input tensor name
        output_names=['mean', 'std', 'value'],  # Your model returns 3 outputs
        dynamic_axes={
            'input': {0: 'batch_size'},
            'mean': {0: 'batch_size'},
            'std': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    print(f"Exported model to {onnx_export_path}")

if __name__ == "__main__":
    main()
