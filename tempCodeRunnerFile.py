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