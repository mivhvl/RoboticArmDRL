from collections import OrderedDict
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, ObjectPositionSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils import transform_utils as T

# Environment Setup
class PickMove(ManipulationEnv):

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.5, 0.2, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=10,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset1 = np.array((0, 0.12, 0.9))
        self.table_offset2 = np.array((0, -0.12, 0.9))
        self.gripper_grip_site_id = None

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.cube_disturbance_threshold = 0.05
        self.initial_cube_pos = None
        self.place_target_pos = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _check_cube_fallen(self):
        """
        Checks if the cube has fallen off the table (e.g., gone below a certain Z-threshold).
        """
        cube_pos = self._observables['cube_pos'].obs
        # Check if the cube's z-position is significantly below the table surface
        # Or below a general floor level
        
        # Using the table_offset1[2] you have:
        # return cube_pos[2] < self.table_offset1[2] - 0.05 # Example threshold: 5cm below table surface
        # Make sure self.table_offset1 is defined at this point (it should be in __init__)
        
        # More robust: check against a defined floor height or threshold relative to table
        return cube_pos[2] < self.table_offset1[2] #- 0.05


    def _check_placed(self):
        """
        Checks if the cube is placed on the target table and released.
        """
        # Cube position relative to target table surface
        cube_pos = self._observables['cube_pos'].obs
        
        # Check if cube is within placing region (x, y horizontal tolerance)
        xy_tolerance = 0.08 # 3cm radius around target center
        z_tolerance = 0.02 # 2cm above table surface for 'on table'
        
        # Cube is considered "placed" if its XY position is near the target AND its Z is near the table surface
        # AND it is NOT grasped by the robot
        
        is_near_target_xy = np.linalg.norm(cube_pos[:2] - self.place_target_pos[:2]) < xy_tolerance
        is_on_table_z = (cube_pos[2] > self.table_offset2[2] - 0.01) and (cube_pos[2] < self.table_offset2[2] + z_tolerance)
        
        # Additionally, ensure the robot is not grasping the object
        is_not_grasped = not self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

        return is_near_target_xy and is_on_table_z and is_not_grasped

    def reward(self, action=None):
        reward = 0.0
        global_z_axis = np.array([0., 0., -1.])

        gripper_pos = self._observables['robot0_eef_pos'].obs
        cube_pos = self._observables['cube_pos'].obs

        cube_to_target_reward = 0.0

        dist = np.linalg.norm(gripper_pos - cube_pos)
        reach_reward = 1 - np.tanh(dist / 0.2)  # tighter shaping
        reach_reward *= reach_reward

        if self._check_cube_fallen():
            return -15  # large negative reward if cube has fallen

        #print("Reaching reward:", reach_reward)
        grasp_reward = 0.0
        gripper_open_reward = 0.0
        cube_disturbance_penalty = 0.0
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            cube_to_target = np.linalg.norm((cube_pos - self.place_target_pos) * np.array([1, 1, 0]))
            cube_to_target_reward = 1 - np.tanh(cube_to_target / 0.2)  # tighter shaping
            #cube_to_target_reward *= cube_to_target_reward  # square the reward for tighter shaping
            if cube_pos[2] > self.table_offset1[2] + 0.05:
                grasp_reward = 2  # lifted
            else:
                grasp_reward = 1.0  # grasped
        else:
            gripper_joint_names = self.robots[0].gripper["right"].joints
            joint_qpos = [self.sim.data.get_joint_qpos(joint) for joint in gripper_joint_names]
            open_val = self.robots[0].gripper["right"].init_qpos[0]
            current_val = joint_qpos[0]
            normalized_open = current_val / open_val
            gripper_open_reward = 0.1 * normalized_open

            if reach_reward > 0.8:
                gripper_open_reward = (1 - normalized_open) * 0.3  # encourage closing when close to cube

            if self.initial_cube_pos is not None:
                current_cube_movement = np.linalg.norm(cube_pos - self.initial_cube_pos)
                if current_cube_movement > self.cube_disturbance_threshold:
                    cube_disturbance_penalty = -0.035

        action_penalty = -0.005 * np.square(action).sum() if action is not None else 0.0

        verticality_penalty = 0.0

        gripper_quat = self._observables['robot0_eef_quat'].obs  # (x, y, z, w)
        gripper_mat = T.quat2mat(gripper_quat)
        gripper_z_axis = gripper_mat[:, 2]  # z-axis of the gripper in world frame

        placed_reward = 0.0
        if self._check_placed():
            placed_reward = 2000.0

        # Compute alignment with global z-axis (vertical)
        verticality = np.dot(gripper_z_axis, global_z_axis) - .98
        if verticality < 0:
            verticality_penalty = -0.075

        #reward = (reach_reward * 1.5 + 3 * grasp_reward + action_penalty + verticality_penalty + gripper_open_reward) * self.reward_scale
        reward = reach_reward + 2 * grasp_reward + action_penalty + gripper_open_reward + verticality_penalty + placed_reward + cube_to_target_reward * 5

        return reward
    
    def _check_success(self):
        return self._check_placed()
        
        #return self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = MultiTableArena(
            table_offsets=[self.table_offset1, self.table_offset2],
            table_rots=[0, 0],
            table_full_sizes=[self.table_full_size, self.table_full_size],
            table_frictions=[self.table_friction, self.table_friction],
        )

        mujoco_arena.set_origin([0, 0, 0])

        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
        )

        self.placement_initializer = ObjectPositionSampler(
            name="FixedCubeSampler",
            mujoco_objects=self.cube,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset1,
            z_offset=0.01
        )


        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube],
        )

        self.place_target_pos = np.array([
            self.table_offset2[0],
            self.table_offset2[1],
            self.table_offset2[2] + self.cube.size[0] / 2.0 + 0.01 # Cube center z above table + small offset
        ])

    def _setup_references(self):

        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

        try:
            self.gripper_grip_site_id = self.sim.model.site_name2id("gripper0_right_grip_site")
            print(f"INFO: 'gripper0_right_grip_site' ID found: {self.gripper_grip_site_id}")
        except KeyError:
            self.gripper_grip_site_id = None
            print("ERROR: 'gripper0_right_grip_site' not found. Verticality reward will be 0.")

    def _setup_observables(self):

        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            #@sensor(modality=modality)
            #def robot0_gripper_qpos(obs_cache):
            #    return np.array([self.sim.data.qpos[x] for x in self.robots[0].gripper.dof_ids])

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables
    
    def _post_load(self):
        super()._post_load()
        # Set cube_body_id here to ensure the model is fully loaded
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        print(f"DEBUG in _post_load: cube_body_id set to {self.cube_body_id}")

        # MODIFIED: Set the precise target position for placing the cube on the second table
        # It's usually slightly above the table surface
        # Assuming table_offset2 is the center of the second table
        self.place_target_pos = np.array([
            self.table_offset2[0],
            self.table_offset2[1],
            self.table_offset2[2] + self.cube.size[0] / 2.0 + 0.01 # Cube center z above table + small offset
        ])
        print(f"DEBUG: Place target position set to: {self.place_target_pos}")

    def _post_reset(self):
        super()._post_reset()
        # Ensure the cube_body_id is available before attempting to get its position
        if hasattr(self, 'cube_body_id'):
            self.initial_cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
            print(f"DEBUG: Initial cube position set in _post_reset: {self.initial_cube_pos}")
        else:
            print("WARNING: cube_body_id not found during _post_reset. initial_cube_pos not set. This implies _post_load might have failed or not run.")

    def _reset_internal(self):
        super()._reset_internal()

        # Reset cube position manually
        if not self.deterministic_reset:
            # Fixed position
            cube_pos = np.array([self.table_offset1[0], self.table_offset1[1], self.table_offset1[2] + 0.01])
            cube_quat = np.array([1, 0, 0, 0])

            self.initial_cube_pos = cube_pos.copy()  # Store the initial position for later checks
            
            # Set the cube's joint position directly
            self.sim.data.set_joint_qpos(
                self.cube.joints[0],
                np.concatenate([cube_pos, cube_quat])
        )
            
    def _get_done(self):
        """
        Overriding the base method to include the cube fallen condition.
        """
        # Default done conditions (horizon reached, optional success condition if `ignore_done` is False)
        done = False
        
        # Add the cube fallen condition
        if self._check_cube_fallen():
            done = True

        if self._check_placed():
            done = True
        
        return done
    
    # FOR RANDOM CUBE PLACEMENT
    # def _reset_internal(self):

    #     super()._reset_internal()

    #     # Reset all object positions using initializer sampler if we're not directly loading from an xml
    #     if not self.deterministic_reset:

    #         # Sample from the placement initializer for all objects
    #         object_placements = self.placement_initializer.sample()

    #         # Loop through all objects and reset their positions
    #         for obj_pos, obj_quat, obj in object_placements.values():
    #             self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def step(self, action):
        """
        Overrides the step function to explicitly handle done conditions,
        including the cube falling.
        """
        # Call the base class's step method first to advance the simulation
        # and get the default next_observation, reward, done, and info.
        next_observation, reward, done, info = super().step(action)

        # Now, explicitly check our custom done condition
        # (This will call your _get_done method, which includes _check_cube_fallen)
        custom_done = self._get_done()

        # Combine the done signals:
        # If the base class says it's done OR our custom condition says it's done,
        # then the episode is truly done.
        final_done = done or custom_done

        info["is_success"] = self._check_success()

        # Return the updated done signal
        return next_observation, reward, final_done, info

    def visualize(self, vis_settings):

        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)