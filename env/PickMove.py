from collections import OrderedDict
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, ObjectPositionSampler
from robosuite.utils.transform_utils import convert_quat

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
        control_freq=20,
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

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

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

    def reward(self, action=None):
        reward = 0.0
        
        # Get current positions
        gripper_pos = self._observables['robot0_eef_pos'].obs
        cube_pos = self._observables['cube_pos'].obs
        
        # Distance between gripper and cube
        dist = np.linalg.norm(gripper_pos - cube_pos)
        
        # Shaped reward components
        reaching_reward = 1 - np.tanh(10.0 * dist)  # Reward for getting closer
        
        # Grasping reward
        if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
            reaching_reward += 0.25
        
        # Lifting reward
        if self._check_success():
            reaching_reward += 1.0
        
        reward = reaching_reward * self.reward_scale
        
        return reward
    
    def _check_success(self):

        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offsets[0][2]

        # cube is higher than the table top above a margin
        return cube_height > table_height + 0.04

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
    def _setup_references(self):

        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

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
    
    def _reset_internal(self):
        super()._reset_internal()

        # Reset cube position manually
        if not self.deterministic_reset:
            # Fixed position
            cube_pos = np.array([self.table_offset1[0], self.table_offset1[1], self.table_offset1[2] + 0.01])
            cube_quat = np.array([1, 0, 0, 0])
            
            # Set the cube's joint position directly
            self.sim.data.set_joint_qpos(
                self.cube.joints[0],
                np.concatenate([cube_pos, cube_quat])
        )
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

    def visualize(self, vis_settings):

        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)