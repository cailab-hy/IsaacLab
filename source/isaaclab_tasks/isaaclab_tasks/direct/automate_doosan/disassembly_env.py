# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import numpy as np
import os
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation # ,RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import axis_angle_from_quat, euler_xyz_from_quat

from . import automate_algo_utils as automate_algo
from . import factory_control, factory_utils
from .disassembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, DisassemblyEnvCfg

# ---------------------------------------------------------
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from .asset_information import get_plug_info, get_socket_info
# ---------------------------------------------------------


class DisassemblyEnv(DirectRLEnv):
    cfg: DisassemblyEnvCfg

    def __init__(self, cfg: DisassemblyEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.tasks[cfg.task_name]

        super().__init__(cfg, render_mode, **kwargs)

        # ---------------------------------------------------------------------------------------------------------------
        # Get data & mesh directory
        self.data_dir = "source/isaaclab_tasks/isaaclab_tasks/direct/automate_doosan/data"
        self.mesh_dir = f"{ISAACLAB_ASSETS_DATA_DIR}/automate_urdf/mesh"

        # Get the information of plug and socket (e.g., height, diameter, ...)
        self.plug_height, self.plug_base_z_offset, self.plug_diameter, self.grasp_scale, self.grasp_calibration = get_plug_info(self.cfg_task.assembly_id)
        self.socket_height, self.socket_base_height = get_socket_info(self.cfg_task.assembly_id)

        # Get the diameter of plug based on plug object bounding box
        if self.plug_diameter is None:
            self.plug_diameter = automate_algo.get_held_asset_diameter(
                os.getcwd() + f"/source/isaaclab_assets/data/automate_urdf/mesh/{self.cfg_task.assembly_id}/" + 'asset_plug.obj'
            )

        # Get the pos_action_bounds
        self.pos_action_bounds = self.cfg.ctrl.pos_action_bounds
        if self.plug_height > self.pos_action_bounds[0]:
            self.pos_action_bounds = [self.plug_height*3, self.plug_height*3, self.plug_height*3]    
        # ---------------------------------------------------------------------------------------------------------------

        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        # initialized logging variables for disassembly paths
        self._init_log_data_per_assembly()

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset
        held_base_x_offset = 0.0
        held_base_z_offset = self.plug_base_z_offset

        self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.held_base_pos_local[:, 0] = held_base_x_offset
        self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        self.plug_grasps, self.disassembly_dists = self._load_assembly_info()

        # Load grasp pose from json files given assembly ID
        # Grasp pose tensors
        self.palm_to_finger_center = (
            torch.tensor([0.0, 0.0, -self.cfg_task.palm_to_finger_dist], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.robot_to_gripper_quat = (
            torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.plug_grasp_pos_local = self.plug_grasps[: self.num_envs, :3]
        self.plug_grasp_pos_local[:, 0] += self.grasp_calibration[0]
        self.plug_grasp_pos_local[:, 1] += self.grasp_calibration[1]
        self.plug_grasp_pos_local[:, 2] += self.grasp_calibration[2]
        self.plug_grasp_quat_local = torch.roll(self.plug_grasps[: self.num_envs, 3:], -1, 1)
        
        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("left_inner_finger")
        self.right_finger_body_idx = self._robot.body_names.index("right_inner_finger")
        self.fingertip_body_idx = self._robot.body_names.index("tool0")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 6), device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_success_pos_local[:, 2] = self.plug_base_z_offset

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _load_assembly_info(self):
        """Load grasp pose and disassembly distance for plugs in each environment."""
        plug_grasp_path = os.path.join(os.getcwd(), self.data_dir, 'plug_grasps.json')
        with open(plug_grasp_path, "r") as in_file:
            plug_grasp_dict = json.load(in_file)
        plug_grasps = [plug_grasp_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        disassembly_dist_path = os.path.join(os.getcwd(), self.data_dir, 'disassembly_dist.json')
        with open(disassembly_dist_path, "r") as in_file:
            disassembly_dist_dict = json.load(in_file)
        disassembly_dists = [disassembly_dist_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        return torch.as_tensor(plug_grasps).to(self.device), torch.as_tensor(disassembly_dists).to(self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]
        
        # === We perform calibration of the rotation axis. (Edit by CAI-LAB) ===============================================================================
        # The Robotiq 2F-85 gripper is mounted with a 90-degree rotation around the Y-axis relative to the reference coordinate frame of the Panda gripper.

        # Euler Angle (ZYX): [0.0, np.pi/2, 0.0] → Quaternion: [0.7071068, 0.0, 0.7071068, 0.0]
        rotate_z_quat = torch.tensor([0.7071068, 0.0, 0.7071068, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.fingertip_midpoint_quat, self.fingertip_midpoint_pos = torch_utils.tf_combine(
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_pos,
            rotate_z_quat,
            torch.zeros_like(self.fingertip_midpoint_pos),
        )

        tmp_fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        fingertip_midpoint_linvel_x = -tmp_fingertip_midpoint_linvel[:, 2]
        fingertip_midpoint_linvel_y = tmp_fingertip_midpoint_linvel[:, 1]
        fingertip_midpoint_linvel_z = tmp_fingertip_midpoint_linvel[:, 0]
        self.fingertip_midpoint_linvel = torch.stack([fingertip_midpoint_linvel_x, fingertip_midpoint_linvel_y, fingertip_midpoint_linvel_z], dim=1)

        tmp_fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]
        fingertip_midpoint_angvel_x = -tmp_fingertip_midpoint_angvel[:, 2]
        fingertip_midpoint_angvel_y = tmp_fingertip_midpoint_angvel[:, 1]
        fingertip_midpoint_angvel_z = tmp_fingertip_midpoint_angvel[:, 0]
        self.fingertip_midpoint_angvel = torch.stack([fingertip_midpoint_angvel_x, fingertip_midpoint_angvel_y, fingertip_midpoint_angvel_z], dim=1)
        # ==================================================================================================================================================

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:6]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:6]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:6, 0:6]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()
        
        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.gripper_goal_quat,
            self.gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:6] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:6].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            # --- Change in degrees of freedom: 7 (panda) → 6 (doosan) -----
            "joint_pos": self.joint_pos[:, 0:6],
            # --------------------------------------------------------------
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

    def move_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # --- Control code for the Robotiq 2F-85 gripper --------------------------------
        # -- When the gripper width is 85 mm (open), the joint angle is 0 degrees
        # -- When the gripper width is 0 mm (closed), the joint angle is 45 degrees
        # -- For accurate distance-to-angle proportionality, it should be divided by 1.8889, but for a safety margin, it is divided by 'grasp_scale'.
        
        tmp_target_gripper_degree = 45-((self.plug_diameter*1e3)/self.grasp_scale)
        ctrl_target_gripper_dof_pos = np.deg2rad(tmp_target_gripper_degree)
        # -------------------------------------------------------------------------------

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6] * self.rot_threshold

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6] * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.pos_action_bounds[0], self.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals()

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use physx's PD controller
        # --- Control code for the Robotiq 2F-85 gripper ----------------------------------------------------------
        self.ctrl_target_joint_pos[:, 6] = self.ctrl_target_gripper_dof_pos     # finger_joint
        self.ctrl_target_joint_pos[:, 7] = self.ctrl_target_gripper_dof_pos     # right_outer_knuckle_joint
        self.ctrl_target_joint_pos[:, 10] = -self.ctrl_target_gripper_dof_pos   # left_inner_finger_joint
        self.ctrl_target_joint_pos[:, 11] = self.ctrl_target_gripper_dof_pos    # right_inner_finger_joint
        self.ctrl_target_joint_pos[:, 12] = -self.ctrl_target_gripper_dof_pos   # left_inner_finger_knuckle_joint
        self.ctrl_target_joint_pos[:, 13] = -self.ctrl_target_gripper_dof_pos   # right_inner_finger_knuckle_joint
        self.joint_torque[:, 6:] = 0.0
        # ---------------------------------------------------------------------------------------------------------

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= 1

        if time_out[0]:
            self.close_gripper(env_ids=np.array(range(self.num_envs)).reshape(-1))
            self._disassemble_plug_from_socket()

            if_intersect = (self.held_pos[:, 2] < self.fixed_pos[:, 2] + self.disassembly_dists).cpu().numpy()
            success_env_ids = np.argwhere(if_intersect == 0).reshape(-1)

            self._log_robot_state(success_env_ids)
            self._log_object_state(success_env_ids)
            self._save_log_traj()

        return time_out, time_out

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep

        rew_buf = self._update_rew_buf()
        return rew_buf

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        return torch.zeros((self.num_envs,), device=self.device)

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_DOOSAN_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        prev_fingertip_midpoint_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).unsqueeze(
            1
        )  # (num_envs, 1, 3)

        self.prev_fingertip_midpoint_pos = torch.repeat_interleave(
            prev_fingertip_midpoint_pos, self.cfg_task.num_point_robot_traj, dim=1
        )  # (num_envs, num_point_robot_traj, 3)

        self._init_log_data_per_episode()

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def _move_gripper_to_grasp_pose(self, env_ids):
        """Define grasp pose for plug and move gripper to pose."""
        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            gripper_goal_quat,
            gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = gripper_goal_pos.clone()

        # Set target rot
        self.ctrl_target_fingertip_midpoint_quat = gripper_goal_quat.clone()

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.50:
            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = factory_control._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:6] += delta_dof_pos[:, 0:6]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:6] = self.joint_pos[env_ids, 0:6]
            
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def _move_gripper_to_eef_pose(self, env_ids, goal_pos, goal_quat, sim_steps, if_log=False):

        for _ in range(sim_steps):
            if if_log:
                self._log_robot_state_per_timestep()

            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=goal_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=goal_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            self.actions *= 0.0
            self.actions[env_ids, :6] = delta_hand_pose

            is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
            # perform physics stepping
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # set actions into buffers
                self._apply_action()
                # set actions into simulator
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False)
                # render between steps only if the GUI or an RTX sensor needs it
                # note: we assume the render interval to be the shortest accepted rendering interval.
                #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                # update buffers at sim dt
                self.scene.update(dt=self.physics_dt)

            # Simulate and update tensors.
            self.step_sim_no_action()

    def _set_DOOSAN_to_default_pose(self, joints, env_ids):
        """Return Doosan M0609 to its default joint position."""
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 6:] = 0.0
        joint_pos[:, :6] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)

        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_fixed_initial_state(self, env_ids):
        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        fixed_state[:, 2] += self.cfg_task.fixed_asset_z_offset

        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat

        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel

        # (1.d.) Update values.
        self._fixed_asset.write_root_state_to_sim(fixed_state, env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids):
        # Set plug pos to assembled state
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0

        self._held_asset.write_root_state_to_sim(held_state)
        self._held_asset.reset()

        self.step_sim_no_action()

    def close_gripper(self, env_ids):
        # Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 1.00: # 0.25:
            self.ctrl_target_joint_pos[env_ids, 6:] = 0.0   # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        self.randomize_fixed_initial_state(env_ids)

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.socket_height
        fixed_tip_pos_local[:, 2] += self.socket_base_height

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        self.randomize_held_initial_state(env_ids)
        self._move_gripper_to_grasp_pose(env_ids)
        self.close_gripper(env_ids)

        self.prev_joint_pos = self.joint_pos[:, 0:6].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _disassemble_plug_from_socket(self):
        """Lift plug from socket till disassembly and then randomize end-effector pose."""
        if_intersect = np.ones(self.num_envs, dtype=np.float32)
        env_ids = np.argwhere(if_intersect == 1).reshape(-1)

        print(f"\n----- Step 1: lift gripper up to {self.disassembly_dists[0] * 3.0} meter")
        self._lift_gripper(self.disassembly_dists * 3.0, self.cfg_task.disassemble_sim_steps, env_ids)
        self.step_sim_no_action()

        if_intersect = (self.held_pos[:, 2] < self.fixed_pos[:, 2] + self.disassembly_dists).cpu().numpy()
        env_ids = np.argwhere(if_intersect == 0).reshape(-1)

        print(f"\n----- Step 2: randomize gripper pose\n")
        self._randomize_gripper_pose(self.cfg_task.move_gripper_sim_steps, env_ids)

    def _lift_gripper(self, lift_distance, sim_steps, env_ids=None):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""
        ctrl_tgt_pos = torch.empty_like(self.fingertip_midpoint_pos).copy_(self.fingertip_midpoint_pos)
        ctrl_tgt_quat = torch.empty_like(self.fingertip_midpoint_quat).copy_(self.fingertip_midpoint_quat)
        ctrl_tgt_pos[:, 2] += lift_distance
        if len(env_ids) == 0:
            env_ids = np.array(range(self.num_envs)).reshape(-1)

        self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log=True)

    def _randomize_gripper_pose(self, sim_steps, env_ids):
        """Move gripper to random pose."""
        ctrl_tgt_pos = torch.empty_like(self.fingertip_midpoint_pos).copy_(self.fingertip_midpoint_pos)
        ctrl_tgt_pos[:, 2] += self.cfg_task.gripper_rand_z_offset

        fingertip_centered_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5
        )  # [-1, 1]
        fingertip_centered_pos_noise = fingertip_centered_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.gripper_rand_pos_noise, device=self.device)
        )
        ctrl_tgt_pos += fingertip_centered_pos_noise

        # Set target rot
        ctrl_target_fingertip_centered_euler = torch.zeros((self.num_envs, 3), device=self.device)
        rx, ry, rz = euler_xyz_from_quat(quat=self.fingertip_midpoint_quat[:, :])
        ctrl_target_fingertip_centered_euler[:, 0] = rx
        ctrl_target_fingertip_centered_euler[:, 1] = ry
        ctrl_target_fingertip_centered_euler[:, 2] = rz

        fingertip_centered_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5
        )  # [-1, 1]
        fingertip_centered_rot_noise = fingertip_centered_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.gripper_rand_rot_noise, device=self.device)
        )
        ctrl_target_fingertip_centered_euler += fingertip_centered_rot_noise
        ctrl_tgt_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_centered_euler[:, 0],
            ctrl_target_fingertip_centered_euler[:, 1],
            ctrl_target_fingertip_centered_euler[:, 2],
        )

        self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log=True)

    def _init_log_data_per_assembly(self):

        self.log_assembly_id = []
        self.log_plug_pos = []
        self.log_plug_quat = []
        self.log_init_plug_pos = []
        self.log_init_plug_quat = []
        self.log_plug_grasp_pos = []
        self.log_plug_grasp_quat = []
        self.log_fingertip_centered_pos = []
        self.log_fingertip_centered_quat = []
        self.log_arm_dof_pos = []

    def _init_log_data_per_episode(self):

        self.log_fingertip_centered_pos_traj = []
        self.log_fingertip_centered_quat_traj = []
        self.log_arm_dof_pos_traj = []
        self.log_plug_pos_traj = []
        self.log_plug_quat_traj = []

        self.init_plug_grasp_pos = self.gripper_goal_pos.clone().detach()
        self.init_plug_grasp_quat = self.gripper_goal_quat.clone().detach()
        self.init_plug_pos = self.held_pos.clone().detach()
        self.init_plug_quat = self.held_quat.clone().detach()

    def _log_robot_state(self, env_ids):

        self.log_plug_pos += torch.stack(self.log_plug_pos_traj, dim=1)[env_ids].cpu().tolist()
        self.log_plug_quat += torch.stack(self.log_plug_quat_traj, dim=1)[env_ids].cpu().tolist()
        self.log_arm_dof_pos += torch.stack(self.log_arm_dof_pos_traj, dim=1)[env_ids].cpu().tolist()
        self.log_fingertip_centered_pos += (
            torch.stack(self.log_fingertip_centered_pos_traj, dim=1)[env_ids].cpu().tolist()
        )
        self.log_fingertip_centered_quat += (
            torch.stack(self.log_fingertip_centered_quat_traj, dim=1)[env_ids].cpu().tolist()
        )

    def _log_robot_state_per_timestep(self):

        self.log_plug_pos_traj.append(self.held_pos.clone().detach())
        self.log_plug_quat_traj.append(self.held_quat.clone().detach())
        self.log_arm_dof_pos_traj.append(self.joint_pos[:, 0:6].clone().detach())
        self.log_fingertip_centered_pos_traj.append(self.fingertip_midpoint_pos.clone().detach())
        self.log_fingertip_centered_quat_traj.append(self.fingertip_midpoint_quat.clone().detach())

    def _log_object_state(self, env_ids):

        self.log_plug_grasp_pos += self.init_plug_grasp_pos[env_ids].cpu().tolist()
        self.log_plug_grasp_quat += self.init_plug_grasp_quat[env_ids].cpu().tolist()
        self.log_init_plug_pos += self.init_plug_pos[env_ids].cpu().tolist()
        self.log_init_plug_quat += self.init_plug_quat[env_ids].cpu().tolist()

    def _save_log_traj(self):

        if len(self.log_arm_dof_pos) > self.cfg_task.num_log_traj:

            log_item = []
            for i in range(self.cfg_task.num_log_traj):
                curr_dict = dict({})
                curr_dict["fingertip_centered_pos"] = self.log_fingertip_centered_pos[i]
                curr_dict["fingertip_centered_quat"] = self.log_fingertip_centered_quat[i]
                curr_dict["arm_dof_pos"] = self.log_arm_dof_pos[i]
                curr_dict["plug_grasp_pos"] = self.log_plug_grasp_pos[i]
                curr_dict["plug_grasp_quat"] = self.log_plug_grasp_quat[i]
                curr_dict["init_plug_pos"] = self.log_init_plug_pos[i]
                curr_dict["init_plug_quat"] = self.log_init_plug_quat[i]
                curr_dict["plug_pos"] = self.log_plug_pos[i]
                curr_dict["plug_quat"] = self.log_plug_quat[i]

                log_item.append(curr_dict)

            log_filename = os.path.join(
                os.getcwd(), self.data_dir, self.cfg_task.assembly_id + "_disassemble_traj.json"
            )

            with open(log_filename, "w+") as out_file:
                json.dump(log_item, out_file, indent=6)

            print(f"Trajectory collection complete! Collected {len(self.log_arm_dof_pos)} trajectories!")
            exit(0)
            
        else:
            print(
                f"Collected {len(self.log_arm_dof_pos)} trajectories so far (target: > {self.cfg_task.num_log_traj})."
            )