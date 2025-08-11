# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import numpy as np
import os
import torch
from datetime import datetime

import carb
import isaacsim.core.utils.torch as torch_utils
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import axis_angle_from_quat

from . import automate_algo_utils as automate_algo
from . import automate_log_utils as automate_log
from . import factory_control, factory_utils
from .assembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, AssemblyEnvCfg
from .soft_dtw_cuda import SoftDTW

# ---------------------------------------------------------------
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from .asset_information import get_plug_info, get_socket_info
# ---------------------------------------------------------------


class AssemblyEnv(DirectRLEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
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

        # Get the information of plug and socket
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
            self.pos_action_bounds = [self.plug_height, self.plug_height, self.plug_height]        
        # ---------------------------------------------------------------------------------------------------------------

        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        # Create criterion for dynamic time warping (later used for imitation reward)
        self.soft_dtw_criterion = SoftDTW(use_cuda=True, gamma=self.cfg_task.soft_dtw_gamma)

        # Evaluate
        if self.cfg_task.if_logging_eval:
            self._init_eval_logging()

        if self.cfg_task.sample_from != "rand":
            self._init_eval_loading()

        # ---------------------------------------------------------------------------------------------------------------
        # Variables for measuring assembly success rate during testing
        if not self.cfg_task.if_train:
            self.test_attempt = 0
            self.total_test_attempt = 8     # if num_envs=128, total attempts = 128*8 = 1024            
            self.total_success_rates = np.array([])
        # ---------------------------------------------------------------------------------------------------------------

    def _init_eval_loading(self):
        eval_held_asset_pose, eval_fixed_asset_pose, eval_success = automate_log.load_log_from_hdf5(
            self.cfg_task.eval_filename
        )
        if self.cfg_task.sample_from == "gp":
            self.gp = automate_algo.model_succ_w_gp(eval_held_asset_pose, eval_fixed_asset_pose, eval_success)
        elif self.cfg_task.sample_from == "gmm":
            self.gmm = automate_algo.model_succ_w_gmm(eval_held_asset_pose, eval_fixed_asset_pose, eval_success)

    def _init_eval_logging(self):
        self.held_asset_pose_log = torch.empty(
            (0, 7), dtype=torch.float32, device=self.device
        )  # (position, quaternion)
        self.fixed_asset_pose_log = torch.empty((0, 7), dtype=torch.float32, device=self.device)
        self.success_log = torch.empty((0, 1), dtype=torch.float32, device=self.device)

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

        self.plug_grasps, self.disassembly_dists = self._load_assembly_info()
        self.curriculum_height_bound, self.curriculum_height_step = self._get_curriculum_info(self.disassembly_dists)
        self._load_disassembly_data()

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

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # SBC = False
        self.curr_max_disp = self.curriculum_height_bound[:, 1]

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

    def _get_curriculum_info(self, disassembly_dists):
        """Calculate the ranges and step sizes for Sampling-based Curriculum (SBC) in each environment."""
        curriculum_height_bound = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        curriculum_height_step = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        curriculum_height_bound[:, 1] = disassembly_dists + self.cfg_task.curriculum_freespace_range

        curriculum_height_step[:, 0] = curriculum_height_bound[:, 1] / self.cfg_task.num_curriculum_step
        curriculum_height_step[:, 1] = -curriculum_height_step[:, 0] / 2.0

        return curriculum_height_bound, curriculum_height_step

    def _load_disassembly_data(self):
        """Load pre-collected disassembly trajectories (end-effector position only)."""

        disassembly_traj_path = os.path.join(os.getcwd(), self.data_dir, f'{self.cfg_task.assembly_id}_disassemble_traj.json')
        with open(disassembly_traj_path, "r") as in_file:
            disassembly_traj = json.load(in_file)

        eef_pos_traj = []
        for i in range(len(disassembly_traj)):
            curr_ee_traj = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))
            curr_ee_goal = np.asarray(disassembly_traj[i]["fingertip_centered_pos"]).reshape((-1, 3))[0, :]

            # offset each trajectory to be relative to the goal
            eef_pos_traj.append(curr_ee_traj - curr_ee_goal)

        self.eef_pos_traj = torch.tensor(eef_pos_traj, dtype=torch.float32, device=self.device).squeeze()

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

        # --- Stop the robot's actions when the assembly is successful in the test environment. -----
        if not self.cfg_task.if_train:
            if len(self.nonzero_success_ids_for_test)>0:
                for tmp_success_ids in self.nonzero_success_ids_for_test:
                    action[tmp_success_ids] = torch.zeros(6, device=self.device)
        # -------------------------------------------------------------------------------------------

        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

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
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_curr_successes(self):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.plug_base_z_offset, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.plug_base_z_offset,
            self.num_envs,
            self.device,
        )

        # XY dist threshold to target (2.5mm)
        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
        is_centered = torch.where(
            xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        
        # Height threshold to target (1mm)
        is_close_or_below = torch.where(
            z_disp < 0.001, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )

        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        return curr_successes

    def _get_curr_engaged(self):
        """Get engaged mask at current timestep."""
        curr_engaged = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.plug_base_z_offset, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.plug_base_z_offset,
            self.num_envs,
            self.device,
        )

        # XY dist threshold to target (2.5mm)
        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
        is_centered = torch.where(
            xy_dist < 0.0025, torch.ones_like(curr_engaged), torch.zeros_like(curr_engaged)
        )
        
        # Height threshold to target
        engaged_threshold = (self.socket_height - self.plug_base_z_offset) * 0.9 + self.plug_base_z_offset
        is_close_or_below = torch.where(
            z_disp < engaged_threshold, torch.ones_like(curr_engaged), torch.zeros_like(curr_engaged)
        )

        curr_engaged = torch.logical_and(is_centered, is_close_or_below)

        return curr_engaged

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        # ---------------------------------------------------------------------------------------------------
        if not self.cfg_task.if_train:
            self.nonzero_success_ids_for_test = self.ep_succeeded.nonzero(as_tuple=False).squeeze(-1)

            self.total_success_rates = np.append(self.total_success_rates, self.extras["successes"].item())
            self.test_attempt += 1
            print(f"=== {self.test_attempt} attempt =========================================")
            print(f'Average success rates: {self.extras["successes"].item()*100:.2f}') 

            if self.test_attempt >= self.total_test_attempt:
                total_success_avg = np.mean(self.total_success_rates)
                total_success_std = np.std(self.total_success_rates)
                print(f"\n!!!!! Total Success rates, Avg: {total_success_avg*100:.2f}, Std: {total_success_std*100:.2f} !!!!!\n")
                self.extras["terminate"] = True
                exit(0)
        # ---------------------------------------------------------------------------------------------------

    def _get_rewards(self):
        """Update rewards and compute success statistics."""

        # Get successful and failed envs at current timestep
        curr_successes = self._get_curr_successes()

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["imitation"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        # --- 1) Compute the imitation reward --------------------------------------------------------
        curr_eef_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).reshape(
            -1, 3
        )  # relative position instead of absolute position
        
        imitation_reward = automate_algo.get_imitation_reward_from_dtw(
            self.eef_pos_traj, curr_eef_pos, self.prev_fingertip_midpoint_pos, self.soft_dtw_criterion, self.device
        )

        self.prev_fingertip_midpoint_pos = torch.cat(
            (self.prev_fingertip_midpoint_pos[:, 1:, :], curr_eef_pos.unsqueeze(1).clone().detach()), dim=1
        )
        
        # --- 2) Compute the keypoint-based reward ---------------------------------------------------
        # Compute pos of keypoints on held asset, and fixed asset in world frame
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.plug_base_z_offset, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.plug_base_z_offset,
            self.num_envs,
            self.device,
        )

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

        # Action penalties.
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_engaged().clone().float()

        # --- 3) Compute the final reward ------------------------------------------------------------
        # Use key-point based reward separated into XY alignment and Z alignment components.
        if self.cfg_task.use_xy_align:
            keypoint_dist_xy = torch.norm(keypoints_held[:, :, 0:2] - keypoints_fixed[:, :, 0:2], p=2, dim=-1).mean(-1)
            keypoint_dist_z = torch.norm(keypoints_held[:, :, 2] - keypoints_fixed[:, :, 2], p=2, dim=-1).mean(-1)

            a0, b0 = self.cfg_task.keypoint_coef_baseline_xy
            a1, b1 = self.cfg_task.keypoint_coef_coarse_xy        
            a2, b2 = self.cfg_task.keypoint_coef_fine_xy

            a0z, b0z = self.cfg_task.keypoint_coef_baseline_z
            a1z, b1z = self.cfg_task.keypoint_coef_coarse_z
            a2z, b2z = self.cfg_task.keypoint_coef_fine_z

            rew_dict = {
                "kp_baseline_xy": factory_utils.squashing_fn(keypoint_dist_xy, a0, b0),
                "kp_coarse_xy": factory_utils.squashing_fn(keypoint_dist_xy, a1, b1),
                "kp_fine_xy": factory_utils.squashing_fn(keypoint_dist_xy, a2, b2),
                "kp_baseline_z": factory_utils.squashing_fn(keypoint_dist_z, a0z, b0z),
                "kp_coarse_z": factory_utils.squashing_fn(keypoint_dist_z, a1z, b1z),
                "kp_fine_z": factory_utils.squashing_fn(keypoint_dist_z, a2z, b2z),
                "imitation": imitation_reward,
                "action_penalty_ee": action_penalty_ee,
                "action_grad_penalty": action_grad_penalty,
                "curr_engaged": curr_engaged.float(),
                "curr_success": curr_successes.float(),
            }
            rew_scales = {
                "kp_baseline_xy": 1.0,
                "kp_coarse_xy": 1.0,
                "kp_fine_xy": 1.0,
                "kp_baseline_z": 1.0,
                "kp_coarse_z": 1.0,
                "kp_fine_z": 1.0,
                "imitation": 1.0,
                "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
                "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
                "curr_engaged": 1.0,
                "curr_success": 1.0,
            }

        # Use keypoint-based reward integrated into XYZ alignment components.
        else:
            keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

            a0, b0 = self.cfg_task.keypoint_coef_baseline
            a1, b1 = self.cfg_task.keypoint_coef_coarse
            a2, b2 = self.cfg_task.keypoint_coef_fine

            rew_dict = {
                "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
                "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
                "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
                "imitation": imitation_reward,
                "action_penalty_ee": action_penalty_ee,
                "action_grad_penalty": action_grad_penalty,
                "curr_engaged": curr_engaged.float(),
                "curr_success": curr_successes.float(),
            }
            rew_scales = {
                "kp_baseline": 1.0,
                "kp_coarse": 1.0,
                "kp_fine": 1.0,
                "imitation": 1.0,
                "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
                "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
                "curr_engaged": 1.0,
                "curr_success": 1.0,
            }
        return rew_dict, rew_scales

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_DOOSAN_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        if self.cfg_task.if_logging_eval:
            self.held_asset_pose_log = torch.cat(
                [self.held_asset_pose_log, torch.cat([self.held_pos, self.held_quat], dim=1)], dim=0
            )
            self.fixed_asset_pose_log = torch.cat(
                [self.fixed_asset_pose_log, torch.cat([self.fixed_pos, self.fixed_quat], dim=1)], dim=0
            )

        prev_fingertip_midpoint_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).unsqueeze(
            1
        )  # (num_envs, 1, 3)

        self.prev_fingertip_midpoint_pos = torch.repeat_interleave(
            prev_fingertip_midpoint_pos, self.cfg_task.num_point_robot_traj, dim=1
        )  # (num_envs, num_point_robot_traj, 3)

        self.nonzero_success_ids_for_test = torch.tensor([], device=self.device)

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
        # fixed_state[:, 2] += self.cfg_task.fixed_asset_z_offset

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

    def randomize_held_initial_state(self, env_ids, pre_grasp):

        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp

        if pre_grasp:
            self.curriculum_disp = self.curr_max_disp + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )

            if self.cfg_task.sample_from == "rand":
                rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
                held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                held_asset_init_pos_rand = torch.tensor(
                    self.cfg_task.held_asset_init_pos_noise, dtype=torch.float32, device=self.device
                )
                self.held_pos_init_rand = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)

            if self.cfg_task.sample_from == "gp":
                rand_sample = torch.rand((self.cfg_task.num_gp_candidates, 3), dtype=torch.float32, device=self.device)
                held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                held_asset_init_pos_rand = torch.tensor(
                    self.cfg_task.held_asset_init_pos_noise, dtype=torch.float32, device=self.device
                )
                held_asset_init_candidates = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)
                self.held_pos_init_rand, _ = automate_algo.propose_failure_samples_batch_from_gp(
                    self.gp, held_asset_init_candidates.cpu().detach().numpy(), len(env_ids), self.device
                )

            if self.cfg_task.sample_from == "gmm":
                self.held_pos_init_rand = automate_algo.sample_rel_pos_from_gmm(self.gmm, len(env_ids), self.device)

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0

        held_state[env_ids, 2] += self.curriculum_disp

        plug_in_freespace_idx = torch.argwhere(self.curriculum_disp > self.disassembly_dists)
        held_state[plug_in_freespace_idx, :2] += self.held_pos_init_rand[plug_in_freespace_idx, :2]

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

        self.randomize_held_initial_state(env_ids, pre_grasp=True)
        self._move_gripper_to_grasp_pose(env_ids)
        self.randomize_held_initial_state(env_ids, pre_grasp=False)
        self.close_gripper(env_ids)

        self.prev_joint_pos = self.joint_pos[:, 0:6].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        
        # Back out what actions should be for initial state.
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        pos_actions = self.fingertip_midpoint_pos - self.fixed_pos_action_frame
        _pos_action_bounds = torch.tensor(self.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / _pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))