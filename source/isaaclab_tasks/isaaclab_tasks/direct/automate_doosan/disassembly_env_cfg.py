# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .disassembly_tasks_cfg import Extraction

# ---------------------------------------------------------------
import numpy as np
import os

DOOSAN_ASSET_DIR = os.getcwd() + f"/source/isaaclab_assets/data/doosan_robot_usd"
# ---------------------------------------------------------------

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    # --- Change in degrees of freedom: 7 (panda) → 6 (doosan) -----
    "joint_pos": 6,
    # --------------------------------------------------------------
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [0.01, 0.01, 0.01]

    pos_action_threshold = [0.01, 0.01, 0.01] 
    rot_action_threshold = [0.01, 0.01, 0.01]

    reset_joints = [0.0, 0.0, np.pi/2, 0.0, np.pi/2, 0.0]
    reset_task_prop_gains = [1000, 1000, 1000, 50, 50, 50]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [0.0, 0.0, np.pi/2, 0.0, np.pi/2, 0.0]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class DisassemblyEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = [
        "fingertip_pos_rel_fixed", 
        "fingertip_quat", 
        "ee_linvel", 
        "ee_angvel"
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "extraction"
    tasks: dict = {"extraction": Extraction()}
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 10.0
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,   # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,           # Important for stable simulation.
            # ------------------------------------
            gpu_collision_stack_size=200_000_000,
            # ------------------------------------
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # -----------------------------------------------------
            usd_path=f"{DOOSAN_ASSET_DIR}/m0609_2F85_combine.usd",
            # -----------------------------------------------------
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # ----------------------------------------
                "joint_1": 0.0,
                "joint_2": 0,
                "joint_3": np.pi/2, 
                "joint_4": 0.0,
                "joint_5": np.pi/2,
                "joint_6": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "finger_joint": 0.0,
                "right_outer_finger_joint": 0.0,
                "left_outer_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_knuckle_joint": 0.0,
                "left_inner_finger_knuckle_joint": 0.0
                # ----------------------------------------
            },
        ),
        actuators={
            # ----------------------------------------
            "doosan_arm1": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "doosan_arm2": ImplicitActuatorCfg(
                joint_names_expr=["joint_[5-6]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),

            "doosan_hand": ImplicitActuatorCfg(
                joint_names_expr=['finger_joint', 'right_inner_finger_joint', 'left_inner_finger_joint', 'right_inner_finger_knuckle_joint', 'left_inner_finger_knuckle_joint'],
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
            # ----------------------------------------
        },
    )
