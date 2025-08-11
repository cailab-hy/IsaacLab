# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR    # ISAACLAB_NUCLEUS_DIR

# --------------------------------------------------------
USD_ASSET_DIR= f"{ISAACLAB_ASSETS_DATA_DIR}/automate_usd"
# --------------------------------------------------------


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    friction: float = 0.75


@configclass
class AssemblyTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    # Use key-point based reward separated into XY alignment and Z alignment components.
    use_xy_align: bool = True

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # Robot
    palm_to_finger_dist: float = 0.1134

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.10, 0.10, 0.05] # [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0

    # Held Asset (applies to all tasks)
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]

    # 1) Reward
    action_penalty_ee_scale: float = 0.0
    action_grad_penalty_scale: float = 0.0

    # 1-1) keypint-based reward
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]   # General movement towards fixed object.
    keypoint_coef_coarse: list = [50, 2]    # Movement to align the assets.
    keypoint_coef_fine: list = [100, 0]     # Smaller distances for threading or last-inch insertion.
    # XY scale
    keypoint_coef_baseline_xy: list = [50, 2]   # General movement towards fixed object.
    keypoint_coef_coarse_xy: list = [100, 0]    # Movement to align the assets.
    keypoint_coef_fine_xy: list = [200, -0.5]   # Smaller distances for threading or last-inch insertion.
    # Z scale
    keypoint_coef_baseline_z: list = [5, 4]   # General movement towards fixed object.
    keypoint_coef_coarse_z: list = [50, 2]    # Movement to align the assets.
    keypoint_coef_fine_z: list = [100, 0]     # Smaller distances for threading or last-inch insertion.

    # 1-2) imitation reward
    soft_dtw_gamma: float = 0.01    # set to 0 if want to use the original DTW without any smoothing
    num_point_robot_traj: int = 10  # number of waypoints included in the end-effector trajectory

    # SBC
    curriculum_freespace_range: float = 0.01
    num_curriculum_step: int = 10
    curriculum_height_step: list = [
        -0.005,
        0.003,
    ]  # how much to increase max initial downward displacement after hitting success or failure thresh

    # Logging evaluation results
    if_train: bool = True
    if_logging_eval: bool = False
    eval_filename: str = 'evaluation_00175.h5'

    # Fine-tuning
    sample_from: str = "rand"       # gp, gmm, idv, rand
    num_gp_candidates: int = 1000


@configclass
class Peg(HeldAssetCfg):
    mass = 0.019


@configclass
class Hole(FixedAssetCfg):
    height = 0.050896
    base_height = 0.0


@configclass
class Insertion(AssemblyTask):
    name = "insertion"
    assembly_id = '00175'
    plug_usd_path = f"{USD_ASSET_DIR}/{assembly_id}_plug/{assembly_id}_plug.usd"
    socket_usd_path = f"{USD_ASSET_DIR}/{assembly_id}_socket/{assembly_id}_socket.usd"

    fixed_asset_cfg = Hole()
    held_asset_cfg = Peg()
    asset_size = 8.0
    duration_s = 10.0

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.10, 0.10, 0.05] # [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360 # 10.0
    fixed_asset_z_offset: float = 0.1435

    # Held Asset (applies to all tasks)
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01] # noise level of the held asset in gripper

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=socket_usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
                enabled_self_collisions=True,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=plug_usd_path,
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
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={}
        ),
        actuators={}
    )