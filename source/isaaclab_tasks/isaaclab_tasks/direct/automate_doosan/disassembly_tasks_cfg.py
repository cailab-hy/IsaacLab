# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
    base_height: float = 0.0    # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0       # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    friction: float = 0.75


@configclass
class DisassemblyTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # Robot
    palm_to_finger_dist: float = 0.1134

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.10, 0.10, 0.05] # [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0

    # number of waypoints included in the end-effector trajectory
    num_point_robot_traj: int = 10 


@configclass
class Peg(HeldAssetCfg):
    mass = 0.019


@configclass
class Hole(FixedAssetCfg):
    height = 0.050896
    base_height = 0.0


@configclass
class Extraction(DisassemblyTask):
    name = "extraction"

    assembly_id = '00175'
    plug_usd_path = f"{USD_ASSET_DIR}/{assembly_id}_plug/{assembly_id}_plug.usd"
    socket_usd_path = f"{USD_ASSET_DIR}/{assembly_id}_socket/{assembly_id}_socket.usd"
    num_log_traj = 100

    fixed_asset_cfg = Hole()
    held_asset_cfg = Peg()
    asset_size = 8.0
    duration_s = 10.0

    move_gripper_sim_steps = 64
    disassemble_sim_steps = 64

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.10, 0.10, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0 # 10.0
    fixed_asset_z_offset: float = 0.1435

    gripper_rand_pos_noise: list = [0.05, 0.05, 0.05]
    gripper_rand_rot_noise: list = [0.174533, 0.174533, 0.174533]   # +-10 deg for roll/pitch/yaw
    gripper_rand_z_offset: float = 0.05

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path= socket_usd_path,
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
            # ---------------------
            pos=(0.3, 0.0, 0.05),
            # ---------------------
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path = plug_usd_path,
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
                enabled_self_collisions=True,
                fix_root_link=False,  # add this so the fixed asset is set to have a fixed base
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