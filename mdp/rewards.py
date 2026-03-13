# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward if object is lifted above minimal_height relative to its default reset height."""
    object: RigidObject = env.scene[object_cfg.name]
    lift_delta = object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2]
    return torch.where(lift_delta > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w.mean(dim=-2)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold (relative to reset/default height)
    lift_delta = object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2]
    return (lift_delta > minimal_height) * (1 - torch.tanh(distance / std))


def object_lift_height(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense lift-progress reward using height delta relative to default reset height."""
    object: RigidObject = env.scene[object_cfg.name]
    lift_delta = object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2]
    height_progress = (lift_delta - minimal_height) / max(target_height - minimal_height, 1e-6)
    return torch.clamp(height_progress, min=0.0, max=1.0)


def object_is_lifted_when_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    distance_threshold: float,
    closed_joint_pos_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Lift reward gated by grasp proxy (close + gripper closed)."""
    lifted = object_is_lifted(env, minimal_height=minimal_height, object_cfg=object_cfg)
    grasp_proxy = gripper_closed_near_object(
        env,
        distance_threshold=distance_threshold,
        closed_joint_pos_threshold=closed_joint_pos_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    return lifted * grasp_proxy


def object_lift_height_when_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    target_height: float,
    distance_threshold: float,
    closed_joint_pos_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Lift-height progress reward gated by grasp proxy (close + gripper closed)."""
    lift_progress = object_lift_height(
        env, minimal_height=minimal_height, target_height=target_height, object_cfg=object_cfg
    )
    grasp_proxy = gripper_closed_near_object(
        env,
        distance_threshold=distance_threshold,
        closed_joint_pos_threshold=closed_joint_pos_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    return lift_progress * grasp_proxy


def object_goal_success(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary success reward when lifted object reaches goal neighborhood."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    lift_delta = object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2]
    lifted = lift_delta > minimal_height
    return (lifted & (distance < threshold)).float()


def stagnation_near_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    speed_threshold: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty mask when end-effector is close to object but object is not being lifted/moved."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w.mean(dim=-2)
    dist = torch.norm(cube_pos_w - ee_w, dim=1)
    obj_speed = torch.norm(object.data.root_lin_vel_w[:, :3], dim=1)
    lift_delta = object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2]
    not_lifted = lift_delta <= minimal_height

    return ((dist < distance_threshold) & (obj_speed < speed_threshold) & not_lifted).float()


def gripper_closed_near_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    closed_joint_pos_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Proxy grasp reward: end-effector is close to object and gripper is closed."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w.mean(dim=-2)
    dist = torch.norm(cube_pos_w - ee_w, dim=1)
    near_object = dist < distance_threshold

    # For Franka fingers, lower joint position means a more closed gripper.
    mean_finger_pos = torch.mean(robot.data.joint_pos[:, robot_cfg.joint_ids], dim=1)
    gripper_closed = mean_finger_pos < closed_joint_pos_threshold
    return (near_object & gripper_closed).float()


def object_upward_velocity_near_ee(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    velocity_clip: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward for upward object motion when the end-effector is close."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w.mean(dim=-2)
    dist = torch.norm(cube_pos_w - ee_w, dim=1)
    near_object = (dist < distance_threshold).float()

    upward_speed = torch.clamp(object.data.root_lin_vel_w[:, 2], min=0.0, max=velocity_clip) / max(velocity_clip, 1e-6)
    return near_object * upward_speed


def singularity_penalty(
    env: ManagerBasedRLEnv,
    min_singular_value_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]),
) -> torch.Tensor:
    """Continuous penalty for near-singular Jacobian configurations using the minimum singular value."""
    robot: Articulation = env.scene[robot_cfg.name]

    if len(robot_cfg.body_ids) != 1:
        raise ValueError("singularity_penalty expects exactly one end-effector body in robot_cfg.body_names.")

    body_id = robot_cfg.body_ids[0]
    jacobians = robot.root_physx_view.get_jacobians()
    jacobian_body_id = body_id - 1 if robot.is_fixed_base else body_id

    # Use translational + rotational Jacobian rows for the selected arm joints.
    jacobian = jacobians[:, jacobian_body_id, :6, robot_cfg.joint_ids]

    singular_values = torch.linalg.svdvals(jacobian)
    min_singular_value = singular_values[:, -1]
    return (
        (min_singular_value_threshold - min_singular_value).clamp(min=0.0)
        / max(min_singular_value_threshold, 1e-6)
    )


def ee_close(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w.mean(dim=-2)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
   
    ee_close_rewards = torch.where(
        object_ee_distance <= 0.01,
        robot.data.joint_pos[:, 12] - robot.data.joint_pos[:, 13] + robot.data.joint_pos[:, 16] - robot.data.joint_pos[:, 17],
        torch.zeros_like(robot.data.joint_pos[:, 0])
    )
    # print("left1:", robot.data.joint_pos[:, 12])
    # print("left2:", robot.data.joint_pos[:, 13])
    # print("left11:", robot.data.joint_pos[:, 16])
    # print("left22:", robot.data.joint_pos[:, 17])
    return ee_close_rewards / 2

def object_ee_distance_close(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w.mean(dim=-2)# Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return object_ee_distance <= 0.01

