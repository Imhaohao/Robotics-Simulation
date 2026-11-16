"""Replay the logged joint trajectory and render the calibrated cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from real2sim.episode import Camera, Episode
from real2sim.kinematics import gripper_commands


@dataclass
class Rollout:
    joint_rmse: float
    flange_rmse: float
    yellow: np.ndarray
    cup: np.ndarray
    teal: np.ndarray
    cup_up: np.ndarray
    frames: dict[str, list[np.ndarray]]


def _fovy(camera: Camera) -> float:
    return math.degrees(2.0 * math.atan((0.5 * camera.height) / camera.fy))


def _camera_pose(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(extrinsics[:3], dtype=np.float64)
    world_from_opencv = Rotation.from_euler("xyz", extrinsics[3:]).as_matrix()
    # MuJoCo cameras look along -Z with +Y up. OpenCV looks along +Z with +Y down.
    world_from_mujoco = world_from_opencv @ np.diag([1.0, -1.0, -1.0])
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(world_from_mujoco).reshape(9))
    return translation, quat


def _body(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


def rollout(
    model: mujoco.MjModel,
    episode: Episode,
    render_size: tuple[int, int] = (360, 640),
    render: bool = True,
) -> Rollout:
    data = mujoco.MjData(model)
    actuators = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"fr3_joint{i}") for i in range(1, 8)
    ]
    fingers = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")
    flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    commands = gripper_commands(episode.gripper_position)
    camera_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        for name in ("ext1", "ext2", "wrist")
    }
    for name, camera in episode.cameras.items():
        if camera.extrinsics is None:
            continue
        translation, quat = _camera_pose(camera.extrinsics)
        camera_ids[name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        model.cam_pos[camera_ids[name]] = translation
        model.cam_quat[camera_ids[name]] = quat
        model.cam_fovy[camera_ids[name]] = _fovy(camera)

    data.qpos[:7] = episode.joint_position[0]
    for actuator, value in zip(actuators, episode.joint_position[0]):
        data.ctrl[actuator] = value
    data.ctrl[fingers] = 0.0
    settle = int(0.4 / model.opt.timestep)
    for _ in range(settle):
        mujoco.mj_step(model, data)

    height, width = render_size
    renderer = mujoco.Renderer(model, height, width) if render else None
    frames = {name: [] for name in ("ext1", "ext2", "wrist")}
    joint_err = []
    flange_err = []
    video_steps = min(100, episode.steps)
    for index in range(episode.steps):
        for actuator, value in zip(actuators, episode.joint_position[index]):
            data.ctrl[actuator] = value
        data.ctrl[fingers] = float(commands[index] * 255.0)
        substeps = max(1, int(round(float(episode.dt[index]) / model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        joint_err.append(episode.joint_position[index] - data.qpos[:7])
        flange_err.append(episode.cartesian[index, :3] - data.site_xpos[flange])
        if not render or index >= video_steps:
            continue
        wrist_t, wrist_q = _camera_pose(episode.wrist_extrinsics[index])
        model.cam_pos[camera_ids["wrist"]] = wrist_t
        model.cam_quat[camera_ids["wrist"]] = wrist_q
        model.cam_fovy[camera_ids["wrist"]] = _fovy(episode.cameras["wrist"])
        for name in frames:
            renderer.update_scene(data, camera=name)
            frames[name].append(renderer.render().copy())

    yellow, _ = _body(model, data, "yellow")
    cup, cup_rot = _body(model, data, "cup")
    teal, _ = _body(model, data, "teal")
    joint_err = np.stack(joint_err)
    flange_err = np.stack(flange_err)
    return Rollout(
        joint_rmse=float(np.sqrt(np.mean(joint_err**2))),
        flange_rmse=float(np.sqrt(np.mean(np.sum(flange_err**2, axis=1)))),
        yellow=yellow,
        cup=cup,
        teal=teal,
        cup_up=cup_rot[:, 2].copy(),
        frames=frames,
    )
