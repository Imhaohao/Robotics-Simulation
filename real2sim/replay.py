"""Replay the logged joint trajectory and render the calibrated cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
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
    yellow_path: np.ndarray
    cup_path: np.ndarray
    background_match: float


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


def _gripper_command(episode: Episode, mode: str) -> np.ndarray:
    if mode == "stall":
        return gripper_commands(episode.gripper_position)
    if mode == "raw_target":
        # The logged target, including the early close that misses the block.
        return np.clip(episode.gripper_target, 0.0, 1.0)
    if mode == "open":
        return np.zeros(episode.steps, dtype=np.float64)
    raise ValueError(mode)


_PLATE_BODIES = {"world", "table_body", "mount", "gray_wall", "stripes"}


def _foreground_geoms(model: mujoco.MjModel) -> np.ndarray:
    ids = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body not in _PLATE_BODIES:
            ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)


def _object_mask(real_bgr: np.ndarray, table_only: bool) -> np.ndarray:
    """Compact yellow, blue, and teal blobs on the table. The striped wall is left alone."""
    hsv = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2HSV)
    ranges = (
        ((18, 200, 150), (42, 255, 255)),
        ((95, 90, 60), (115, 255, 255)),
        ((70, 80, 60), (88, 255, 210)),
    )
    mask = np.zeros(real_bgr.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    height = real_bgr.shape[0]
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not 30 <= area <= 8000:
            continue
        top = int(stats[index, cv2.CC_STAT_TOP])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        center_row = top + component_height / 2.0
        # On the fixed cameras the backdrop is the upper part of the frame.
        if table_only and center_row < height * 0.46:
            continue
        keep[labels == index] = 255
    if keep.max() == 0:
        return keep
    return cv2.dilate(keep, np.ones((5, 5), np.uint8))


def _over_plate(
    rgb: np.ndarray,
    seg: np.ndarray,
    real_bgr: np.ndarray,
    foreground: np.ndarray,
    table_only: bool,
) -> tuple[np.ndarray, float]:
    """Robot and objects stay rendered. The lab behind them is the real frame.

    The real block, cup, and cube are inpainted out of the plate first, so the
    rendered bodies are the only copies of those objects.
    """
    geom = int(mujoco.mjtObj.mjOBJ_GEOM)
    type_on_right = int((seg[:, :, 1] == geom).sum()) >= int((seg[:, :, 0] == geom).sum())
    obj_type = seg[:, :, 1] if type_on_right else seg[:, :, 0]
    obj_id = seg[:, :, 0] if type_on_right else seg[:, :, 1]
    mask = (obj_type == geom) & np.isin(obj_id, foreground)
    original = cv2.cvtColor(cv2.resize(real_bgr, (rgb.shape[1], rgb.shape[0])), cv2.COLOR_BGR2RGB)
    holes = _object_mask(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), table_only=table_only)
    plate = original.copy()
    if holes.max() > 0:
        plate = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), holes, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    plated = plate.copy()
    plated[mask] = rgb[mask]
    # Match is measured on the lab, not on the pixels we inpainted or redrew.
    keep = ~mask & (holes == 0)
    if not np.any(keep):
        return plated, 0.0
    delta = np.abs(plated[keep].astype(np.int16) - original[keep].astype(np.int16))
    return plated, float(1.0 - delta.mean() / 255.0)


def rollout(
    model: mujoco.MjModel,
    episode: Episode,
    render_size: tuple[int, int] = (360, 640),
    render: bool = True,
    gripper: str = "stall",
    real_frames: dict[str, list[np.ndarray]] | None = None,
    composite: bool = False,
    video_steps: int | None = None,
) -> Rollout:
    data = mujoco.MjData(model)
    actuators = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"fr3_joint{i}") for i in range(1, 8)
    ]
    fingers = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")
    flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    commands = _gripper_command(episode, gripper)
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
    yellow_path = np.zeros((episode.steps, 3), dtype=np.float64)
    cup_path = np.zeros((episode.steps, 3), dtype=np.float64)
    video_steps = min(100 if video_steps is None else video_steps, episode.steps)
    foreground = _foreground_geoms(model) if render and composite else np.zeros(0, dtype=np.int32)
    background_match = 0.0
    for index in range(episode.steps):
        for actuator, value in zip(actuators, episode.joint_position[index]):
            data.ctrl[actuator] = value
        data.ctrl[fingers] = float(commands[index] * 255.0)
        substeps = max(1, int(round(float(episode.dt[index]) / model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        joint_err.append(episode.joint_position[index] - data.qpos[:7])
        flange_err.append(episode.cartesian[index, :3] - data.site_xpos[flange])
        yellow_path[index], _ = _body(model, data, "yellow")
        cup_path[index], _ = _body(model, data, "cup")
        if not render or index >= video_steps:
            continue
        wrist_t, wrist_q = _camera_pose(episode.wrist_extrinsics[index])
        model.cam_pos[camera_ids["wrist"]] = wrist_t
        model.cam_quat[camera_ids["wrist"]] = wrist_q
        model.cam_fovy[camera_ids["wrist"]] = _fovy(episode.cameras["wrist"])
        # cam_xpos is filled by the last forward. The wrist pose is written after
        # that step, so the renderer would keep the previous frame's camera.
        mujoco.mj_forward(model, data)
        for name in frames:
            renderer.update_scene(data, camera=name)
            rgb = renderer.render().copy()
            if composite and real_frames is not None and index < len(real_frames[name]):
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=name)
                seg = renderer.render()
                renderer.disable_segmentation_rendering()
                rgb, match = _over_plate(
                    rgb, seg, real_frames[name][index], foreground, table_only=(name != "wrist")
                )
                if name == "ext1" and index == 0:
                    background_match = match
            frames[name].append(rgb)

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
        yellow_path=yellow_path,
        cup_path=cup_path,
        background_match=background_match,
    )
