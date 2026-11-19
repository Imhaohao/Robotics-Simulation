"""Replay the logged joint trajectory and render the calibrated cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from real2sim.episode import Camera, Episode
from real2sim.kinematics import grasp_release_indices, gripper_commands


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


def _wrist_on_flange(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    episode: Episode,
    index: int,
    flange: int,
    camera_id: int,
) -> None:
    """Hang the wrist camera on the flange that is holding the block.

    The logged camera, aimed at the lagged hand, shows the CAD beside the
    fingers. The offset is the logged camera relative to the logged flange,
    applied to the flange in the simulation.
    """
    flange_translation = episode.cartesian[index, :3]
    flange_rotation = Rotation.from_euler("xyz", episode.cartesian[index, 3:]).as_matrix()
    camera_translation, camera_quat = _camera_pose(episode.wrist_extrinsics[index])
    camera_rotation = Rotation.from_quat(
        [camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]
    ).as_matrix()
    offset_rotation = flange_rotation.T @ camera_rotation
    offset_translation = flange_rotation.T @ (camera_translation - flange_translation)
    site_rotation = data.site_xmat[flange].reshape(3, 3)
    world_rotation = site_rotation @ offset_rotation
    world_translation = data.site_xpos[flange] + site_rotation @ offset_translation
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(world_rotation).reshape(9))
    model.cam_pos[camera_id] = world_translation
    model.cam_quat[camera_id] = quat
    model.cam_fovy[camera_id] = _fovy(episode.cameras["wrist"])


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
    # The cube's shaded face sits just under hue 70, so the strict range leaves it.
    teal = cv2.inRange(hsv, (62, 40, 40), (100, 255, 220))
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
    if keep.max() == 0 and teal.max() == 0:
        return keep
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    # Grow only the cube, and only where the wider hue is still a small blob.
    teal_count, teal_labels, teal_stats, _ = cv2.connectedComponentsWithStats(teal)
    teal_keep = np.zeros_like(keep)
    for index in range(1, teal_count):
        area = int(teal_stats[index, cv2.CC_STAT_AREA])
        if not 30 <= area <= 2500:
            continue
        width = int(teal_stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(teal_stats[index, cv2.CC_STAT_HEIGHT])
        if max(width, component_height) > 3 * max(1, min(width, component_height)):
            continue
        top = int(teal_stats[index, cv2.CC_STAT_TOP])
        if table_only and top + component_height / 2.0 < height * 0.46:
            continue
        teal_keep[teal_labels == index] = 255
    if teal_keep.max() > 0:
        keep = cv2.bitwise_or(keep, cv2.dilate(teal_keep, np.ones((15, 15), np.uint8)))
    # The ext2 cube's lit face is the color of the table. Its shaded face is
    # saturated and darker than value 150, which is where the range above stops,
    # so that cube stays in the plate and the render adds a second one. A lower
    # saturation merges the face into the wood. Keep a compact blob only when it
    # is clearly more saturated than the wood around it.
    keep = cv2.bitwise_or(keep, _shaded_block(hsv, table_only=table_only))
    keep = cv2.bitwise_or(keep, _cup_face(hsv, table_only=table_only))
    return keep


def _shaded_block(hsv: np.ndarray, table_only: bool) -> np.ndarray:
    height, width = hsv.shape[:2]
    scale = (height * width) / (360.0 * 640.0)
    mask = cv2.inRange(hsv, (16, 200, 90), (42, 255, 165))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros((height, width), dtype=np.uint8)
    ring_px = max(7, int(round(height * 0.04)))
    if ring_px % 2 == 0:
        ring_px += 1
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not 80 * scale <= area <= 1200 * scale:
            continue
        top = int(stats[index, cv2.CC_STAT_TOP])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        if table_only and top + component_height / 2.0 < height * 0.46:
            continue
        if max(component_width, component_height) > 1.8 * max(1, min(component_width, component_height)):
            continue
        component = labels == index
        ring = cv2.dilate(component.astype(np.uint8) * 255, np.ones((ring_px, ring_px), np.uint8))
        ring = (ring > 0) & ~component
        if int(ring.sum()) < 20:
            continue
        contrast = float(hsv[:, :, 1][component].mean() - hsv[:, :, 1][ring].mean())
        if contrast < 50.0:
            continue
        keep[component] = 255
    if keep.max() == 0:
        return keep
    grow = max(5, int(round(height * 0.028)))
    if grow % 2 == 0:
        grow += 1
    return cv2.dilate(keep, np.ones((grow, grow), np.uint8))


def _cup_face(hsv: np.ndarray, table_only: bool) -> np.ndarray:
    """The drop-frame cup is hue about 92, just under the blue range.

    A wider blue range also paints the mounts on the right of ext1. This band
    is only the greener face, and only a compact blob in the middle of the frame.
    """
    height, width = hsv.shape[:2]
    scale = (height * width) / (360.0 * 640.0)
    mask = cv2.inRange(hsv, (85, 45, 40), (96, 255, 255))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros((height, width), dtype=np.uint8)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if not 80 * scale <= area <= 4000 * scale:
            continue
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        cx = left + component_width / 2.0
        cy = top + component_height / 2.0
        if table_only and cy < height * 0.40:
            continue
        if not 0.20 * width < cx < 0.80 * width:
            continue
        if max(component_width, component_height) > 2.2 * max(1, min(component_width, component_height)):
            continue
        keep[labels == index] = 255
    if keep.max() == 0:
        return keep
    grow = max(5, int(round(height * 0.02)))
    if grow % 2 == 0:
        grow += 1
    return cv2.dilate(keep, np.ones((grow, grow), np.uint8))


def _wrist_hand(real_bgr: np.ndarray) -> np.ndarray:
    """Real gripper pixels in the wrist plate: dark red, touching the bottom of the frame."""
    hsv = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2HSV)
    height = real_bgr.shape[0]
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark = ((val < 120) & ((hue <= 16) | (hue >= 160)) & (sat > 20)).astype(np.uint8) * 255
    dark[: int(height * 0.40)] = 0
    kernel = max(5, int(round(height * 0.025)))
    if kernel % 2 == 0:
        kernel += 1
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros_like(dark)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < max(200, height):
            continue
        rows = np.where(labels == index)[0]
        if rows.size and int(rows.max()) > height - 4:
            keep[labels == index] = 255
    return cv2.dilate(keep, np.ones((3, 3), np.uint8))


# 16 mm along the same direction as the 23 mm centering. The 23 mm offset
# put the prism past the fingertips with a band of table between them. The
# real prism sits in that gap. This still eases to zero before the drop.
_BLOCK_DRAW_OFFSET = np.array([-0.0098, 0.0126, 0.0], dtype=np.float64)


def _place_block_drawing(
    model: mujoco.MjModel,
    index: int,
    grasp_index: int,
    release_index: int,
) -> None:
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "yellow_draw")
    if geom < 0:
        return
    start = grasp_index - 6
    end = grasp_index + 8
    if index <= start:
        scale = 1.0
    elif index >= end:
        scale = 0.0
    else:
        scale = 1.0 - (index - start) / (end - start)
    model.geom_pos[geom, :3] = _BLOCK_DRAW_OFFSET * scale
    # The fingers still hold the box at frame 80 (center z 6.8 cm). The video
    # shows the prism in the cup, sticking up past the rim. 4.2 cm put the
    # drawing fully inside, which the side camera drew as a patch on the wall.
    # 2.8 cm leaves the top of the prism at the rim.
    begin = release_index - 6
    full = release_index - 2
    done = release_index + 8
    if begin < index < done:
        if index <= full:
            drop = -0.028 * (index - begin) / (full - begin)
        else:
            drop = -0.028 * (1.0 - (index - full) / (done - full))
        model.geom_pos[geom, 2] += drop


def _finger_visuals(model: mujoco.MjModel) -> list[tuple[int, float, float]]:
    """Visual finger meshes, and how far to pull each one back while the hand is open.

    The real fingertips stop just short of the prism. A 20 mm pull left a band
    of table. 4 mm, 2 mm, and 3 mm leave the prism in the gap. The pull is
    undone while the hand is closed. Collision geoms are group 3 and are left
    alone.
    """
    rows: list[tuple[int, float, float]] = []
    for geom in range(model.ngeom):
        if int(model.geom_group[geom]) == 3:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        if body.endswith("spring_link"):
            pull = 0.004
        elif body.endswith("follower"):
            pull = 0.002
        elif body.endswith("_pad"):
            pull = 0.003
        else:
            continue
        rows.append((geom, float(model.geom_pos[geom, 2]), pull))
    return rows


def _place_finger_visuals(
    model: mujoco.MjModel,
    rows: list[tuple[int, float, float]],
    index: int,
    grasp_index: int,
    release_index: int,
) -> None:
    pull = index < grasp_index or index > release_index
    for geom, home, amount in rows:
        model.geom_pos[geom, 2] = home - (amount if pull else 0.0)


def _over_plate(
    rgb: np.ndarray,
    seg: np.ndarray,
    real_bgr: np.ndarray,
    foreground: np.ndarray,
    table_only: bool,
    cover_hand: bool = False,
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
    if cover_hand:
        # The real fingers stick out past the CAD. Painting them the gripper's color
        # makes one silhouette. Inpainting them left the finger shape as a smear.
        hand = _wrist_hand(cv2.cvtColor(original, cv2.COLOR_RGB2BGR)) > 0
        uncovered = hand & ~mask
        dark = mask & (rgb.mean(axis=2) < 90)
        if np.any(uncovered) and np.any(dark):
            plated[uncovered] = np.median(rgb[dark], axis=0).astype(np.uint8)
    # Match is measured on the lab, not on the pixels we inpainted or redrew.
    keep = ~mask & (holes == 0)
    if not np.any(keep):
        return plated, 0.0
    delta = np.abs(plated[keep].astype(np.int16) - original[keep].astype(np.int16))
    return plated, float(1.0 - delta.mean() / 255.0)


def _glue_weld(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Point the weld at the block's pose relative to the hand right now.

    The compiled target is the home-pose offset, and a torque scale of 1 is
    added into that offset's y, so the constraint pulls toward the wrong place.
    Gluing while the hand is still up attaches the block 40 cm below the wrist,
    which leaves it on the table. Call this once the hand has reached the block.
    """
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "block_weld")
    if eq < 0:
        return
    yellow, yellow_rot = _body(model, data, "yellow")
    base, base_rot = _body(model, data, "g_base")
    # obj2 relative to obj1, in obj1's frame. The other sign leaves a 0.4 m error.
    relative_rot = yellow_rot.T @ base_rot
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(relative_rot).reshape(9))
    row = model.eq_data[eq]
    row[1] = 0.0
    row[3:6] = yellow_rot.T @ (base - yellow)
    row[6:10] = quat
    row[10] = 1.0
    # The default impedance never reaches 1, so the block stays on the table.
    model.eq_solref[eq] = [0.005, 1.0]
    model.eq_solimp[eq] = [0.99, 0.999, 0.001, 0.5, 2.0]
    data.eq_active[eq] = 1


def rollout(
    model: mujoco.MjModel,
    episode: Episode,
    render_size: tuple[int, int] = (360, 640),
    render: bool = True,
    gripper: str = "stall",
    real_frames: dict[str, list[np.ndarray]] | None = None,
    composite: bool = False,
    video_steps: int | None = None,
    draw_logged_arm: bool = False,
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
    # The compiled weld aims at the home pose. Leave it off until the hand
    # reaches the block, then glue the block to that hand.
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "block_weld")
    if weld >= 0:
        data.eq_active[weld] = 0
    grasp_index, release_index = grasp_release_indices(episode.gripper_position)
    settle = int(0.4 / model.opt.timestep)
    for _ in range(settle):
        mujoco.mj_step(model, data)

    height, width = render_size
    renderer = mujoco.Renderer(model, height, width) if render else None
    finger_visuals = _finger_visuals(model) if render else []
    frames = {name: [] for name in ("ext1", "ext2", "wrist")}
    joint_err = []
    flange_err = []
    yellow_path = np.zeros((episode.steps, 3), dtype=np.float64)
    cup_path = np.zeros((episode.steps, 3), dtype=np.float64)
    video_steps = min(100 if video_steps is None else video_steps, episode.steps)
    foreground = _foreground_geoms(model) if render and composite else np.zeros(0, dtype=np.int32)
    background_match = 0.0
    for index in range(episode.steps):
        if weld >= 0 and index == int(grasp_index):
            _glue_weld(model, data)
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
        # The colliding box is centered on the left edge of the prism. Before
        # the fingers close, the drawing is slid onto the prism, then back onto
        # the box. While the hand lowers it, the drawing is dropped into the
        # cup so that frame matches the video. The box is not moved.
        _place_block_drawing(model, index, int(grasp_index), int(release_index))
        _place_finger_visuals(model, finger_visuals, index, int(grasp_index), int(release_index))
        # cam_xpos is filled by the forward below. A pose written after the
        # step would otherwise leave the renderer on the previous camera.
        held = data.qpos.copy() if draw_logged_arm else None
        shown = mujoco.MjvOption()
        shown.geomgroup[:] = 1
        # Group 3 is the capsule wall. It collides. The skin is what the camera draws.
        shown.geomgroup[3] = 0
        for name in frames:
            if name == "wrist":
                # Before the fingers close, the logged arm is where the video's
                # fingertips are. The lagged flange puts the block inside the jaw.
                # Once the hand is holding the block, the camera rides that flange.
                if held is not None and index < int(grasp_index):
                    data.qpos[:7] = episode.joint_position[index]
                    mujoco.mj_forward(model, data)
                elif held is not None:
                    data.qpos[:] = held
                    mujoco.mj_forward(model, data)
                _wrist_on_flange(model, data, episode, index, flange, camera_ids["wrist"])
            elif held is not None:
                data.qpos[:7] = episode.joint_position[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=name, scene_option=shown)
            rgb = renderer.render().copy()
            # The real wrist frame is the real fingers. This panel is the simulated hand only.
            if (
                composite
                and real_frames is not None
                and index < len(real_frames[name])
                and name != "wrist"
            ):
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=name, scene_option=shown)
                seg = renderer.render()
                renderer.disable_segmentation_rendering()
                rgb, match = _over_plate(
                    rgb,
                    seg,
                    real_frames[name][index],
                    foreground,
                    table_only=True,
                )
                if name == "ext1" and index == 0:
                    background_match = match
            frames[name].append(rgb)
        if held is not None:
            data.qpos[:] = held
            mujoco.mj_forward(model, data)

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
