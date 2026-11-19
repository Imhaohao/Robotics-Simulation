"""Franka flange kinematics and the Robotiq command that actually squeezes."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

PINCH_SITE = "g_pinch"
FLANGE_SITE = "attachment_site"


def menagerie(root: Path) -> tuple[Path, Path]:
    root = Path(root)
    arm = root / "franka_fr3" / "fr3.xml"
    hand = root / "robotiq_2f85" / "2f85.xml"
    if not arm.exists() or not hand.exists():
        raise FileNotFoundError(
            f"missing Franka/Robotiq models under {root}. Run scripts/fetch_assets.sh"
        )
    return arm, hand


def robot_spec(assets: Path, hand_yaw_deg: float = 88.0) -> mujoco.MjSpec:
    """Attach the 2F-85 so the fingers enter the wrist frame from the bottom.

    The Menagerie mount draws the hand from the side. A 90 degree yaw matches
    the real wrist picture and leaves the block on the rim when the log ends.
    88 degrees keeps that view and finishes the drop inside the logged horizon.
    The pinch stays on the flange axis.
    """
    arm, hand = menagerie(assets)
    spec = mujoco.MjSpec.from_file(str(arm))
    spec.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    spec.option.impratio = 10
    frame = spec.attach(
        mujoco.MjSpec.from_file(str(hand)),
        site=spec.site(FLANGE_SITE),
        prefix="g_",
    )
    half = float(np.deg2rad(hand_yaw_deg) / 2.0)
    frame.quat = [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]
    return spec


def site_positions(model: mujoco.MjModel, joint_position: np.ndarray, site: str) -> np.ndarray:
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    out = np.zeros((len(joint_position), 3), dtype=np.float64)
    for index, joints in enumerate(joint_position):
        data.qpos[:7] = joints
        mujoco.mj_kinematics(model, data)
        out[index] = data.site_xpos[site_id]
    return out


def gripper_commands(observed: np.ndarray) -> np.ndarray:
    """Follow the measured close, then squeeze once the fingers stall on the block.

    The log's target jumps to fully closed while the hand is still above the
    block. Replaying that target shuts the fingers early and knocks the block
    away. The measured width times the real close. Once that width stalls
    (the block is between the pads), command a firm close so the pads produce
    grasp force instead of hovering at the contact width.
    """
    command = np.array(observed, dtype=np.float64, copy=True)
    previous = 0.0
    for index, width in enumerate(observed):
        if width > 0.50 and width >= previous - 0.02:
            command[index] = 0.95
        previous = float(width)
    return command


def _box_half_along(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int, direction: np.ndarray) -> float:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    return float(sum(size[axis] * abs(rotation[:, axis] @ direction) for axis in range(3)))


def fingertip_below_pinch(model: mujoco.MjModel, joints: np.ndarray, command: float) -> tuple[float, float]:
    """Reach of the pads past the pinch, and the pad's half-depth along that axis.

    Both come from the gripper geoms at this pose. The cup rim is the reach
    plus the pad depth, so the face clears the rim when the hand slides off.
    """
    data = mujoco.MjData(model)
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PINCH_SITE)
    data.qpos[:7] = joints
    data.ctrl[:7] = joints
    data.ctrl[actuator] = float(np.clip(command, 0.0, 1.0) * 255.0)
    for _ in range(300):
        mujoco.mj_step(model, data)
    axis = data.site_xmat[site].reshape(3, 3)[:, 2]
    origin = data.site_xpos[site]
    furthest = 0.0
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("g_"):
            continue
        reach = float((data.geom_xpos[geom_id] - origin) @ axis) + float(model.geom_rbound[geom_id])
        furthest = max(furthest, reach)
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_right_pad1")
    rotation = data.geom_xmat[pad].reshape(3, 3)
    size = model.geom_size[pad]
    # Half-depth of the pad along the pinch axis. The rim has to clear this face.
    pad_depth = float(sum(size[axis_id] * abs(float(rotation[:, axis_id] @ axis)) for axis_id in range(3)))
    return furthest, pad_depth


def finger_inner_gap(model: mujoco.MjModel, joints: np.ndarray, command: float) -> float:
    """Distance between the inner pad faces at a normalized gripper command."""
    data = mujoco.MjData(model)
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "g_fingers_actuator")
    data.qpos[:7] = joints
    data.ctrl[:7] = joints
    data.ctrl[actuator] = float(np.clip(command, 0.0, 1.0) * 255.0)
    for _ in range(400):
        mujoco.mj_step(model, data)
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_right_pad1")
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_left_pad1")
    delta = data.geom_xpos[right] - data.geom_xpos[left]
    gap = float(np.linalg.norm(delta))
    direction = delta / max(gap, 1e-8)
    inner = gap - _box_half_along(model, data, right, direction) - _box_half_along(model, data, left, direction)
    return max(inner, 0.02)


def grasp_release_indices(gripper: np.ndarray) -> tuple[int, int]:
    held = np.flatnonzero(gripper > 0.50)
    if len(held) < 2:
        raise RuntimeError("the log never closes the gripper on an object")
    return int(held[0]), int(held[-1])
