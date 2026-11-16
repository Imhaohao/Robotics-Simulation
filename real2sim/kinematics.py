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


def robot_spec(assets: Path) -> mujoco.MjSpec:
    arm, hand = menagerie(assets)
    spec = mujoco.MjSpec.from_file(str(arm))
    spec.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    spec.option.impratio = 10
    spec.attach(
        mujoco.MjSpec.from_file(str(hand)),
        site=spec.site(FLANGE_SITE),
        prefix="g_",
    )
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
