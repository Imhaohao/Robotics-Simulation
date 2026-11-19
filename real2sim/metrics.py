"""Automatic checks. A bad scene fails here without anyone watching the video."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from real2sim.episode import Camera, Episode
from real2sim.scene import SceneSpec
from real2sim.stereo import image_centroid


@dataclass(frozen=True)
class Report:
    passed: bool
    checks: dict[str, bool]
    numbers: dict[str, float]


def project(camera: Camera, point: np.ndarray) -> np.ndarray:
    if camera.extrinsics is None:
        raise ValueError(camera.name)
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("xyz", camera.extrinsics[3:]).as_matrix()
    local = rotation.T @ (point - camera.extrinsics[:3])
    return np.array(
        [
            camera.fx * local[0] / local[2] + camera.cx,
            camera.fy * local[1] / local[2] + camera.cy,
        ]
    )


def _mask(frame_bgr: np.ndarray, name: str) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ranges = {
        "yellow": ((18, 140, 140), (42, 255, 255)),
        "blue": ((88, 60, 40), (125, 255, 255)),
        "teal": ((55, 40, 40), (90, 255, 255)),
    }
    low, high = ranges[name]
    return cv2.inRange(hsv, low, high) > 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def evaluate(
    episode: Episode,
    scene: SceneSpec,
    rollout_yellow: np.ndarray,
    rollout_cup: np.ndarray,
    rollout_teal: np.ndarray,
    cup_up: np.ndarray,
    joint_rmse: float,
    flange_rmse: float,
    teal_start: np.ndarray,
    real_frames: dict[str, np.ndarray],
    sim_frames: dict[str, np.ndarray],
) -> Report:
    numbers: dict[str, float] = {
        "joint_rmse_rad": joint_rmse,
        "flange_rmse_m": flange_rmse,
    }
    checks: dict[str, bool] = {
        "arm_tracks_log": joint_rmse < 0.08 and flange_rmse < 0.02,
    }
    dxy = float(np.linalg.norm(rollout_yellow[:2] - rollout_cup[:2]))
    rim = float(rollout_cup[2] + scene.cup_height / 2.0)
    bottom = float(rollout_cup[2] - scene.cup_height / 2.0)
    numbers["block_to_cup_m"] = dxy
    numbers["block_z_m"] = float(rollout_yellow[2])
    inside = dxy < scene.cup_bottom_radius and bottom + 0.004 < rollout_yellow[2] < rim - 0.004
    checks["block_ended_in_cup"] = bool(inside)
    teal_drift = float(np.linalg.norm(rollout_teal[:2] - teal_start[:2]))
    numbers["teal_drift_m"] = teal_drift
    checks["teal_cube_stayed"] = teal_drift < 0.03
    cup_drift = float(np.linalg.norm(rollout_cup[:2] - np.array(scene.cup_pos[:2])))
    numbers["cup_drift_m"] = cup_drift
    checks["cup_stayed"] = cup_drift < 0.04
    tilt = math.degrees(math.acos(float(np.clip(cup_up[2], -1.0, 1.0))))
    numbers["cup_tilt_deg"] = tilt
    checks["cup_upright"] = tilt < 20.0

    for name, point in (
        ("yellow", np.array(scene.block_pos)),
        ("blue", np.array(scene.cup_pos)),
        ("teal", np.array(scene.teal_pos)),
    ):
        for camera_name in ("ext1", "ext2"):
            camera = episode.cameras[camera_name]
            predicted = project(camera, point)
            observed = image_centroid(real_frames[camera_name], name)
            if observed is None:
                continue
            error = float(np.linalg.norm(predicted - observed))
            numbers[f"reproj_{camera_name}_{name}_px"] = error
    # ext1 sees the block most clearly. A wrong camera model is hundreds of pixels off.
    if "reproj_ext1_yellow_px" in numbers:
        checks["block_projects_near_real"] = numbers["reproj_ext1_yellow_px"] < 80.0
    if "reproj_ext1_blue_px" in numbers:
        checks["cup_projects_on_ext1"] = numbers["reproj_ext1_blue_px"] < 60.0
    if "reproj_ext2_teal_px" in numbers:
        checks["teal_projects_on_ext2"] = numbers["reproj_ext2_teal_px"] < 80.0

    sim = sim_frames["ext1"]
    if sim.size and float(sim.mean()) > 1.0:
        sim_bgr = cv2.cvtColor(sim, cv2.COLOR_RGB2BGR)
        real = cv2.resize(real_frames["ext1"], (sim_bgr.shape[1], sim_bgr.shape[0]))
        for name in ("yellow", "blue", "teal"):
            numbers[f"iou_ext1_{name}"] = _iou(_mask(real, name), _mask(sim_bgr, name))
        # Global color IoU is reported only. The wood shares the block's hue, so a
        # whole-image mask is not a reliable pass/fail. Reprojection is the check.
    passed = all(checks.values())
    return Report(passed=passed, checks=checks, numbers=numbers)
