"""Load the recorded episode: joints, gripper, cameras, and video frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class Camera:
    name: str
    serial: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    # Pose of the left camera in the robot base frame, xyz + euler xyz.
    # None for the wrist, whose pose is stored per step.
    extrinsics: np.ndarray | None
    video: Path
    stereo_video: Path | None


@dataclass(frozen=True)
class Episode:
    root: Path
    task: str
    joint_position: np.ndarray
    gripper_position: np.ndarray
    gripper_target: np.ndarray
    cartesian: np.ndarray
    dt: np.ndarray
    wrist_extrinsics: np.ndarray
    cameras: dict[str, Camera]

    @property
    def steps(self) -> int:
        return int(self.joint_position.shape[0])


def _camera_from_json(root: Path, name: str, payload: dict) -> Camera:
    stereo = payload.get("stereo_video")
    stereo_name = None
    if isinstance(stereo, str):
        stereo_name = stereo.split()[0]
    extr = payload.get("extrinsics_xyz_euler")
    extrinsics = np.asarray(extr, dtype=np.float64) if isinstance(extr, list) else None
    return Camera(
        name=name,
        serial=str(payload["serial"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        baseline_m=float(payload["baseline_m"]),
        extrinsics=extrinsics,
        video=root / payload["video"],
        stereo_video=(root / stereo_name) if stereo_name else None,
    )


def load_episode(root: Path) -> Episode:
    root = Path(root)
    meta = json.loads((root / "episode.json").read_text())
    calibration = json.loads((root / "calibration.json").read_text())
    cameras = {
        name: _camera_from_json(root, name, payload)
        for name, payload in calibration["cameras"].items()
    }
    with h5py.File(root / "trajectory.h5", "r") as handle:
        joints = handle["observation/robot_state/joint_positions"][:].astype(np.float64)
        gripper = handle["observation/robot_state/gripper_position"][:].astype(np.float64)
        target = handle["action/target_gripper_position"][:].astype(np.float64)
        cartesian = handle["observation/robot_state/cartesian_position"][:].astype(np.float64)
        wrist = handle["observation/camera_extrinsics/18026681_left"][:].astype(np.float64)
        stamps = handle["observation/timestamp/control/step_start"][:].astype(np.float64)
    deltas = np.diff(stamps, append=stamps[-1] + np.median(np.diff(stamps))) / 1000.0
    return Episode(
        root=root,
        task=str(meta["task"]),
        joint_position=joints,
        gripper_position=gripper,
        gripper_target=target,
        cartesian=cartesian,
        dt=deltas,
        wrist_extrinsics=wrist,
        cameras=cameras,
    )


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def split_stereo(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = frame.shape[1] // 2
    return frame[:, :half].copy(), frame[:, half:].copy()
