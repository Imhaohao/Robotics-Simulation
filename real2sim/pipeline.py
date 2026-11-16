"""End-to-end real2sim: episode data in, calibrated MuJoCo scene and checks out."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import numpy as np

from real2sim.episode import Episode, load_episode, read_video
from real2sim.kinematics import (
    finger_inner_gap,
    grasp_release_indices,
    gripper_commands,
    robot_spec,
    site_positions,
)
from real2sim.metrics import evaluate
from real2sim.replay import rollout
from real2sim.scene import SceneSpec, build_model
from real2sim.stereo import reconstruct
from real2sim.textures import median_rgba, write_textures


def _tuple4(values) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in values)


def _color_from_mask(frame_bgr: np.ndarray, name: str) -> np.ndarray | None:
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ranges = {
        "yellow": ((18, 210, 160), (42, 255, 255)),
        "blue": ((90, 100, 70), (115, 255, 255)),
        "teal": ((65, 80, 70), (88, 255, 220)),
    }
    low, high = ranges[name]
    mask = cv2.inRange(hsv, low, high)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    order = np.argsort(-stats[:, cv2.CC_STAT_AREA])
    for index in order[1:]:
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 80 <= area <= 8000:
            bgr = np.median(frame_bgr[labels == index], axis=0) / 255.0
            return np.array([bgr[2], bgr[1], bgr[0], 1.0])
    return None


def calibrate(episode: Episode, assets: Path, stereo_frame, texture_dir: Path | None) -> SceneSpec:
    robot = robot_spec(assets).compile()
    pinch = site_positions(robot, episode.joint_position, "g_pinch")
    grasp, release = grasp_release_indices(episode.gripper_position)
    # Only the first close. The opening pass crosses the same widths above the cup.
    approach = slice(max(0, grasp - 2), grasp + 1)
    block_xy = pinch[approach, :2].mean(axis=0)
    held = np.flatnonzero(episode.gripper_position > 0.50)
    after = held[held > grasp + 8]
    insert = int(after[np.argmin(pinch[after, 2])])
    cup_xy = pinch[insert, :2]

    plane, cloud, segments = reconstruct(episode.cameras["ext1"], stereo_frame)
    on_table = np.abs(plane.height(cloud.points)) < 0.008
    table_points = cloud.points[on_table]
    table_z = float(np.median(table_points[:, 2]))

    stall = float(np.median(episode.gripper_position[episode.gripper_position > 0.50]))
    gap = finger_inner_gap(robot, episode.joint_position[grasp], stall)
    half = float(np.clip(0.44 * gap, 0.012, 0.022))

    yellow = segments.get("yellow")
    blue = segments.get("blue")
    teal = segments.get("teal")
    left = stereo_frame[:, : stereo_frame.shape[1] // 2]
    sampled = {name: _color_from_mask(left, name) for name in ("yellow", "blue", "teal")}
    bias = block_xy - yellow.centroid[:2] if yellow is not None else np.zeros(2)
    if teal is not None:
        teal_xy = teal.centroid[:2] + bias
        teal_half = float(np.clip(float(np.median(teal.size[:2])) / 2.0, 0.012, 0.028))
        teal_rgba = _tuple4(sampled["teal"]) if sampled["teal"] is not None else _tuple4(teal.rgba)
    else:
        teal_xy = np.array([block_xy[0] - 0.18, block_xy[1] + 0.10])
        teal_half = 0.018
        teal_rgba = _tuple4(sampled["teal"]) if sampled["teal"] is not None else (0.16, 0.62, 0.52, 1.0)

    stereo_radius = float(np.max(blue.size[:2]) / 2.0) if blue is not None else 0.04
    bottom_r = float(np.clip(max(0.032, stereo_radius * 0.8), 0.03, 0.046))
    release_offset = float(np.linalg.norm(pinch[release, :2] - cup_xy))
    top_r = float(
        np.clip(max(0.050, bottom_r + 0.016, stereo_radius + 0.012, release_offset + half + 0.01), 0.05, 0.07)
    )
    # Keep the rim below the fingertip at insertion so the departing hand clears it.
    height = float(np.clip(float(pinch[insert, 2]) - table_z - 0.032, 0.055, 0.075))

    xs = [0.0, float(block_xy[0]), float(cup_xy[0]), float(teal_xy[0])]
    ys = [0.0, float(block_xy[1]), float(cup_xy[1]), float(teal_xy[1])]
    if len(table_points):
        xs += [float(np.percentile(table_points[:, 0], 4)), float(np.percentile(table_points[:, 0], 96))]
        ys += [float(np.percentile(table_points[:, 1], 4)), float(np.percentile(table_points[:, 1], 96))]
    x0, x1 = min(xs) - 0.28, max(xs) + 0.28
    y0, y1 = min(ys) - 0.28, max(ys) + 0.28
    wall = median_rgba(left, sat_max=40, value_min=40)
    return SceneSpec(
        table_z=table_z,
        table_center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        table_half=((x1 - x0) / 2.0, (y1 - y0) / 2.0),
        block_pos=(float(block_xy[0]), float(block_xy[1]), table_z + half),
        block_half=half,
        block_rgba=_tuple4(sampled["yellow"]) if sampled["yellow"] is not None else (0.99, 0.92, 0.15, 1.0),
        cup_pos=(float(cup_xy[0]), float(cup_xy[1]), table_z + height / 2.0),
        cup_height=height,
        cup_bottom_radius=bottom_r,
        cup_top_radius=top_r,
        cup_rgba=_tuple4(sampled["blue"]) if sampled["blue"] is not None else (0.25, 0.62, 0.9, 1.0),
        teal_pos=(float(teal_xy[0]), float(teal_xy[1]), table_z + teal_half),
        teal_half=teal_half,
        teal_rgba=teal_rgba,
        wall_rgba=_tuple4(wall),
        texture_dir=str(texture_dir) if texture_dir else None,
    )


def _write_video(path: Path, real: dict[str, list[np.ndarray]], sim: dict[str, list[np.ndarray]], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
    count = len(sim["ext1"])
    for index in range(count):
        rows = []
        for name in ("ext1", "ext2", "wrist"):
            left = cv2.cvtColor(cv2.resize(real[name][index], (640, 360)), cv2.COLOR_BGR2RGB)
            right = cv2.resize(sim[name][index], (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(left, f"real {name}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right, f"sim {name}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(np.hstack((left, right)))
        writer.append_data(np.vstack(rows))
    writer.close()


def _scene_dict(scene: SceneSpec) -> dict:
    raw = scene.__dict__.copy()
    return {key: (list(value) if isinstance(value, tuple) else value) for key, value in raw.items()}


def run(data: Path, assets: Path, out: Path, render: bool = True) -> dict:
    data = Path(data).resolve()
    assets = Path(assets).resolve()
    out = Path(out).resolve()
    import warnings

    warnings.filterwarnings("ignore", message="Attach conflict")
    episode = load_episode(data)
    # Frame i lines up with trajectory step i. Stereo is left|right of the same instant.
    stereo = read_video(episode.cameras["ext1"].stereo_video)[0]
    real = {name: read_video(camera.video) for name, camera in episode.cameras.items()}
    texture_dir = out / "textures"
    write_textures(stereo[:, : stereo.shape[1] // 2], texture_dir)
    scene = calibrate(episode, assets, stereo, texture_dir if render else texture_dir)
    # A second calibration must reproduce the scene. Stereo matching is deterministic.
    again = calibrate(episode, assets, stereo, texture_dir)
    if again.block_pos != scene.block_pos or again.cup_pos != scene.cup_pos:
        raise RuntimeError("calibration changed between two runs")
    model = build_model(scene, assets)
    result = rollout(
        model,
        episode,
        render=render,
        render_size=(720, 1280) if render else (360, 640),
    )
    report = evaluate(
        episode,
        scene,
        result.yellow,
        result.cup,
        result.teal,
        result.cup_up,
        result.joint_rmse,
        result.flange_rmse,
        np.array(scene.teal_pos),
        {name: frames[0] for name, frames in real.items()},
        {name: (frames[0] if frames else np.zeros((360, 640, 3), dtype=np.uint8)) for name, frames in result.frames.items()},
    )
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": episode.task,
        "scene": _scene_dict(scene),
        "gripper_command": "measured width, with a close bias after the fingers stall",
        "checks": report.checks,
        "numbers": report.numbers,
        "passed": report.passed,
        "deterministic_calibration": True,
    }
    (out / "scene.json").write_text(json.dumps(payload["scene"], indent=2))
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    if render:
        fps = float(1.0 / np.median(episode.dt))
        _write_video(out / "side_by_side.mp4", real, result.frames, fps)
        stills = out / "stills"
        stills.mkdir(exist_ok=True)
        for name in ("ext1", "ext2", "wrist"):
            for index in (0, 40, 80, 99):
                sim = cv2.cvtColor(cv2.resize(result.frames[name][index], (640, 360)), cv2.COLOR_RGB2BGR)
                real_frame = cv2.resize(real[name][index], (640, 360))
                cv2.imwrite(str(stills / f"{name}_{index:03d}.jpg"), np.hstack((real_frame, sim)))
    # Touch the command helper so an import-time refactor cannot drop it silently.
    assert gripper_commands(episode.gripper_position).shape == episode.gripper_position.shape
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate one Franka episode in MuJoCo")
    parser.add_argument("--data", type=Path, default=Path("data/episode"))
    parser.add_argument("--assets", type=Path, default=Path("third_party/mujoco_menagerie"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    payload = run(args.data, args.assets, args.out, render=not args.no_render)
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"], "numbers": payload["numbers"]}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)
