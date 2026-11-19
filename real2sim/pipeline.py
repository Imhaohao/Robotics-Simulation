"""End-to-end real2sim: episode data in, calibrated MuJoCo scene and checks out."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import numpy as np

from real2sim.episode import Episode, load_episode, read_video
from real2sim.kinematics import (
    finger_inner_gap,
    fingertip_below_pinch,
    grasp_release_indices,
    gripper_commands,
    robot_spec,
    site_positions,
)
from real2sim.metrics import evaluate, project
from real2sim.replay import rollout
from real2sim.scene import SceneSpec, build_model
from real2sim.stereo import measure_cup, reconstruct
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


def _seat_teal(episode: Episode, teal_xy: np.ndarray, z: float, ext1, ext1_bgr: np.ndarray) -> np.ndarray:
    """Slide the cube toward the corner the wrist actually shows.

    The ext1 cloud plus the block's bias leaves the whole cube in the approach
    frame. The real cube is a few pixels in the top-right corner. The slide
    stops when ext1's nearest teal pixel would move more than 8 px.
    """
    wrist_bgr = read_video(episode.cameras["wrist"].video)[40]
    hsv = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2HSV)
    sliver = cv2.inRange(hsv, (65, 40, 40), (100, 255, 255))
    sliver[80:, :] = 0
    sliver[:, :1000] = 0
    ys, xs = np.nonzero(sliver)
    if len(xs) < 20:
        return teal_xy
    corner = np.array([float(xs.mean()), float(ys.mean())])
    # Past the corner, so the body of the cube leaves the frame.
    target = corner + np.array([50.0, -40.0])
    wrist = replace(episode.cameras["wrist"], extrinsics=episode.wrist_extrinsics[40])
    point = np.array([float(teal_xy[0]), float(teal_xy[1]), z], dtype=np.float64)
    base = project(wrist, point)
    jac = np.zeros((2, 2), dtype=np.float64)
    for axis in range(2):
        nudged = point.copy()
        nudged[axis] += 0.01
        jac[:, axis] = (project(wrist, nudged) - base) / 0.01
    step = np.linalg.lstsq(jac, target - base, rcond=None)[0]
    ext1_hsv = cv2.cvtColor(ext1_bgr, cv2.COLOR_BGR2HSV)
    ext1_mask = cv2.inRange(ext1_hsv, (70, 80, 60), (90, 255, 220))
    ext1_mask[:300, :] = 0
    ey, ex = np.nonzero(ext1_mask)
    if len(ex) < 20:
        return teal_xy
    ext1_pts = np.stack([ex, ey], axis=1).astype(np.float64)

    def nearest(xy: np.ndarray) -> float:
        predicted = project(ext1, np.array([xy[0], xy[1], z]))
        return float(np.min(np.linalg.norm(ext1_pts - predicted.reshape(1, 2), axis=1)))

    start = nearest(teal_xy)
    for scale in np.linspace(1.0, 0.0, 11):
        trial = teal_xy + step * float(scale)
        if nearest(trial) <= start + 8.0:
            return trial
    return teal_xy


def calibrate(episode: Episode, assets: Path, stereo_frame, texture_dir: Path | None) -> tuple[SceneSpec, dict]:
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
    yellow = segments.get("yellow")
    blue = segments.get("blue")
    teal = segments.get("teal")
    # Stereo extent of the yellow blob, kept smaller than the opening the fingers
    # actually stalled at so the pads can close around the faces.
    if yellow is not None:
        half = float(np.median(yellow.size[:2]) / 2.0)
        block_source = "stereo"
    else:
        half = float(0.5 * gap)
        block_source = "finger_gap"
    if half >= 0.5 * gap:
        half = float(0.46 * gap)
        block_source = "finger_gap"
    block_clipped = half < 0.008 or half > 0.040
    half = float(np.clip(half, 0.008, 0.040))

    left = stereo_frame[:, : stereo_frame.shape[1] // 2]
    sampled = {name: _color_from_mask(left, name) for name in ("yellow", "blue", "teal")}
    bias = block_xy - yellow.centroid[:2] if yellow is not None else np.zeros(2)
    if teal is not None:
        teal_xy = teal.centroid[:2] + bias
        teal_half = float(np.clip(float(np.median(teal.size[:2])) / 2.0, 0.012, 0.028))
        teal_rgba = _tuple4(sampled["teal"]) if sampled["teal"] is not None else _tuple4(teal.rgba)
        teal_xy = _seat_teal(episode, teal_xy, table_z + teal_half, episode.cameras["ext1"], left)
    else:
        teal_xy = np.array([block_xy[0] - 0.18, block_xy[1] + 0.10])
        teal_half = 0.018
        teal_rgba = _tuple4(sampled["teal"]) if sampled["teal"] is not None else (0.16, 0.62, 0.52, 1.0)

    if blue is None:
        raise RuntimeError("the ext1 stereo frame has no blue cup")
    cloud_hsv = cv2.cvtColor(cloud.colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    blue_mask = (cloud_hsv[:, 0] >= 88) & (cloud_hsv[:, 0] <= 115) & (cloud_hsv[:, 1] >= 80) & (cloud_hsv[:, 2] >= 40)
    hull_height, hull_bottom, hull_top = measure_cup(
        cloud.points[blue_mask], plane.height(cloud.points)[blue_mask], blue.centroid[:2]
    )
    # A clip this wide does not bind the measured cup. It only rejects a broken cloud.
    bottom_r = float(np.clip(hull_bottom, 0.015, 0.06))
    top_r = float(np.clip(hull_top, 0.015, 0.06))
    height = float(np.clip(hull_height, 0.03, 0.12))
    cup_hull_clipped = (bottom_r != hull_bottom) or (top_r != hull_top) or (height != hull_height)
    protrusion, pad_depth = fingertip_below_pinch(
        robot, episode.joint_position[insert], float(episode.gripper_position[insert])
    )

    xs = [0.0, float(block_xy[0]), float(cup_xy[0]), float(teal_xy[0])]
    ys = [0.0, float(block_xy[1]), float(cup_xy[1]), float(teal_xy[1])]
    if len(table_points):
        xs += [float(np.percentile(table_points[:, 0], 4)), float(np.percentile(table_points[:, 0], 96))]
        ys += [float(np.percentile(table_points[:, 1], 4)), float(np.percentile(table_points[:, 1], 96))]
    x0, x1 = min(xs) - 0.28, max(xs) + 0.28
    y0, y1 = min(ys) - 0.28, max(ys) + 0.28
    wall = median_rgba(left, sat_max=40, value_min=40)
    provenance = {
        "finger_gap_m": gap,
        "block_half_m": half,
        "block_source": block_source,
        "block_width_clipped": block_clipped,
        "stereo_cup_radius_m": hull_bottom,
        "hull_height_m": hull_height,
        "hull_bottom_radius_m": hull_bottom,
        "hull_top_radius_m": hull_top,
        "cup_hull_clipped": cup_hull_clipped,
        "fingertip_protrusion_m": protrusion,
        "pad_depth_m": pad_depth,
    }
    scene = SceneSpec(
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
    return scene, provenance


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


def _perturb(scene: SceneSpec, block_scale: float, table_lift: float) -> SceneSpec:
    if table_lift:
        scene = replace(
            scene,
            table_z=scene.table_z + table_lift,
            block_pos=(scene.block_pos[0], scene.block_pos[1], scene.block_pos[2] + table_lift),
            cup_pos=(scene.cup_pos[0], scene.cup_pos[1], scene.cup_pos[2] + table_lift),
            teal_pos=(scene.teal_pos[0], scene.teal_pos[1], scene.teal_pos[2] + table_lift),
        )
    if block_scale != 1.0:
        half = scene.block_half * block_scale
        scene = replace(
            scene,
            block_half=half,
            block_pos=(scene.block_pos[0], scene.block_pos[1], scene.table_z + half),
        )
    return scene


def _mask_distance(frame_bgr: np.ndarray, name: str, pixel: np.ndarray) -> float | None:
    """Pixels from a projected point to the nearest real object pixel. None if the colour is absent.

    The first range is the bright face. In the gripper the block is darker and the
    wrist cup is a shadowed face, so later ranges cover those. The distance is the
    closest range that is still the object. A range covering most of the frame is
    not used.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    candidates = {
        "yellow": (
            ((18, 210, 160), (42, 255, 255)),
            ((18, 190, 100), (42, 255, 255)),
        ),
        "blue": (
            ((88, 80, 40), (115, 255, 255)),
            ((88, 60, 30), (120, 255, 255)),
            ((70, 25, 50), (120, 255, 255)),
        ),
    }
    # The bright range is the highlight. On the grasp frame the projection sits
    # on the darker face of the same block, so the closest of these masks is
    # the distance. A mask that covers more than a fifth of the frame is the table.
    best = None
    area = hsv.shape[0] * hsv.shape[1]
    for low, high in candidates[name]:
        trial = cv2.inRange(hsv, low, high)
        count = int((trial > 0).sum())
        if count < 20 or count > 0.2 * area:
            continue
        points = np.column_stack(np.nonzero(trial))
        # nonzero is (row, col). pixel is (u, v) = (col, row).
        delta = points[:, ::-1] - pixel.reshape(1, 2)
        distance = float(np.min(np.linalg.norm(delta, axis=1)))
        if best is None or distance < best:
            best = distance
    return best


def run(
    data: Path,
    assets: Path,
    out: Path,
    render: bool = True,
    gripper: str = "stall",
    block_scale: float = 1.0,
    table_lift: float = 0.0,
    weld_block: bool = False,
    write: bool = True,
) -> dict:
    data = Path(data).resolve()
    assets = Path(assets).resolve()
    out = Path(out).resolve()
    import warnings

    warnings.filterwarnings("ignore", message="Attach conflict")
    started = time.perf_counter()
    episode = load_episode(data)
    # Frame i lines up with trajectory step i. Stereo is left|right of the same instant.
    stereo = read_video(episode.cameras["ext1"].stereo_video)[0]
    real = {name: read_video(camera.video) for name, camera in episode.cameras.items()}
    texture_dir = out / "textures"
    write_textures(stereo[:, : stereo.shape[1] // 2], texture_dir, real["wrist"][40])
    scene, provenance = calibrate(episode, assets, stereo, texture_dir)
    # A second calibration must reproduce the scene. Stereo matching is deterministic.
    again, _ = calibrate(episode, assets, stereo, texture_dir)
    if again.block_pos != scene.block_pos or again.cup_pos != scene.cup_pos:
        raise RuntimeError("calibration changed between two runs")
    scene = _perturb(scene, block_scale, table_lift)
    model = build_model(scene, assets, weld_block=weld_block)
    result = rollout(
        model,
        episode,
        render=render,
        render_size=(720, 1280) if render else (360, 640),
        gripper=gripper,
        real_frames=real if render else None,
        composite=render,
        draw_logged_arm=render,
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
    grasp_index, _ = grasp_release_indices(episode.gripper_position)
    for when, index in (("frame0", 0), ("grasp", int(grasp_index))):
        cameras = {
            "ext1": episode.cameras["ext1"],
            "ext2": episode.cameras["ext2"],
            "wrist": replace(episode.cameras["wrist"], extrinsics=episode.wrist_extrinsics[index]),
        }
        for camera_name, camera in cameras.items():
            for label, point in (
                ("yellow", result.yellow_path[index]),
                ("blue", result.cup_path[index]),
            ):
                predicted = project(camera, point)
                distance = _mask_distance(real[camera_name][min(index, len(real[camera_name]) - 1)], label, predicted)
                if distance is not None:
                    report.numbers[f"reproj_{when}_{camera_name}_{label}_px"] = distance
    if render:
        report.numbers["background_match"] = result.background_match
    simulate_s = time.perf_counter() - started
    import mujoco

    free = []
    for name in ("yellow_free", "cup_free", "teal_free"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        free.append(int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE))
    payload = {
        "task": episode.task,
        "scene": _scene_dict(scene),
        "provenance": provenance,
        "gripper": gripper,
        "block_scale": block_scale,
        "table_lift": table_lift,
        "weld_block": weld_block,
        "free_bodies": all(free),
        "bodies": {
            "yellow": result.yellow.tolist(),
            "cup": result.cup.tolist(),
            "teal": result.teal.tolist(),
        },
        "gripper_command": {
            "stall": "measured width, with a close bias after the fingers stall",
            "raw_target": "logged target gripper, including the early close",
            "open": "gripper held open",
        }[gripper],
        "checks": report.checks,
        "numbers": report.numbers,
        "passed": report.passed,
        "deterministic_calibration": True,
        "simulate_s": simulate_s,
        "block_peak_z": float(np.max(result.yellow_path[:, 2])),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.json").write_text(json.dumps(payload["scene"], indent=2))
        (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    if render and write:
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
