"""Stereo geometry for the table and the objects that the arm never grasps."""

from __future__ import annotations

from dataclasses import dataclass

import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from real2sim.episode import Camera, split_stereo

SEED = 0


@dataclass(frozen=True)
class Cloud:
    points: np.ndarray
    colors_bgr: np.ndarray
    pixels: np.ndarray


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    offset: float

    def height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset


@dataclass(frozen=True)
class Segment:
    name: str
    centroid: np.ndarray
    size: np.ndarray
    rgba: np.ndarray
    pixel_centroid: np.ndarray


def disparity_map(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 10,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        disp12MaxDiff=2,
        uniquenessRatio=7,
        speckleWindowSize=120,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0


def unproject(camera: Camera, left: np.ndarray, disp: np.ndarray, step: int = 2) -> Cloud:
    if camera.extrinsics is None:
        raise ValueError(camera.name)
    valid = disp > 0.8
    ys, xs = np.mgrid[0 : disp.shape[0] : step, 0 : disp.shape[1] : step]
    ys = ys.ravel()
    xs = xs.ravel()
    depth_px = disp[ys, xs]
    keep = depth_px > 0.8
    ys, xs, depth_px = ys[keep], xs[keep], depth_px[keep]
    z = camera.fx * camera.baseline_m / depth_px
    points_cam = np.stack(
        ((xs - camera.cx) * z / camera.fx, (ys - camera.cy) * z / camera.fy, z),
        axis=1,
    )
    rotation = Rotation.from_euler("xyz", camera.extrinsics[3:]).as_matrix()
    points = (rotation @ points_cam.T).T + camera.extrinsics[:3]
    return Cloud(points=points, colors_bgr=left[ys, xs], pixels=np.stack((xs, ys), axis=1))


def fit_table(points: np.ndarray, iterations: int = 600, thresh: float = 0.007) -> Plane:
    band = (
        (points[:, 2] > -0.2)
        & (points[:, 2] < 0.2)
        & (np.linalg.norm(points[:, :2], axis=1) < 1.6)
    )
    pool = points[band]
    if len(pool) < 100:
        raise RuntimeError("not enough stereo points to fit the table")
    rng = np.random.default_rng(SEED)
    best_count = -1
    best_mask = None
    for _ in range(iterations):
        choice = pool[rng.choice(len(pool), 3, replace=False)]
        normal = np.cross(choice[1] - choice[0], choice[2] - choice[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        if normal[2] < 0:
            normal = -normal
        if normal[2] < 0.95:
            continue
        offset = -float(normal @ choice[0])
        mask = np.abs(pool @ normal + offset) < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None:
        raise RuntimeError("table plane search failed")
    inliers = pool[best_mask]
    design = np.c_[inliers[:, :2], np.ones(len(inliers))]
    coeff, *_ = np.linalg.lstsq(design, inliers[:, 2], rcond=None)
    normal = np.array([coeff[0], coeff[1], -1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.median(inliers @ normal))
    return Plane(normal=normal, offset=offset)


def _largest_blob(points: np.ndarray, radius: float = 0.045) -> np.ndarray:
    if len(points) == 0:
        return points
    bins = np.round(points[:, :2] / 0.02).astype(np.int32)
    keys, counts = np.unique(bins, axis=0, return_counts=True)
    center_bin = keys[np.argmax(counts)]
    center = (center_bin + 0.5) * 0.02
    near = np.linalg.norm(points[:, :2] - center, axis=1) < radius
    chosen = points[near]
    if len(chosen) < 20:
        return points
    median = np.median(chosen[:, :2], axis=0)
    tight = np.linalg.norm(chosen[:, :2] - median, axis=1) < radius
    return chosen[tight]


def _segment(
    name: str,
    points: np.ndarray,
    colors: np.ndarray,
    hsv: np.ndarray,
    height: np.ndarray,
    hue: tuple[int, int],
    sat: int,
    val: int,
) -> Segment | None:
    mask = (
        (hsv[:, 0] >= hue[0])
        & (hsv[:, 0] <= hue[1])
        & (hsv[:, 1] >= sat)
        & (hsv[:, 2] >= val)
        & (height > 0.012)
        & (height < 0.12)
    )
    if int(mask.sum()) < 25:
        return None
    chosen = points[mask]
    colors_kept = colors[mask]
    blob = _largest_blob(chosen)
    if len(blob) < 20:
        return None
    median_xy = np.median(blob[:, :2], axis=0)
    color_gate = np.linalg.norm(chosen[:, :2] - median_xy, axis=1) < 0.045
    bgr = np.median(colors_kept[color_gate], axis=0) / 255.0
    rgba = np.array([bgr[2], bgr[1], bgr[0], 1.0], dtype=np.float64)
    low, high = np.percentile(blob, [10, 90], axis=0)
    return Segment(
        name=name,
        centroid=np.median(blob, axis=0),
        size=np.maximum(high - low, 0.02),
        rgba=rgba,
        pixel_centroid=np.zeros(2),
    )


def reconstruct(camera: Camera, stereo_bgr: np.ndarray) -> tuple[Plane, Cloud, dict[str, Segment]]:
    left, right = split_stereo(stereo_bgr)
    disp = disparity_map(left, right)
    cloud = unproject(camera, left, disp, step=2)
    plane = fit_table(cloud.points)
    height = plane.height(cloud.points)
    x0, x1 = np.percentile(cloud.points[np.abs(height) < 0.01][:, 0], [8, 92]) if np.any(np.abs(height) < 0.01) else (-0.2, 1.0)
    y0, y1 = np.percentile(cloud.points[np.abs(height) < 0.01][:, 1], [8, 92]) if np.any(np.abs(height) < 0.01) else (-0.5, 0.5)
    inside = (
        (cloud.points[:, 0] > x0)
        & (cloud.points[:, 0] < x1)
        & (cloud.points[:, 1] > y0)
        & (cloud.points[:, 1] < y1)
    )
    hsv = cv2.cvtColor(cloud.colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    specs = {
        "yellow": ((18, 40), 150, 150),
        "blue": ((88, 115), 80, 40),
        "teal": ((60, 86), 60, 70),
    }
    found: dict[str, Segment] = {}
    for name, (hue, sat, val) in specs.items():
        gate = inside if name != "yellow" else inside
        segment = _segment(
            name,
            cloud.points[gate],
            cloud.colors_bgr[gate],
            hsv[gate],
            height[gate],
            hue,
            sat,
            val,
        )
        if segment is not None:
            found[name] = segment
    return plane, cloud, found


def image_centroid(frame_bgr: np.ndarray, name: str) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ranges = {
        # Wood is yellow too, but much less saturated than the block.
        "yellow": ((18, 210, 160), (42, 255, 255)),
        "blue": ((88, 80, 40), (115, 255, 255)),
        "teal": ((60, 60, 70), (86, 255, 255)),
    }
    low, high = ranges[name]
    mask = cv2.inRange(hsv, low, high)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    # Skip the giant wood-colored component for yellow by preferring a
    # compact blob. The table fails this for blue and teal.
    best = None
    best_score = -1.0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < 40 or area > 6000:
            continue
        if max(width, height) > 160 or min(width, height) < 8:
            continue
        if max(width, height) / max(1, min(width, height)) > 3.0:
            continue
        component = labels == index
        score = float(hsv[:, :, 1][component].mean()) * math.sqrt(area)
        if score > best_score:
            best_score = score
            best = centroids[index].astype(np.float64)
    return best
