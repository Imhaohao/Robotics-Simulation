"""Appearance sampled from the real frame: wood tone, wall tone, stripe column."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _wood_patch(frame_bgr: np.ndarray, wrist_bgr: np.ndarray | None = None) -> np.ndarray:
    # The wrist frame's upper table is wood grain with no hand in it. Averaging
    # that photo to one color is what made the wrist table a flat tone.
    if wrist_bgr is not None and wrist_bgr.shape[0] > 180 and wrist_bgr.shape[1] > 560:
        crop = wrist_bgr[10:170, 40:520].copy()
        value_scale = 0.75
    else:
        crop = frame_bgr[400:700, 160:900].copy()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        objects = cv2.inRange(hsv, (45, 60, 40), (130, 255, 255))
        objects = cv2.dilate(objects, np.ones((5, 5), np.uint8))
        if int(objects.sum()) > 0:
            crop = cv2.inpaint(crop, objects, 4, cv2.INPAINT_TELEA)
        value_scale = 0.65
    patch = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    # The lights clip the raw table photo to yellow in the wrist camera.
    # The wrist crop's grain is raised around its mean so the lines are still
    # there after that lighting. A flat scale of the same crop washes them out.
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    if wrist_bgr is not None:
        mean = float(hsv[:, :, 2].mean())
        hsv[:, :, 2] = np.clip((hsv[:, :, 2] - mean) * 2.4 + mean * 0.75, 0, 255)
    else:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _stripe_column(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    region = hsv[:430, 900:1240]
    saturation = region[:, :, 1].mean(axis=0)
    column = int(np.argmax(saturation)) + 900
    strip = frame_bgr[40:430, column : column + 1]
    if strip.size == 0:
        strip = np.zeros((390, 1, 3), dtype=np.uint8)
        strip[:, :, 1] = 140
    tiled = np.repeat(strip, 48, axis=1)
    return cv2.resize(tiled, (128, 256), interpolation=cv2.INTER_NEAREST)


def write_textures(frame_bgr: np.ndarray, directory: Path, wrist_bgr: np.ndarray | None = None) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    wood = directory / "wood.png"
    stripe = directory / "stripe.png"
    cv2.imwrite(str(wood), _wood_patch(frame_bgr, wrist_bgr))
    cv2.imwrite(str(stripe), _stripe_column(frame_bgr))
    return {"wood": wood, "stripe": stripe}


def median_rgba(frame_bgr: np.ndarray, sat_max: int, value_min: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] < sat_max) & (hsv[:, :, 2] > value_min)
    if int(mask.sum()) < 100:
        bgr = np.array([140, 140, 140], dtype=np.float64)
    else:
        bgr = np.median(frame_bgr[mask], axis=0)
    return np.array([bgr[2], bgr[1], bgr[0], 1.0], dtype=np.float64) / np.array([255, 255, 255, 1])
