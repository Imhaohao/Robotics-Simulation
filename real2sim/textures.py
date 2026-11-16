"""Appearance sampled from the real frame: wood tone, wall tone, stripe column."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _wood_patch(frame_bgr: np.ndarray) -> np.ndarray:
    # Near tabletop in the external view, with the cup and cube painted out.
    crop = frame_bgr[400:700, 160:900].copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    objects = cv2.inRange(hsv, (45, 60, 40), (130, 255, 255))
    objects = cv2.dilate(objects, np.ones((5, 5), np.uint8))
    if int(objects.sum()) > 0:
        crop = cv2.inpaint(crop, objects, 4, cv2.INPAINT_TELEA)
    return cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)


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


def write_textures(frame_bgr: np.ndarray, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    wood = directory / "wood.png"
    stripe = directory / "stripe.png"
    cv2.imwrite(str(wood), _wood_patch(frame_bgr))
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
