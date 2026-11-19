"""Assemble the MuJoCo scene from a calibrated description of the episode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from real2sim.kinematics import robot_spec


@dataclass(frozen=True)
class SceneSpec:
    table_z: float
    table_center: tuple[float, float]
    table_half: tuple[float, float]
    block_pos: tuple[float, float, float]
    block_half: float
    block_rgba: tuple[float, float, float, float]
    cup_pos: tuple[float, float, float]
    cup_height: float
    cup_bottom_radius: float
    cup_top_radius: float
    cup_rgba: tuple[float, float, float, float]
    teal_pos: tuple[float, float, float]
    teal_half: float
    teal_rgba: tuple[float, float, float, float]
    wall_rgba: tuple[float, float, float, float]
    texture_dir: str | None


def _cup_skin(scene: SceneSpec) -> tuple[list[float], list[int]]:
    """The outer wall only. An inner wall 1.6 mm in z-fights and reads as ribs."""
    count = 128
    z0 = -scene.cup_height / 2.0
    z1 = scene.cup_height / 2.0
    vertices: list[float] = []
    for radius, z in ((scene.cup_bottom_radius, z0), (scene.cup_top_radius, z1)):
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            vertices.extend((radius * math.cos(angle), radius * math.sin(angle), z))
    faces: list[int] = []
    for index in range(count):
        nxt = (index + 1) % count
        a, b = index, nxt
        c, d = count + index, count + nxt
        faces.extend((a, d, b, a, c, d))
    return vertices, faces


def _add_cup(spec: mujoco.MjSpec, scene: SceneSpec) -> None:
    """A hollow shell. One mesh would be replaced by its convex hull and the block could not fall in.

    The capsules are the wall the block hits. They are the stereo radii. A lit stack
    of capsules reads as ribs, so the cameras see one smooth skin of those same radii.
    The skin does not collide. Boxes at the same radii left corners the block could sit on.
    """
    cup = spec.worldbody.add_body(name="cup", pos=list(scene.cup_pos))
    cup.add_freejoint(name="cup_free")
    bottom = cup.add_geom(
        name="cup_bottom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[min(scene.cup_bottom_radius, scene.cup_top_radius) - 0.0016, 0.0025, 0.0],
        pos=[0.0, 0.0, -scene.cup_height / 2.0 + 0.0025],
        mass=0.22,
        rgba=list(scene.cup_rgba),
    )
    bottom.friction = [1.4, 0.02, 0.001]
    # Vertical capsules, two bands, so the wall is the hull and does not show the
    # horizontal ribs of a stack of boxes. A capsule is convex. The ring is hollow.
    count = 64
    for band, radius in ((0, scene.cup_bottom_radius), (1, scene.cup_top_radius)):
        center_z = -scene.cup_height / 4.0 if band == 0 else scene.cup_height / 4.0
        for index in range(count):
            angle = 2.0 * math.pi * (index + 0.5) / count
            wall = cup.add_geom(
                name=f"cup_wall_{band}_{index}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[0.0022, scene.cup_height / 4.0],
                pos=[radius * math.cos(angle), radius * math.sin(angle), center_z],
                mass=0.0008,
                rgba=list(scene.cup_rgba),
            )
            wall.friction = [0.35, 0.005, 0.0001]
            wall.group = 3
    skin = spec.add_mesh()
    skin.name = "cup_skin"
    vertices, faces = _cup_skin(scene)
    skin.uservert = vertices
    skin.userface = faces
    skin.smoothnormal = True
    drawn = cup.add_geom(
        name="cup_skin",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cup_skin",
        rgba=list(scene.cup_rgba),
        mass=0.0,
        contype=0,
        conaffinity=0,
        group=0,
    )
    del drawn


def build_model(scene: SceneSpec, assets: Path, weld_block: bool = False) -> mujoco.MjModel:
    spec = robot_spec(assets)
    spec.option.timestep = 0.001
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.headlight.ambient = [0.72, 0.66, 0.55]
    spec.visual.headlight.diffuse = [0.55, 0.53, 0.48]
    spec.visual.headlight.specular = [0.08, 0.08, 0.08]
    spec.worldbody.add_light(
        pos=[0.35, -0.15, 1.7],
        dir=[0.15, 0.25, -1.0],
        diffuse=[0.16, 0.16, 0.14],
        ambient=[0.18, 0.18, 0.17],
        cutoff=80,
    )

    if scene.texture_dir:
        rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
        for name, filename in (("wood", "wood.png"), ("stripe", "stripe.png")):
            texture = spec.add_texture()
            texture.name = name
            texture.type = int(mujoco.mjtTexture.mjTEXTURE_2D)
            texture.file = str(Path(scene.texture_dir, filename).resolve())
            material = spec.add_material()
            material.name = f"{name}_mat"
            # The wood photo is one patch of the real table, shown once. Averaging
            # it (texuniform) is the flat tone. The stripe stays averaged.
            material.texuniform = name != "wood"
            material.texrepeat = [1.2, 1.2] if name == "wood" else [1.0, 1.0]
            # The texture is the color. A tint here would stain the photograph.
            material.rgba = [1.0, 1.0, 1.0, 1.0]
            slots = [""] * 10
            slots[rgb] = name
            material.textures = slots

    table = spec.worldbody.add_body(
        name="table_body",
        pos=[scene.table_center[0], scene.table_center[1], scene.table_z - 0.02],
    )
    top = table.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[scene.table_half[0], scene.table_half[1], 0.02],
        rgba=[0.72, 0.52, 0.30, 1.0],
    )
    top.friction = [1.15, 0.005, 0.0001]
    if scene.texture_dir:
        top.material = "wood_mat"
        top.rgba = [1.0, 1.0, 1.0, 1.0]

    mount = spec.worldbody.add_body(name="mount", pos=[0.0, 0.0, scene.table_z / 2.0])
    puck = mount.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.09, abs(scene.table_z) / 2.0, 0.0],
        rgba=[0.12, 0.12, 0.12, 1.0],
        contype=0,
        conaffinity=0,
    )
    del puck

    wall = spec.worldbody.add_body(name="gray_wall", pos=[0.55, 0.85, 0.55])
    wall.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.9, 0.02, 0.7],
        rgba=list(scene.wall_rgba),
        contype=0,
        conaffinity=0,
    )
    stripes = spec.worldbody.add_body(name="stripes", pos=[1.15, 0.05, 0.6])
    stripe_geom = stripes.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.55, 0.7],
        rgba=[0.4, 0.55, 0.7, 1.0],
        contype=0,
        conaffinity=0,
    )
    if scene.texture_dir:
        stripe_geom.material = "stripe_mat"
        stripe_geom.rgba = [1.0, 1.0, 1.0, 1.0]

    block = spec.worldbody.add_body(name="yellow", pos=list(scene.block_pos))
    block.add_freejoint(name="yellow_free")
    # The real block is a translucent prism. Alpha does not enter the contact.
    block_rgba = list(scene.block_rgba)
    block_rgba[3] = 0.45
    yellow = block.add_geom(
        name="yellow_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[scene.block_half, scene.block_half, scene.block_half],
        mass=0.035,
        rgba=block_rgba,
        condim=6,
    )
    yellow.friction = [2.5, 0.1, 0.01]
    yellow.solref = [0.01, 1.0]
    yellow.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]

    _add_cup(spec, scene)

    teal = spec.worldbody.add_body(name="teal", pos=list(scene.teal_pos))
    teal.add_freejoint(name="teal_free")
    cube = teal.add_geom(
        name="teal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[scene.teal_half, scene.teal_half, scene.teal_half],
        mass=0.04,
        rgba=list(scene.teal_rgba),
    )
    cube.friction = [0.9, 0.01, 0.0001]

    for name in ("ext1", "ext2", "wrist"):
        spec.worldbody.add_camera(name=name, fovy=50, pos=[0.3, 0.0, 0.4])
    if weld_block:
        # A block glued to the gripper must fail the open-hand check.
        weld = spec.add_equality()
        weld.name = "block_weld"
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1 = "yellow"
        weld.name2 = "g_base"
        # data[1] is torquescale. At 1, that value is added to the weld's y
        # offset, so the glue aims a meter beside the hand and never holds.
        weld.data[1] = 0.0
    return spec.compile()
