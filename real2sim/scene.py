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


def _quat_z(angle: float) -> list[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def _cup_shell(scene: SceneSpec) -> tuple[list[float], list[int]]:
    """An open frustum. Collision stays on the hidden boxes; this is what the cameras see."""
    count = 48
    thickness = 0.0016
    z0 = -scene.cup_height / 2.0
    z1 = scene.cup_height / 2.0
    vertices: list[float] = []
    # Outer bottom, outer top, inner bottom, inner top.
    for radius, z in (
        (scene.cup_bottom_radius, z0),
        (scene.cup_top_radius, z1),
        (max(0.004, scene.cup_bottom_radius - thickness), z0),
        (max(0.004, scene.cup_top_radius - thickness), z1),
    ):
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            vertices.extend((radius * math.cos(angle), radius * math.sin(angle), z))
    faces: list[int] = []

    def quad(a0: int, b0: int) -> None:
        for index in range(count):
            nxt = (index + 1) % count
            a, b = a0 + index, a0 + nxt
            c, d = b0 + index, b0 + nxt
            faces.extend((a, d, b, a, c, d))

    quad(0, count)  # outer wall
    quad(3 * count, 2 * count)  # inner wall, facing in
    quad(count, 3 * count)  # rim
    return vertices, faces


def _add_cup(spec: mujoco.MjSpec, scene: SceneSpec) -> None:
    cup = spec.worldbody.add_body(name="cup", pos=list(scene.cup_pos))
    cup.add_freejoint(name="cup_free")
    bottom = cup.add_geom(
        name="cup_bottom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[scene.cup_bottom_radius, 0.0025, 0.0],
        pos=[0.0, 0.0, -scene.cup_height / 2.0 + 0.0025],
        mass=0.22,
        rgba=list(scene.cup_rgba),
    )
    bottom.friction = [1.4, 0.02, 0.001]
    shell = spec.add_mesh()
    shell.name = "cup_shell"
    vertices, faces = _cup_shell(scene)
    shell.uservert = vertices
    shell.userface = faces
    visible = cup.add_geom(
        name="cup_shell",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cup_shell",
        rgba=list(scene.cup_rgba),
        mass=0.0,
        contype=0,
        conaffinity=0,
    )
    del visible
    # Contact is a ring of boxes. They are not drawn. A convex mesh would fill the cup.
    rings = 3
    for ring in range(rings):
        t0 = ring / rings
        t1 = (ring + 1) / rings
        radius = scene.cup_bottom_radius + (scene.cup_top_radius - scene.cup_bottom_radius) * (t0 + t1) / 2.0
        center_z = -scene.cup_height / 2.0 + scene.cup_height * (t0 + t1) / 2.0
        half_z = scene.cup_height / rings * 0.62
        count = max(16, int(2.0 * math.pi * radius / 0.012))
        for index in range(count):
            angle = 2.0 * math.pi * (index + 0.5) / count
            wall = cup.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.0018, math.pi * radius / count * 1.05, half_z],
                pos=[radius * math.cos(angle), radius * math.sin(angle), center_z],
                quat=_quat_z(angle),
                mass=0.0015,
                rgba=[0.0, 0.0, 0.0, 0.0],
            )
            wall.friction = [0.35, 0.005, 0.0001]


def build_model(scene: SceneSpec, assets: Path, weld_block: bool = False) -> mujoco.MjModel:
    spec = robot_spec(assets)
    spec.option.timestep = 0.001
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.headlight.ambient = [0.45, 0.44, 0.42]
    spec.visual.headlight.diffuse = [0.55, 0.53, 0.48]
    spec.visual.headlight.specular = [0.08, 0.08, 0.08]
    spec.worldbody.add_light(
        pos=[0.35, -0.15, 1.7],
        dir=[0.15, 0.25, -1.0],
        diffuse=[0.65, 0.62, 0.55],
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
            material.texuniform = True
            material.texrepeat = [1.5, 1.5] if name == "wood" else [1.0, 1.0]
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
    yellow = block.add_geom(
        name="yellow_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[scene.block_half, scene.block_half, scene.block_half],
        mass=0.035,
        rgba=list(scene.block_rgba),
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
    return spec.compile()
