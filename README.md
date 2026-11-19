# Robotics simulation

Real2sim of one Franka FR3 episode: drop a yellow block into a blue cup. The pipeline reads the raw episode (three ZED cameras, `trajectory.h5`, calibration), builds a MuJoCo scene, replays the arm under physics, and checks itself.

The robot is the MuJoCo Menagerie Franka Research 3 with a Robotiq 2F-85. The flange pose from that model matches the logged Cartesian pose to solver noise, so the arm is used as the measuring instrument for the objects it touches. The teal cube, which nothing grasps, comes from stereo.

## Requirements

- Linux, CPU. No GPU. Tested on Python 3.12.
- An OpenGL offscreen backend. `MUJOCO_GL=egl` with `libegl1` and Mesa, or `MUJOCO_GL=osmesa` with `libosmesa6`.
- Packages in `requirements.txt` (MuJoCo 4.0.1, NumPy, SciPy, OpenCV, h5py, imageio).

```bash
python3 -m pip install -r requirements.txt
bash scripts/fetch_assets.sh
MUJOCO_GL=egl python3 -m real2sim --data data/episode --assets third_party/mujoco_menagerie --out outputs
MUJOCO_GL=egl python3 -m real2sim.evalgate --data data/episode --assets third_party/mujoco_menagerie --out outputs
python3 -m real2sim.monitor --port 8765
python3 -m unittest tests.test_real2sim
```

The monitor serves `http://127.0.0.1:8765` and reads `outputs/progress.json` while the eval is running. The eval is `trekion_real2sim_v1`: the block has to end in the cup, a second run has to match, and three broken scenes (the raw gripper target, a block twice as wide, a table 5 cm too high) have to fail on their own. It also checks that an open gripper leaves the block on the table and that welding the block to the hand does not.

`scripts/fetch_assets.sh` clones MuJoCo Menagerie at commit `370c439` and checks out `franka_fr3` and `robotiq_2f85`. Those meshes stay under Apache 2.0 and the Robotiq BSD-style license in the upstream repo. They are not vendored here.

A full run is about one minute on CPU. Physics without rendering is a few seconds. `python3 -m real2sim` exits non-zero if the nominal checks in `outputs/metrics.json` fail. Those checks pass: the block ends in the cup. `evalgate` passes every gate. A headless run scores 86, because appearance is not rendered. The rendered frame adds 12 appearance points, and `outputs/eval.json` records 98. The numbers are in `writeup.md`.

## What it writes

| File | What it is |
|---|---|
| `outputs/side_by_side.mp4` | Real frame and simulated frame, ext1 / ext2 / wrist, 100 frames |
| `outputs/stills/` | The same comparison at steps 0, 40, 80 and 99 |
| `outputs/scene.json` | Table, object poses, sizes, colors |
| `outputs/metrics.json` | Tracking error, outcome checks, reprojection error |
| `outputs/eval.json` | The hiring eval: gates and the 100-point score |
| `outputs/progress.json` | The same score, updated while the eval runs |

## Layout

```
data/episode/          the recorded episode
real2sim/              calibrate, simulate, check
tests/test_real2sim.py physics outcome and a repeated calibration
writeup.md             approach, results, scaling
DECISIONS.md           why each choice was made, and which review changed it
```

Frame `i` of each video is trajectory step `i`. There are 100 frames and 101 steps. The control period in the log is about 71 ms.

See `writeup.md` for the method and the numbers.
