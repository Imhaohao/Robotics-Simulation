import os
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from real2sim.episode import load_episode, read_video
from real2sim.kinematics import grasp_release_indices, gripper_commands
from real2sim.metrics import project
from real2sim.pipeline import calibrate, run
from real2sim.replay import rollout
from real2sim.scene import build_model
from real2sim.textures import write_textures


class Real2SimTests(unittest.TestCase):
    def test_physics_puts_the_block_in_the_cup(self):
        root = Path(__file__).resolve().parents[1]
        payload = run(
            root / "data" / "episode",
            root / "third_party" / "mujoco_menagerie",
            root / "outputs" / "test",
            render=False,
        )
        checks = payload["checks"]
        self.assertTrue(checks["arm_tracks_log"], payload["numbers"])
        self.assertTrue(checks["block_ended_in_cup"], payload["numbers"])
        self.assertTrue(checks["teal_cube_stayed"], payload["numbers"])
        self.assertTrue(checks["cup_upright"], payload["numbers"])
        self.assertTrue(checks["cup_stayed"], payload["numbers"])
        self.assertTrue(checks["block_projects_near_real"], payload["numbers"])
        self.assertTrue(payload["passed"], payload)
        self.assertTrue(payload["deterministic_calibration"])

    def test_wrist_render_uses_the_logged_pose(self):
        root = Path(__file__).resolve().parents[1]
        episode = load_episode(root / "data" / "episode")
        stereo = read_video(episode.cameras["ext1"].stereo_video)[0]
        texture_dir = root / "outputs" / "test" / "textures"
        write_textures(stereo[:, : stereo.shape[1] // 2], texture_dir)
        scene, _ = calibrate(episode, root / "third_party" / "mujoco_menagerie", stereo, texture_dir)
        model = build_model(scene, root / "third_party" / "mujoco_menagerie")
        result = rollout(model, episode, render_size=(180, 320), video_steps=1)
        frame = result.frames["wrist"][0]
        camera = episode.cameras["wrist"]
        posed = camera.__class__(**{**camera.__dict__, "extrinsics": episode.wrist_extrinsics[0]})
        pixel = project(posed, np.array(scene.cup_pos))
        column = int(np.clip(pixel[0] * frame.shape[1] / camera.width, 0, frame.shape[1] - 1))
        row = int(np.clip(pixel[1] * frame.shape[0] / camera.height, 0, frame.shape[0] - 1))
        window = frame[max(0, row - 3) : row + 4, max(0, column - 3) : column + 4]
        blue = window[:, :, 2].astype(np.int16)
        red = window[:, :, 0].astype(np.int16)
        self.assertGreater(int((blue - red).max()), 20, (column, row, window.mean(axis=(0, 1))))

    def test_logged_target_closes_before_the_fingers_do(self):
        root = Path(__file__).resolve().parents[1]
        episode = load_episode(root / "data" / "episode")
        grasp, _ = grasp_release_indices(episode.gripper_position)
        earlier = grasp - 5
        self.assertGreater(float(episode.gripper_target[earlier]), float(episode.gripper_position[earlier]) + 0.2)
        stall = gripper_commands(episode.gripper_position)
        self.assertLess(float(stall[earlier]), float(episode.gripper_target[earlier]))


if __name__ == "__main__":
    unittest.main()
