import os
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from real2sim.episode import load_episode
from real2sim.kinematics import grasp_release_indices, gripper_commands
from real2sim.pipeline import run


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
