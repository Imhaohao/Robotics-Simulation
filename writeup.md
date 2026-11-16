# Real2sim from one Franka episode

Task: drop the yellow block into the blue cup. Robot: Franka FR3, Robotiq 2F-85. The episode is 101 steps at about 14 Hz, with two fixed ZED cameras and a wrist camera.

## Approach

The Menagerie FR3 is not an approximation here. With the logged joint positions, its flange site matches `observation/robot_state/cartesian_position` to numerical noise on every step I checked, including the rotation. That makes the arm a coordinate frame, not just a mesh.

**Table.** StereoSGBM on the rectified ext1 pair, unprojected with the factory intrinsics and the given extrinsics (`scipy` xyz Euler, OpenCV camera axes). A seeded RANSAC plane with a refit gives the table. ext1’s plane height agrees with the fingertip height at the grasp. ext2’s cloud is offset by a few centimeters, so ext1 is the geometric camera and the arm breaks the tie.

**Yellow block and blue cup.** These are the objects the hand touches, so their positions come from the pinch site (15.6 cm along the flange axis, measured on the assembled gripper), not from the noisier cloud. The block’s xy is the pinch over the two steps where the measured gripper width first passes 0.5. Its half-width is 0.44 times the inner pad gap at that stalled width, so the fingers close onto it instead of stopping in the air. The cup’s xy is the pinch at the lowest point of the hold, which is the insertion. The rim is kept 3.2 cm below that pinch so the departing hand clears it. The mouth is at least 5 cm in radius, wide enough to contain the release point. Both numbers are functions of the log, not hand-placed coordinates.

**Teal cube.** Nothing grasps it. Its stereo centroid on ext1 is shifted by the same xy correction that takes the stereo block onto the pinch. It then sits on the table.

**Gripper command.** The logged *target* jumps to nearly closed while the hand is still above the block. Replaying that target shuts the fingers early and knocks the block aside. The measured width times the real close. Once that width stalls above 0.5, the command goes to 0.95 so the pads produce a squeeze instead of hovering at the contact opening. The open follows the measurement again, which is when the block is released.

**Physics.** Joint position actuators track the log. The block, cup and cube are free bodies. The cup is a compound rigid (disk bottom, three rings of wall segments) so a block can fall in. Nothing is welded to the hand. Contacts use an elliptic cone. The arm settles for 0.4 s, then each logged step is integrated for its real timestamp delta at 1 ms.

**Appearance.** The table texture is a crop of the ext1 tabletop with the blue and teal pixels inpainted. Object colors are the median of the saturated blobs in that frame. A gray wall and a stripe panel, sampled from a column of the backdrop, stand in for the room. People in the lab are not modeled.

**What is automatic.** Stereo, plane fit, pinch anchoring, finger-gap sizing, color sampling, replay and the checks. The funnel clearance and the 0.95 squeeze are fixed rules in code, chosen because a literal replay of the target width does not grasp and a rim at the fingertip makes the hand shove the cup. They do not change per run.

## Results

From `outputs/metrics.json` on this episode:

| Check | Value |
|---|---|
| Joint RMSE | 0.013 rad |
| Flange position RMSE | 1.26 cm |
| Block distance to cup axis at the end | 7.7 mm, and its height is between the cup bottom and the rim |
| Cup translation / tilt | ~0 mm / ~0 deg |
| Teal cube translation | 0 |
| Block reprojection, ext1 / ext2 | 22 px / 39 px |
| Cup reprojection, ext1 / ext2 | 24 px / 39 px |
| Teal reprojection, ext2 | 35 px |

ext1’s teal *pixel* detector locks onto the striped backdrop (559 px). The 3D cube still projects to 35 px of the teal blob on ext2, which sees the cube cleanly. Color IoU of the yellow block on ext1 at frame 0 is 0.49. Whole-image IoU for the cup and cube stays low because the backdrop and the wood share those hues. The reprojection numbers are the visual check.

The side-by-side video shows the same story. At step 0 the three objects sit in the same arrangement in both fixed views. At step 40 the gripper is on the block. At step 80 the block is over the cup. At step 99 the block is no longer on the table and the cup and cube are where they started. That last state is the physics result: the block was a free body the whole time.

The background does not match the lab. The stripe panel reads as horizontal bands, the right-hand side of ext1 falls off to black, and the wrist close-up magnifies every centimeter of cup-shape error. The wood grain is a photograph, not a metrically aligned texture.

## Determinism and verifiability

RANSAC uses a fixed seed. MuJoCo is stepped with no injected noise. `run()` builds the scene twice and aborts if the block or cup pose differs. `tests/test_real2sim.py` replays the episode and requires the block to end in the cup, the cup and cube to stay put, the arm to track, and the block to project within 80 px. `python3 -m real2sim` writes `metrics.json` and exits non-zero when any of those checks fail. A bad scene is a failed process, not a video someone has to watch.

## Scaling

Per scene, on CPU: a few seconds to calibrate and simulate, about half a minute to render three 1280×720 views for 100 frames. At a thousand scenes that is hours of render, not days, and it parallelizes per episode. The failure that shows up first is contact, not geometry. A few millimeters of block width or rim height turns a grasp into a push. The second is extrinsics: the two fixed cameras already disagree by centimeters, and only the arm made them consistent. The third is appearance: a cropped photo does not survive a new viewpoint or a moved object.

What I would build first is the verifier, which this repo already runs, and then a per-scene search over the two contact parameters (block scale, rim clearance) that accepts the first value whose checks pass. That search stays deterministic if the grid is fixed. A human looks only at scenes the checks reject: transparent objects, a gripper stall that is not a grasp, or a camera whose reprojection is hundreds of pixels. Novel colors need a new mask, not a new pipeline.

## Next

With more time I would fit the cup as a mesh from the stereo hull instead of rings, align the table texture by projection onto the plane, and replace the stripe panel with a depth-fused backdrop. The wrist view is the one I would re-render first after that, because small errors fill the frame. Torque tracking, rather than stiff position actuators, would be the physics upgrade if the goal moved from “this episode” to “a nearby episode the policy has not logged.”
