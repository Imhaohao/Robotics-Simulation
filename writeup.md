# Real2sim from one Franka episode

Task: drop the yellow block into the blue cup. Robot: Franka FR3, Robotiq 2F-85. The episode is 101 steps at about 14 Hz, with two fixed ZED cameras and a wrist camera. The grader used here is `trekion_real2sim_v1` (`python3 -m real2sim.evalgate`). Every number below is taken from `outputs/metrics.json` or `outputs/eval.json`.

## Approach

The Menagerie FR3 flange matches `cartesian_position` to solver noise, so the arm is the coordinate frame for anything it touches.

**Table.** StereoSGBM on the rectified ext1 pair, unprojected with the factory intrinsics. A seeded RANSAC plane, then a refit. ext1’s height agrees with the fingertips. ext2’s cloud is centimeters off, so it is not used for geometry.

**Block and cup.** The block’s xy is the pinch over the two steps where the measured gripper width first passes 0.5. Its half-width is the median stereo extent of the yellow blob, 15.9 mm, and it is kept under the empty-hand opening at the stall (finger gap 37.5 mm) so the pads can close on the faces. The cup’s xy is the pinch at its lowest point during the hold. The bottom radius is the stereo radius, 24.2 mm. The mouth is wide enough for the block’s corner at the pinch while the fingers are still shut. The rim is the lowest fingertip during that time. The pads reach 21.6 mm past the pinch, and the pad box itself is another 9.4 mm along that axis (`g_right_pad1`), so the rim sits one pad-depth below the fingertip. Both lengths are read off the gripper at the insertion pose.

**Teal cube.** Nothing grasps it. Its ext1 stereo centroid is shifted by the same xy correction that takes the stereo block onto the pinch.

**Gripper.** The logged target jumps toward closed while the hand is still above the block. Replaying it leaves the block 11.9 cm from the cup. The command follows the measured width, then goes to 0.95 once the width stalls, which is the squeeze. The open follows the measurement again.

**Physics.** Joint position actuators track the log. The block, cup, and cube are free bodies. The cup is a compound rigid so a block can fall in. No weld on the nominal scene. Contacts use an elliptic cone. Settle 0.4 s, then each logged step at 1 ms.

**Appearance.** The room in the rendered view is the real frame. The three objects are inpainted out of that frame, on the table only, so the striped wall is not painted over. The robot and the three bodies are the MuJoCo render, masked by the simulator’s segmentation. The lab is not a relightable mesh. That is stated here so the picture is not mistaken for one.

## Results

| Check | Value |
|---|---|
| Joint RMSE | 0.0130 rad |
| Flange RMSE | 1.26 cm |
| Block to cup axis, final | 1.3 mm, and inside the cup |
| Cup drift / tilt | ~0 / ~0 deg |
| Teal drift | 0 |
| Block and cup, frame 0, ext1 | 1.1 px, 2.2 px |
| Block and cup, frame 0, ext2 | 11.9 px, 9.2 px |
| Block and cup, grasp, ext1 | 24.4 px, 3.9 px |
| Block and cup, grasp, ext2 | 17.9 px, 7.7 px |

On ext1 at frame 0 the color overlap is 0.9919 for the block, 0.8510 for the cup, and 0.8053 for the cube. Unpainted lab pixels match the real frame (`background_match` 1.0). ext1’s teal *centroid* detector still locks onto the backdrop (559 px). The 3D cube projects to 35 px of the teal blob on ext2. An open gripper leaves the block on the table (z = −1.19 cm, the rest height). The nominal block peaks at 13.9 cm, so the gate is not “it ended in the volume” alone: it was lifted. Welding the block to the palm and opening the hand lifts it to z = 5.5 cm, and that run is rejected. A block twice as wide is still narrower than the 85 mm hand, and this run does lift it (peak z = 12.1 cm). It does not settle inside the cup, which is why that gate fails. It is not evidence that the fingers missed. A table 5 cm too high leaves the block 9.9 cm from the cup. The logged target, replayed raw, leaves it 11.9 cm away.

The side-by-side is real | sim for all three cameras and all 100 frames. The wrist view is the least convincing: a close camera makes the cup’s ring geometry and any centimeter of pose error fill the frame.

## Determinism and verifiability

RANSAC is seeded. Two full replays match to 0 m on the block, cup, and cube. `evalgate` exits non-zero unless every gate passes, including the three injected failures and the weld check. A bad scene is a failed process.

## Scaling

On this CPU, calibration plus physics is 3.1 s. Rendering three 1280×720 views for 100 frames, with the plate, is the rest of a run and is about two minutes. A thousand episodes is under an hour of physics and on the order of a day of rendering on one machine, and each episode is independent.

What breaks first is contact, and the three bad scenes are that break: an early gripper close, a block the fingers cannot swallow, and a table the hand no longer meets. Extrinsics are next. The two fixed cameras already disagree, and only the arm made the touched objects consistent. A human looks when a gate fails, or when the object is transparent or a color the masks have never seen. The thing to build first is the verifier, which this repo runs, and then a fixed grid over block scale and rim clearance that accepts the first value whose gates pass.

## Next

A cup mesh from the stereo hull, instead of rings, would fix the wrist view. A static backdrop fused from depth would replace the real plate on scenes where the camera moves. Torque tracking, instead of stiff position actuators, is the physics step if the next episode is one the log has not already played.
