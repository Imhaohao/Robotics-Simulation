# Real2sim from one Franka episode

Task: drop the yellow block into the blue cup. Robot: Franka FR3, Robotiq 2F-85. The episode is 101 steps at about 14 Hz, with two fixed ZED cameras and a wrist camera. The grader used here is `trekion_real2sim_v1` (`python3 -m real2sim.evalgate`). Every number below is taken from `outputs/metrics.json` or `outputs/eval.json`.

## Approach

The Menagerie FR3 flange matches `cartesian_position` to solver noise, so the arm is the coordinate frame for anything it touches.

**Table.** StereoSGBM on the rectified ext1 pair, unprojected with the factory intrinsics. A seeded RANSAC plane, then a refit. ext1’s height agrees with the fingertips. ext2’s cloud is centimeters off, so it is not used for geometry.

**Block and cup.** The block’s xy is the pinch over the two steps where the measured gripper width first passes 0.5. Its half-width is the median stereo extent of the yellow blob, 15.9 mm, and it is kept under the empty-hand opening at the stall (finger gap 37.5 mm) so the pads can close on the faces. The cup’s xy is the pinch at its lowest point during the hold. The stereo centroid of the cup is centimeters from that pinch, and the hand is what touches the cup, so the cloud is not used for the cup’s position. The height and both radii are the cloud. The rim is the mode of the blue points’ height, 81.3 mm. Each radius is a circle fit to that band: 27.8 mm at the base and 30.9 mm at the rim. A 95th-percentile rim is the sparse tail above the mode, and it put the wall through the pads. The wall the block falls into is this hull. One colliding mesh would be replaced by its convex hull and would fill the cup, so the contact is a ring of capsules at those two radii. Boxes at the same radii left corners, and the block settled 9.9 mm off the axis. The capsules settle it 2.1 mm off the axis. A lit stack of capsules reads as ribs, so the cameras see one smooth skin of the same radii. The skin does not collide.

**Teal cube.** Nothing grasps it. Its ext1 stereo centroid is shifted by the same xy correction that takes the stereo block onto the pinch.

**Gripper.** The logged target jumps toward closed while the hand is still above the block. Replaying it leaves the block 11.9 cm from the cup. The command follows the measured width, then goes to 0.95 once the width stalls, which is the squeeze. The open follows the measurement again.

**Physics.** Joint position actuators track the log. The block, cup, and cube are free bodies. The cup is a compound rigid so a block can fall in. No weld on the nominal scene. Contacts use an elliptic cone. Settle 0.4 s, then each logged step at 1 ms.

**Appearance.** The room in the rendered view is the real frame. The three objects are inpainted out of that frame, on the table only, so the striped wall is not painted over. The robot and the three bodies are the MuJoCo render, masked by the simulator’s segmentation. The lab is not a relightable mesh. That is stated here so the picture is not mistaken for one.

## Results

| Check | Value |
|---|---|
| Joint RMSE | 0.0130 rad |
| Flange RMSE | 1.26 cm |
| Block to cup axis, final | 2.1 mm, and inside the cup |
| Cup drift / tilt | 1.3 mm / ~0 deg |
| Teal drift | 0 |
| Block and cup, frame 0, ext1 | 1.1 px, 2.7 px |
| Block and cup, frame 0, ext2 | 11.9 px, 7.7 px |
| Block and cup, frame 0, wrist | 0.6 px, 0.2 px |
| Block and cup, grasp, ext1 | 24.4 px, 2.5 px |
| Block and cup, grasp, ext2 | 17.9 px, 6.6 px |
| Block and cup, grasp, wrist | 29.7 px, 0.6 px |

On ext1 at frame 0 the color overlap is 0.9918 for the block, 0.8498 for the cup, and 0.8068 for the cube. Unpainted lab pixels match the real frame (`background_match` 1.0). ext1’s teal *centroid* detector still locks onto the backdrop (559 px). The 3D cube projects to 35 px of the teal blob on ext2. An open gripper leaves the block on the table (z = −1.19 cm, the rest height). The nominal block peaks at 13.9 cm, so the gate is not “it ended in the volume” alone: it was lifted. Welding the block to the palm and opening the hand lifts it to z = 5.8 cm, and that run is rejected. A block twice as wide is still narrower than the 85 mm hand, and this run does lift it (peak z = 12.1 cm). It ends 31.0 mm from the axis, outside the 27.8 mm base, which is why that gate fails. It was picked up. A table 5 cm too high leaves the block 9.8 cm from the cup. The logged target, replayed raw, leaves it 11.9 cm away.

The side-by-side is real | sim for all three cameras and all 100 frames. The wrist camera uses the logged extrinsics, and the pose is forwarded before the frame is drawn. The wrist picture draws the arm at the logged joints, so the gripper sits on the real hand, and paints the real fingers that stick past the mesh the color of that gripper. Those joints are put back before the next physics step. The contact is still the lagged replay, and the flange error below is that replay. At the grasp the cup projection lands on the cup. That face is in shadow, hue 72 and saturation 35, so the fixed-camera mask is empty there. The distance in the gate is 0.6 px, to that face. The 54 px figure is the distance to a more saturated rim of the same cup, which the stricter mask was selecting. The wider range covers 0.57% of that frame, all of it on the cup, and a 60 px shift of the projection fails the gate.

## Determinism and verifiability

RANSAC is seeded. Two full replays match to 0 m on the block, cup, and cube. `evalgate` exits non-zero unless every gate passes, including the three injected failures and the weld check. A bad scene is a failed process.

## Scaling

On this CPU, calibration plus physics is 3.1 s. Rendering three 1280×720 views for 100 frames, with the plate, is the rest of a run and is about two minutes. A thousand episodes is under an hour of physics and on the order of a day of rendering on one machine, and each episode is independent.

What breaks first is contact, and the three bad scenes are that break: an early gripper close, a block the fingers cannot swallow, and a table the hand no longer meets. Extrinsics are next. The two fixed cameras already disagree, and only the arm made the touched objects consistent. A human looks when a gate fails, or when the object is transparent or a color the masks have never seen. The thing to build first is the verifier, which this repo runs.

## Next

The cup the block falls into is the stereo hull. The contact is capsules, because a single colliding mesh would be replaced by its convex hull. The cameras see a smooth skin of the same radii. A static backdrop fused from depth would replace the real plate on scenes where the camera moves. Torque tracking, instead of stiff position actuators, is the physics step if the next episode is one the log has not already played. The flange during replay is 1.26 cm off the log. The Menagerie wrist torque limit is 12 Nm, and that servo does not reach the logged joints in one 70 ms step. Prescribing those joints removes the error and puts the open fingertips through the block, which is then knocked aside. The object positions come from forward kinematics of the log, which matches `cartesian_position`. The lagged replay is not that ruler.
