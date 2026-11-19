# Real2sim from one Franka episode

Task: drop the yellow block into the blue cup. Robot: Franka FR3, Robotiq 2F-85. The episode is 101 steps at about 14 Hz, with two fixed ZED cameras and a wrist camera. The grader used here is `trekion_real2sim_v1` (`python3 -m real2sim.evalgate`). Every number below is taken from `outputs/metrics.json` or `outputs/eval.json`.

## Approach

The Menagerie FR3 flange matches `cartesian_position` to solver noise, so the arm is the coordinate frame for anything it touches.

**Table.** StereoSGBM on the rectified ext1 pair, unprojected with the factory intrinsics. A seeded RANSAC plane, then a refit. ext1’s height agrees with the fingertips. ext2’s cloud is centimeters off, so it is not used for geometry.

**Block and cup.** The block’s xy is the pinch over the two steps where the measured gripper width first passes 0.5. Its half-width is the median stereo extent of the yellow blob, 15.9 mm, and it is kept under the empty-hand opening at the stall (finger gap 37.5 mm) so the pads can close on the faces. The cup’s xy is the pinch at its lowest point during the hold. The stereo centroid of the cup is centimeters from that pinch, and the hand is what touches the cup, so the cloud is not used for the cup’s position. The height and both radii are the cloud. The rim is the mode of the blue points’ height, 81.3 mm. Each radius is a circle fit to that band: 27.8 mm at the base and 30.9 mm at the rim. A 95th-percentile rim is the sparse tail above the mode, and it put the wall through the pads. The wall the block falls into is this hull. One colliding mesh would be replaced by its convex hull and would fill the cup, so the contact is a ring of capsules at those two radii. Boxes at the same radii left corners, and the block settled 9.9 mm off the axis. With the hand yawed 88 degrees the capsules settle it 0.22 mm off the axis. A lit stack of capsules reads as ribs, so the cameras see one smooth skin of the same radii. The skin does not collide.

**Teal cube.** Nothing grasps it. Its ext1 stereo centroid is shifted by the same xy correction that takes the stereo block onto the pinch.

**Gripper.** The logged target jumps toward closed while the hand is still above the block. Replaying it leaves the block 16.3 cm from the cup. The command follows the measured width, then goes to 0.95 once the width stalls, which is the squeeze. The open follows the measurement again.

**Physics.** Joint position actuators track the log. The block, cup, and cube are free bodies. The cup is a compound rigid so a block can fall in. No weld on the nominal scene. Contacts use an elliptic cone. Settle 0.4 s, then each logged step at 1 ms.

**Hand and wrist camera.** The Menagerie 2F-85, mounted as shipped, is drawn from the side of the wrist frame. The real fingers enter from the bottom. The hand is yawed 88 degrees about the flange axis. Ninety degrees matches the picture and leaves the block on the rim when the log ends; 88 degrees keeps the fingertips around the block and finishes the drop. The pinch stays on the flange axis. Before the fingers close, the wrist panel uses the logged arm, because the lagged flange puts the block down inside the jaw. The real block at that frame is a translucent prism just past the fingertips. The colliding box is centered on the left edge of that prism, and moving it 2.2 cm onto the prism leaves it on the rim (z = 6.4 cm, 10 mm off the axis), so the contact point stays where the drop finishes. The drawing is a second box with no collision, 18 mm toward the prism, until the fingers close. It eases back onto the colliding box over the grasp, and the drop draws the box that is actually in the cup. A constant 18 mm offset leaves the drawing on the rim. Once the hand is holding the block, the camera rides that flange, so the panel is the hand that has the block. The block is drawn at alpha 0.45 with no specular. The drawing is a hexagonal prism with the same flat-to-flat width as the cube, because that is the silhouette in the wrist frame. The contact stays the cube. The cup skin is matte. The headlight has no diffuse, because a camera light turns the curve into a highlight, and an overhead light supplies the shading and the contact shadow on the wrist table. Its blue is lighter than the stereo sample, which is the shadowed face and was rendering as a cyan highlight. Copying the sim shadow onto the fixed-camera photograph painted a blob across the table, so those panels stay the photograph. The fixed cameras are still the real image with the robot and the three objects rendered on top. The wrist panel is the simulation only. The wrist table is one crop of the real wood from the top of the approach frame. The grain in that crop is raised so the lines survive the lights, and it is shown once across the table. It is that photograph. The specks are not in it. The tape is a separate pale rectangle, unprojected from the drop frame onto the table, and it does not collide. A tiled photograph and a periodic grain both read as stripes, so neither is used. The soft band across the top of the drop frame is the lighting on that crop, not another object.

## Results

| Check | Value |
|---|---|
| Joint RMSE | 0.0130 rad |
| Flange RMSE | 1.26 cm |
| Block to cup axis, final | 0.22 mm, inside the cup, center z 1.1 cm |
| Cup drift / tilt | 1.1 mm / ~0 deg |
| Teal drift | 0 |
| Block and cup, frame 0, ext1 | 0.9 px, 0.8 px |
| Block and cup, frame 0, ext2 | 10.4 px, 6.0 px |
| Block and cup, frame 0, wrist | 0.6 px, 0.2 px |
| Block and cup, grasp, ext1 | 0.3 px, 0.8 px |
| Block and cup, grasp, ext2 | 17.5 px, 6.0 px |
| Block and cup, grasp, wrist | 0.6 px, 0.6 px |

On ext1 at frame 0 the color overlap is 0.9870 for the block, 0.8200 for the cup, and 0.6506 for the cube. Unpainted lab pixels match the real frame (`background_match` 1.0). ext1’s teal centroid detector still locks onto the backdrop (532 px). The wrist approach shows the real cube as a few pixels in the corner, so the cube is slid along the table until that camera shows a corner too. That slide moves the ext2 projection from 35 px to 79 px: the two cameras do not agree, and the wrist is the one the slide follows. The shaded face of the real cube is inpainted when that blob stays small and compact, so ext2 shows one cube rather than the real one beside the render. The overlap numbers are lower than before that inpaint. An open gripper leaves the block on the table (z = −1.19 cm). The nominal block peaks at 13.9 cm. Replaying the logged gripper target leaves the block 16.3 cm from the cup. A block twice as wide is not lifted (peak z = 1.8 cm) and ends 12.0 cm from the axis. A table 5 cm too high leaves it 15.3 cm away.

The side-by-side is real | sim for the two fixed cameras, and the simulated wrist alone, for all 100 frames. Every frame-0 and grasp reprojection of the block and the cup, on all three cameras, is under 40 px. The largest is the grasp on ext2, 17.5 px. On the grasp wrist the projection sits on the darker face of the block. The bright highlight on that same block is 44 px away, so the distance is the closer of the two object masks. A mask that covers the table is not used, and the 40 px bar was not moved. An open gripper leaves the block on the table. Welding the block to the hand once the hand has reached it, and then holding the gripper open, carries the block to z = 13.6 cm, so that run is rejected. `python3 -m unittest tests.test_real2sim` passes. `python3 -m real2sim.evalgate` passes every gate. Without rendering, appearance scores 0 and the total is 86. The rendered frame adds the other 12 appearance points (block overlap 0.9870, cup overlap 0.8200, unpainted lab pixels matching the real frame), and `outputs/eval.json` records 98.

## Determinism and verifiability

RANSAC is seeded. Two full replays match to 0 m on the block, cup, and cube. The three injected failures still miss the cup. The open hand leaves the block on the table, and the welded hand carries it.

## Scaling

On this CPU, calibration plus physics is 2.9 s. Rendering three 1280×720 views for 100 frames is the rest of a run, about one minute. A thousand episodes is under an hour of physics and on the order of a day of rendering on one machine, and each episode is independent.

What breaks first is contact, and the three bad scenes are that break: an early gripper close, a block the fingers cannot swallow, and a table the hand no longer meets. Extrinsics are next. The two fixed cameras already disagree, and only the arm made the touched objects consistent. A human looks when a gate fails, or when the object is transparent or a color the masks have never seen. The thing to build first is the verifier, which this repo runs.

## Next

The cup the block falls into is the stereo hull. The contact is capsules, because a single colliding mesh would be replaced by its convex hull. The cameras see a 512-sided skin of the same radii. A 128-sided wall read as a jagged rim. A static backdrop fused from depth would replace the real plate on scenes where the camera moves. Torque tracking, instead of stiff position actuators, is the physics step if the next episode is one the log has not already played. The flange during replay is 1.26 cm off the log. The Menagerie wrist torque limit is 12 Nm, and that servo does not reach the logged joints in one 70 ms step. Prescribing those joints removes the error and puts the open fingertips through the block, which is then knocked aside. The object positions come from forward kinematics of the log, which matches `cartesian_position`. The lagged replay is not that ruler.
