# Decision log

Choices for the Trekion simulation take-home, and why each one was made. Later entries override earlier ones when a review showed the earlier choice was doing hidden work.

Times are America/Los_Angeles.

## Track and stack

**Simulation, not the residual-policy notebook.** The email says pick one track. This repository is the simulation repo, and the episode was already on the branch. The robot-learning notebook is a different assignment and is not in this submission.

**MuJoCo 4.0.1 on CPU, Menagerie FR3 plus Robotiq 2F-85, pinned at menagerie `370c439`.** The machine has no GPU, so Isaac Sim was a poor fit. The Menagerie flange reproduces the logged Cartesian pose to solver noise, which is the reason the arm can be used as a measuring instrument. Meshes stay in the upstream repos and are fetched by `scripts/fetch_assets.sh`, because vendoring 60 MB of licensed meshes into git does not help the pipeline.

## Where each number comes from

**Table height from ext1 stereo, not from ext2.** Both clouds are a plane, but they disagree by centimeters. ext1’s plane height agrees with the fingertip height at the grasp. ext2 is kept for reprojection checks only.

**Block and cup position from the pinch, not from the stereo centroid.** These are the objects the hand touches. The flange is exact and the clouds are not. The block’s xy is the pinch on the two steps where the measured gripper width first crosses 0.5. Averaging a wider window was rejected: the opening pass over the cup uses the same widths and pulled the block halfway to the cup.

**Block size from the stereo blob, rejected if it is wider than the finger opening.** The empty-hand gap at the stalled command is 37.5 mm. The yellow blob is 31.8 mm across. Using the full gap as the block width made the fingers hit the corners and shove the block (11 cm from the cup). The stereo size grasps. A hand-tuned 0.44 factor on the gap was an earlier version of the same idea and was removed once the blob was measured.

**Do not replay `action/target_gripper_position`.** The target jumps toward closed while the hand is still above the block. Replaying it leaves the block 11.9 cm from the cup, and that run is a required failure of the eval. The command that ships follows the measured width, then goes to 0.95 once the width stalls above 0.5. During the hold the log’s own target is already about 0.95, so 0.95 is the saturated close the controller asked for, applied only after the fingers have arrived. Following the measured width alone does not squeeze, and the block slips.

**Cup size is the stereo hull, not the hand.** The position stays on the pinch, because the stereo centroid is about 4.7 cm from the hand and the hand is what touches the cup. The height is the median of the mode of the blue points’ height, 81.3 mm. The mode bin holds 92 points. The 95th percentile, 85.5 mm, is the tail above that mode: it put the wall through the pads and the cup moved 13 mm, so it is not the rim. Each radius is a circle fit. A 90th-percentile distance from the median of a partial arc is larger than the circle, and that is what had produced a 39.9 mm mouth. The fits are 27.8 mm at the base and 30.9 mm at the rim. The block ends 2.1 mm from the axis. The cup moves 1.3 mm and does not tilt.

**The hull is the contact.** A single colliding mesh is replaced by its convex hull, which would fill the cup. The wall is a ring of capsules at the two hull radii. Boxes at those same radii were the contact first. Their corners left the block 9.9 mm off the axis. The capsules, at the same radii and the same height, settle it 2.1 mm off the axis, and the cup tilt goes to about 0. A lit stack of those capsules reads as ribs, so the cameras see one smooth skin of the same radii. The skin has no contact, and the capsules are in a render group the camera does not draw. An earlier scene drew a frustum of a different size and hid the boxes. That was a second shape. This skin is the hull the capsules already are.

**The skin is the outer wall only (2026-11-16, 5:07pm PT).** The first skin also drew an inner wall 1.6 mm inside the outer one. On the wrist camera those two surfaces z-fight, and the cup looks ribbed with a seam across the middle. Dropping the inner wall leaves a smooth outside. The opening shows the table and the cup floor, which is the cylinder that was already there.

**Painting the real hand out of the wrist plate, a third time (2026-11-16, 5:07pm PT).** The fingers are not gray. On the grasp frame their value is about 30 and their saturation is about 250, hue about 9, which is why a low-saturation mask missed them and ate the cup shadow instead. A mask of that dark red, closed over the screw holes and filled with Navier-Stokes, still leaves the finger silhouette as a brown smear. Same result as the earlier smear and the striped wood copy. The real hand stays in the plate.

**Replacing those fingers with the simulated table (2026-11-16, 5:20pm PT).** The same dark-red mask, with the hole filled from the wrist camera’s own render instead of an inpaint. The table texture’s mean color is about (200, 152, 70), and the lights clip the wrist view of it to about (255, 255, 110), so the hole was a bright yellow copy of the fingers. Shifting that color to the wood just outside the hand turned the yellow into a flat brown patch that is still the shape of the fingers. A full simulated wrist, with no real plate, is the same clipped table. Not shipped. The real hand stays.

**The fingertip-plus-pad rim is retired.** The pads reach 21.6 mm past the pinch and the pad box adds 9.4 mm of support. That sum was the clearance that kept the departing hand off a shorter cup. It is a fact about the gripper. It is not the cup’s height, and the geometry score no longer awards points for matching it.

**Teal cube from ext1 stereo plus the block’s xy bias.** Nothing grasps it, so the arm cannot locate it. The bias is the offset that takes the stereo block onto the pinch. The cube then sits on the table and does not move.

**Objects are free bodies.** No weld and no mocap on the nominal scene. An open gripper leaves the block at rest height. A weld of the block to the palm, with the hand held open, lifts the block, and that run is rejected. The nominal block also has to rise (it peaks at 13.9 cm). Ending inside the cup is not, by itself, proof of a grasp.

## What the eval is allowed to claim

**`trekion_real2sim_v1` is the grader, because it was accepted before the iteration loops.** Realism 35, pipeline 35, scaling 30, pass at 85 with every hard gate true. The gates include the three injected failures, the double run, and reprojection under 40 px on both fixed cameras at frame 0 and at the grasp.

**Reprojection is distance to the nearest object-colored pixel, not the centroid of the largest blob.** The ext1 teal centroid locks onto the striped backdrop (559 px) even when the cube is in the right place. Nearest-pixel distance on a tight mask does not. At the grasp, ext2 sees the block darker, so the mask falls back to a slightly lower saturation only when the strict mask is almost empty. That fallback is there so a shadowed block is not scored as “missing.”

**A block twice as wide fails the “ended in the cup” gate, and the log says why.** The 64 mm block still fits in the 85 mm hand. This run lifts it (peak z 12.1 cm) and leaves it 1.9 mm from the cup axis without settling inside the volume. The gate fails. It is not evidence that the fingers missed. Saying otherwise would be the wrong claim. The raw target and the high table are the runs that miss by centimeters (11.9 cm and 9.9 cm).

**Two full replays must match.** RANSAC is seeded. The measured pose delta on the block, cup, and cube is 0.

## Appearance

**The lab in the video is the real frame, not a reconstructed room.** The robot and the three bodies are the MuJoCo render, cut out by segmentation. The write-up says this before the pictures, because calling the plate a mesh would be false. Outcome numbers come from headless rollouts and do not read those pixels.

**Inpaint only the table, not the whole frame.** A first mask painted every saturated blue and teal component. That included the striped wall, and the inpaint turned the wall into a smear. Components whose center is in the upper part of a fixed camera are left alone. The wrist camera is all table, so it does not use that cut.

**The person in the ext2 frame is not a simulated body.** The plate is the recorded image, so the person is still in that camera. No mesh and no callout was added for them.

**The wrist camera is published after the physics step.** `data.cam_xpos` is filled by `mj_forward`. The wrist pose was written after `mj_step`, so the renderer kept the previous pose, and the first frame kept the placeholder in the XML. Fixed cameras are written before the settle, so they were already current. The OpenCV projection was right the whole time: at step 40 the pinch lands on the yellow block and the cup lands on the cup at the left edge. `mj_forward` after the wrist write is what makes the rendered view match that projection. On the half-resolution still, the dark gripper centroid moves from (389, 302) in the real frame to (417, 247) in the sim. That is the CAD palm filling more of the frame than the real fingers, not a second pose search. The real fingers stay in the plate wherever the sim mesh does not cover them.

**The wrist grasp cup is in the same 40 px gate.** At frame 0 the block and the cup both land 0.6 px from a real pixel. At the grasp the block is 29.7 px. The cup projection at that frame lands on the cup, on a pixel whose hue is 72 and whose saturation is 35. The fixed-camera mask starts at hue 88 and saturation 80, so it is empty here, and the distance to the hue-80 rim of the same cup is 54 px. That rim is not the face the projection hits. A third range, hue 70–120 and saturation 25, is used only when the first two masks are empty. The fixed cameras still score on the strict mask. The grasp-frame wrist cup is then 0.6 px. On that frame the third range covers 5,242 pixels, 0.57% of the image, all of them on the cup at the left edge (x under 120). A shift of 15 px left and 60 px up already fails the 40 px gate. The band is not a free pass across the frame.

## Reviews that changed the scene

Two hiring-manager reviews read the same tree.

One would take the interview. The reasons they gave: the raw target is a failing gate rather than a smoothed-over story, the open hand and the weld disagree, and the plate is an acceptable method because it is disclosed and the physics does not depend on it. Their objection: the scorer was cut to the scene. An 8 mm margin and a 28 mm radius floor were earning geometry points, nearest-pixel reprojection was easier than the centroid error in the same file, and `background_match` is 1.0 on pixels the composite never changed.

The other would not advance it. Their objection was the same 8 mm, plus a gate that only checks the final center, so a block carried onto the rim fails the same way as a block that never moved.

What changed after those reviews: the 8 mm constant and the 28 mm floor are gone, the rim clearance is protrusion plus the measured pad depth, the bottom radius is the stereo radius, and the nominal run must show a lift. The wide-block result is described as a settle failure, not a missed grasp. The plate stayed, because both the disclosure and the headless physics were already true, and the reviewer who would interview called that method acceptable.

## What is still a judgment

**0.95 after the stall, rather than a torque controller.** It matches the log’s saturated target, and it is applied only after the measured width stops falling. It is still a rule in code.

**Position actuators, not a prescribed joint trajectory.** The Menagerie wrist torque limit is 12 Nm. Over a 70 ms step that servo leaves the flange 1.26 cm off the log. Prescribing `qpos` to the logged joints removes that error, and it also puts the fingertips through the block while the measured gripper is still open: the pinch is below the block top from step 46, and the width is only 0.14. The block is knocked 13 cm away and never lifted. The lag is what lets the fingers close before the hand arrives. The positions of the objects are still taken from forward kinematics of the log, which matches `cartesian_position`. The dynamic replay is not that ruler.

**The real hand stays in the wrist plate.** Painting out the dark border-connected components, so the sim panel would show one gripper, left the silhouette of the fingers as a smear on the wood. Two hands in that panel is the real frame plus the simulated gripper. The smear was worse, so the inpaint was removed.

**The monitor is a local page on port 8765**, polling `outputs/progress.json`. The reference photo for that screen never arrived, so the page is a lab console (score, gates, stills, the loop log) rather than a copy of a picture.

## A review that arrived late

One review scored the scene from before this log: block half-width `0.44 ×` the finger gap, cup radii clipped to 32 mm and 50 mm, rim 32 mm under the pinch, and no grasp-frame gate. It would not have advanced that scene, and it named three changes.

The first and the third are already in the tree. The block half-width is the stereo blob, 15.9 mm. The cup bottom is the stereo radius, 24.2 mm. The rim is the fingertip plus the measured pad depth. `evalgate` scores both fixed cameras at frame 0 and at the grasp, compares two rollouts, and runs the three bad scenes. The grasp-frame miss it reported on ext2, about 49 px, does not reproduce. The current distance is 17.9 px.

The wrist claim does not reproduce as a bad extrinsic. An earlier pass of this log said a 0.5 px and 0.1 px yellow hit proved the pose. That hit was the wooden table, which shares the block’s hue, and it is withdrawn. The pinch at step 40 projects onto the block between the fingers, and the cup projects onto the cup. The side-by-side looked wrong because the renderer was a pose behind, not because the log needed a new euler order. A search over euler orders and transposes did not, and does not need to, replace the stated `xyz` convention.

## The frustum review

[Review the frustum cup scene](bc-ab4d785f-a399-5312-8d47-677874d5f52e) would not advance the frustum. The mouth was the pinch path plus the block’s corner, the height was the clearance under the fingers, the drawn cup was not the contact, and the wrist grasp was outside the gate. The condition they set: derive the height and both radii from the stereo hull, make that shell the contact, put the wrist grasp in the gate, and they advance if the block is still in the cup and the cup does not move.

That is the scene above. The block is in the cup, 2.1 mm from the axis, after a lift to 13.9 cm. The cup moves 1.3 mm and does not tilt. The flange during the replay is 1.26 cm off the log. That error is why a centimeter at the wrist is a large pixel error, and it is not absorbed by moving the cup onto the stereo centroid.
