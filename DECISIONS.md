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

**Cup bottom radius is the stereo radius, 24.2 mm.** An earlier clip floor of 28 mm was rejected. It was larger than the measurement, and the geometry score was willing to treat “close to stereo” as a pass. The block’s corner is 22.5 mm from its center, so 24.2 mm can hold a centered block. After the change the block still ends 1.3 mm from the cup axis.

**Cup rim is the fingertip plus one pad depth, both read off the gripper.** The pads reach 21.6 mm past the pinch site. Putting the rim exactly there lets the departing hand drag the cup 2.5 cm. The pad box `g_right_pad1` extends another 9.4 mm along the pinch axis at the insertion pose. The rim sits that extra depth below the fingertip, and the cup then does not move. An 8 mm constant that had been tuned for the same reason was removed, because the scorer was awarding points for matching that constant. The 9.4 mm is the box support of the pad geom, not a clearance picked to pass.

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

**The wrist view is the weak camera, and it is not in the numeric reprojection gate.** A close camera turns a centimeter of cup error into a large fraction of the frame. The fixed cameras are the gate. The wrist is called out in the write-up as the view to fix next, with a stereo cup mesh rather than rings.

## Reviews that changed the scene

Two hiring-manager reviews read the same tree.

One would take the interview. The reasons they gave: the raw target is a failing gate rather than a smoothed-over story, the open hand and the weld disagree, and the plate is an acceptable method because it is disclosed and the physics does not depend on it. Their objection: the scorer was cut to the scene. An 8 mm margin and a 28 mm radius floor were earning geometry points, nearest-pixel reprojection was easier than the centroid error in the same file, and `background_match` is 1.0 on pixels the composite never changed.

The other would not advance it. Their objection was the same 8 mm, plus a gate that only checks the final center, so a block carried onto the rim fails the same way as a block that never moved.

What changed after those reviews: the 8 mm constant and the 28 mm floor are gone, the rim clearance is protrusion plus the measured pad depth, the bottom radius is the stereo radius, and the nominal run must show a lift. The wide-block result is described as a settle failure, not a missed grasp. The plate stayed, because both the disclosure and the headless physics were already true, and the reviewer who would interview called that method acceptable.

## What is still a judgment

**0.95 after the stall, rather than a torque controller.** It matches the log’s saturated target, and it is applied only after the measured width stops falling. It is still a rule in code.

**Position actuators, not torque tracking.** The brief is to replay this episode and let contact move the objects. Stiff joint tracking does that. Torque tracking is the step if the next episode is one the log has not played.

**The monitor is a local page on port 8765**, polling `outputs/progress.json`. The reference photo for that screen never arrived, so the page is a lab console (score, gates, stills, the loop log) rather than a copy of a picture.
