"""trekion_real2sim_v1, the eval the submission is graded against.

Hard gates have to be true. The score is realism 35, pipeline 35, scaling 30.
A pass is every gate and a score of at least 85.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from real2sim.pipeline import run

PASS_SCORE = 85
REPROJ_PX = 40.0


def _write_progress(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.json").write_text(json.dumps(payload, indent=2))


def _inside(payload: dict) -> bool:
    return bool(payload["checks"].get("block_ended_in_cup"))


def _on_table(payload: dict) -> bool:
    scene = payload["scene"]
    yellow_z = float(payload["bodies"]["yellow"][2])
    rest_z = float(scene["table_z"]) + float(scene["block_half"])
    return abs(yellow_z - rest_z) < 0.03 and not _inside(payload)


def _pose_delta(left: dict, right: dict) -> float:
    worst = 0.0
    for name in ("yellow", "cup", "teal"):
        for index in range(3):
            worst = max(worst, abs(float(left["bodies"][name][index]) - float(right["bodies"][name][index])))
    return worst


def _reproj_ok(numbers: dict) -> tuple[bool, list[str]]:
    missing = []
    for when in ("frame0", "grasp"):
        for camera in ("ext1", "ext2", "wrist"):
            for obj in ("yellow", "blue"):
                key = f"reproj_{when}_{camera}_{obj}_px"
                if key not in numbers or float(numbers[key]) >= REPROJ_PX:
                    missing.append(key)
    return len(missing) == 0, missing


def _points(nominal: dict, gates: dict, visual: dict) -> dict:
    numbers = nominal["numbers"]
    provenance = nominal["provenance"]
    outcome = 0
    if gates["block_in_cup"] and gates["cup_stable"] and gates["teal_stayed"]:
        outcome += 9
    if gates["arm_tracks"]:
        outcome += 6
    geometry = 0
    if provenance.get("block_source") == "stereo" and not provenance.get("block_width_clipped"):
        geometry += 4
    elif provenance.get("block_source") == "finger_gap":
        geometry += 2
    # The cup's two radii and its height are the stereo hull. The fingertip clearance is not a cup measurement.
    bottom = float(nominal["scene"]["cup_bottom_radius"])
    top = float(nominal["scene"]["cup_top_radius"])
    height = float(nominal["scene"]["cup_height"])
    radii_match = (
        abs(bottom - float(provenance.get("hull_bottom_radius_m", -1.0))) < 0.003
        and abs(top - float(provenance.get("hull_top_radius_m", -1.0))) < 0.003
        and not provenance.get("cup_hull_clipped")
    )
    if radii_match:
        geometry += 2
    if abs(height - float(provenance.get("hull_height_m", -1.0))) < 0.003 and not provenance.get("cup_hull_clipped"):
        geometry += 2
    realism = {
        "outcome_and_tracking": outcome,
        "geometry_from_measurement": geometry,
        "visual": visual["points"],
        "total": outcome + geometry + visual["points"],
    }
    pipeline = {
        "one_command": 10,
        "double_run": 10 if gates["two_runs_match"] else 0,
        "bad_scenes": 5 * sum(1 for key in ("raw_target_fails", "wide_block_fails", "high_table_fails") if gates[key]),
    }
    pipeline["total"] = pipeline["one_command"] + pipeline["double_run"] + pipeline["bad_scenes"]
    scaling = {
        "measured_time": 10 if nominal.get("simulate_s") else 0,
        "demonstrated_failures": 10 if gates["raw_target_fails"] and gates["wide_block_fails"] and gates["high_table_fails"] else 0,
        "what_stays_human": 8,
    }
    scaling["total"] = scaling["measured_time"] + scaling["demonstrated_failures"] + scaling["what_stays_human"]
    return {
        "realism": realism,
        "pipeline": pipeline,
        "scaling": scaling,
        "total": realism["total"] + pipeline["total"] + scaling["total"],
        "numbers_used": {
            "joint_rmse_rad": numbers["joint_rmse_rad"],
            "flange_rmse_m": numbers["flange_rmse_m"],
            "block_to_cup_m": numbers["block_to_cup_m"],
            "cup_drift_m": numbers["cup_drift_m"],
            "teal_drift_m": numbers["teal_drift_m"],
            "simulate_s": nominal.get("simulate_s"),
        },
    }


def _visual_points(numbers: dict) -> dict:
    """Appearance points from the rendered frame, when a render exists."""
    yellow = float(numbers.get("iou_ext1_yellow", 0.0))
    blue = float(numbers.get("iou_ext1_blue", 0.0))
    points = 0
    if yellow >= 0.35:
        points += 4
    elif yellow >= 0.2:
        points += 2
    if blue >= 0.15:
        points += 4
    elif blue >= 0.05:
        points += 2
    # The third slice is the background. A pure mesh room does not match the lab.
    background = float(numbers.get("background_match", 0.0))
    if background >= 0.5:
        points += 4
    elif background >= 0.25:
        points += 2
    return {"points": min(12, points), "yellow_iou": yellow, "blue_iou": blue, "background_match": background}


def evaluate(data: Path, assets: Path, out: Path, render: bool = False) -> dict:
    data, assets, out = Path(data), Path(assets), Path(out)
    log: list[dict] = []

    def mark(step: str, detail: str) -> None:
        log.append({"step": step, "detail": detail, "t": time.time()})
        _write_progress(out, {"eval": "trekion_real2sim_v1", "status": "running", "log": log})

    mark("nominal", "calibrate and replay")
    nominal = run(data, assets, out, render=render, write=False)
    mark("repeat", "second run for determinism")
    repeat = run(data, assets, out, render=False, write=False)
    mark("open", "gripper held open")
    opened = run(data, assets, out, render=False, write=False, gripper="open")
    mark("raw", "logged target gripper")
    raw = run(data, assets, out, render=False, write=False, gripper="raw_target")
    mark("wide", "block twice as wide")
    wide = run(data, assets, out, render=False, write=False, block_scale=2.0)
    mark("table", "table 5 cm high")
    high = run(data, assets, out, render=False, write=False, table_lift=0.05)
    mark("weld", "block welded to the gripper, hand open")
    welded = run(data, assets, out, render=False, write=False, gripper="open", weld_block=True)

    reproj_pass, reproj_missing = _reproj_ok(nominal["numbers"])
    numbers = nominal["numbers"]
    gates = {
        "free_bodies": bool(nominal["free_bodies"]) and not nominal["weld_block"],
        "open_leaves_block_on_table": _on_table(opened),
        "welded_block_is_rejected": not _on_table(welded),
        "block_in_cup": _inside(nominal),
        "block_was_lifted": float(nominal["block_peak_z"]) > float(nominal["scene"]["table_z"]) + float(nominal["scene"]["block_half"]) + 0.05,
        "cup_stable": float(numbers["cup_drift_m"]) < 0.02 and float(numbers["cup_tilt_deg"]) < 15.0,
        "teal_stayed": float(numbers["teal_drift_m"]) < 0.02,
        "arm_tracks": float(numbers["joint_rmse_rad"]) < 0.05 and float(numbers["flange_rmse_m"]) < 0.02,
        "two_runs_match": _pose_delta(nominal, repeat) < 1e-6,
        "raw_target_fails": not _inside(raw),
        "wide_block_fails": not _inside(wide),
        "high_table_fails": not _inside(high),
        "reprojection": reproj_pass,
    }
    visual = _visual_points(numbers)
    points = _points(nominal, gates, visual)
    passed = all(gates.values()) and points["total"] >= PASS_SCORE
    report = {
        "eval": "trekion_real2sim_v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "pass_score": PASS_SCORE,
        "gates": gates,
        "score": points,
        "reproj_missing_or_over": reproj_missing,
        "cases": {
            "nominal_block_to_cup_m": numbers["block_to_cup_m"],
            "open_yellow_z": opened["bodies"]["yellow"][2],
            "raw_in_cup": _inside(raw),
            "raw_block_to_cup_m": raw["numbers"]["block_to_cup_m"],
            "wide_in_cup": _inside(wide),
            "wide_block_to_cup_m": wide["numbers"]["block_to_cup_m"],
            "wide_block_peak_z": wide["block_peak_z"],
            "nominal_peak_z": nominal["block_peak_z"],
            "high_in_cup": _inside(high),
            "high_block_to_cup_m": high["numbers"]["block_to_cup_m"],
            "welded_yellow_z": welded["bodies"]["yellow"][2],
            "pose_delta_m": _pose_delta(nominal, repeat),
        },
        "log": log,
    }
    _write_progress(out, report)
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    if render:
        nominal_written = run(data, assets, out, render=True, write=True)
        report["score"]["realism"]["visual"] = _visual_points(nominal_written["numbers"])
        # Recompute the total with the rendered appearance.
        report["score"]["realism"]["total"] = (
            report["score"]["realism"]["outcome_and_tracking"]
            + report["score"]["realism"]["geometry_from_measurement"]
            + report["score"]["realism"]["visual"]["points"]
        )
        report["score"]["total"] = (
            report["score"]["realism"]["total"]
            + report["score"]["pipeline"]["total"]
            + report["score"]["scaling"]["total"]
        )
        report["passed"] = all(gates.values()) and report["score"]["total"] >= PASS_SCORE
        report["status"] = "passed" if report["passed"] else "failed"
        _write_progress(out, report)
        (out / "eval.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score the episode against trekion_real2sim_v1")
    parser.add_argument("--data", type=Path, default=Path("data/episode"))
    parser.add_argument("--assets", type=Path, default=Path("third_party/mujoco_menagerie"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.data, args.assets, args.out, render=args.render)
    print(json.dumps({"passed": report["passed"], "score": report["score"]["total"], "gates": report["gates"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
