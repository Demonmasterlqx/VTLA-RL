#!/usr/bin/env python3
"""Validate Tabero_X OpenPI debug-mode-6 artifacts without importing IsaacLab."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


END_REASONS = ("success", "terminated", "truncated", "max_steps")


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing file: {path}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read JSON {path}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path}")
        return None
    return value


def as_int(value: Any, label: str, errors: list[str]) -> int | None:
    if isinstance(value, bool):
        errors.append(f"{label} must be an integer, got bool")
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be an integer, got {value!r}")
        return None
    if result != value:
        errors.append(f"{label} must be an exact integer, got {value!r}")
        return None
    return result


def require_keys(value: dict[str, Any], keys: tuple[str, ...], label: str, errors: list[str]) -> None:
    for key in keys:
        if key not in value:
            errors.append(f"{label} missing key {key!r}")


def discover_artifact_roots(root: Path) -> list[Path]:
    if (root / "run_meta.json").is_file():
        return [root]
    roots = sorted(path for path in (root / "raw").glob("task_*/artifacts") if path.is_dir())
    if roots:
        return roots
    return sorted(path.parent for path in root.glob("**/run_meta.json") if path.parent.name == "artifacts")


def read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        errors.append(f"missing file: {path}")
        return rows
    except OSError as exc:
        errors.append(f"cannot read JSONL {path}: {exc}")
        return rows
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"blank JSONL row: {path}:{line_no}")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL row {path}:{line_no}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"JSONL row is not an object: {path}:{line_no}")
            continue
        rows.append(value)
    return rows


def ffprobe_video(path: Path, errors: list[str]) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        errors.append("ffprobe is not available on PATH")
        return None
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,width,height,r_frame_rate,codec_name,pix_fmt",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"ffprobe failed for {path}: {exc}")
        return None
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        errors.append(f"ffprobe failed for {path}: {detail}")
        return None
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        stream = streams[0]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        errors.append(f"ffprobe returned no usable video stream for {path}: {exc}")
        return None
    raw_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        frame_count = int(raw_frames)
    except (TypeError, ValueError):
        errors.append(f"ffprobe returned invalid frame count for {path}: {raw_frames!r}")
        return None
    stream["decoded_frame_count"] = frame_count
    return stream


def validate_experiment(
    artifact_root: Path,
    run_meta: dict[str, Any],
    exp_dir: Path,
    errors: list[str],
) -> dict[str, Any]:
    label = str(exp_dir)
    exp_meta = load_json(exp_dir / "exp_meta.json", errors)
    if exp_meta is None:
        return {"path": label, "valid": False}
    require_keys(
        exp_meta,
        (
            "task_suite",
            "task_id",
            "exp_idx",
            "success",
            "end_reason",
            "num_frames",
            "env_steps",
            "video_frame_count",
            "video_complete",
            "video_file",
        ),
        f"{label}/exp_meta.json",
        errors,
    )

    exp_idx = as_int(exp_meta.get("exp_idx"), f"{label} exp_idx", errors)
    env_steps = as_int(exp_meta.get("env_steps"), f"{label} env_steps", errors)
    num_frames = as_int(exp_meta.get("num_frames"), f"{label} num_frames", errors)
    meta_video_frames = as_int(
        exp_meta.get("video_frame_count"), f"{label} video_frame_count", errors
    )
    if exp_idx is not None and exp_dir.name != f"exp_{exp_idx:03d}":
        errors.append(f"{label} directory name does not match exp_idx={exp_idx}")
    if exp_meta.get("end_reason") not in END_REASONS:
        errors.append(f"{label} invalid end_reason={exp_meta.get('end_reason')!r}")
    if not isinstance(exp_meta.get("success"), bool):
        errors.append(f"{label} success must be bool")
    if exp_meta.get("success") != (exp_meta.get("end_reason") == "success"):
        errors.append(f"{label} success does not match end_reason")
    if exp_meta.get("video_complete") is not True:
        errors.append(f"{label} video_complete is not true: {exp_meta.get('video_error')!r}")

    for key in ("task_suite", "task_id"):
        if exp_meta.get(key) != run_meta.get(key):
            errors.append(
                f"{label} {key}={exp_meta.get(key)!r} does not match run_meta={run_meta.get(key)!r}"
            )

    rows = read_jsonl(exp_dir / "forces.jsonl", errors)
    replan_steps = as_int(run_meta.get("replan_steps"), f"{artifact_root} replan_steps", errors)
    for row_no, row in enumerate(rows):
        row_label = f"{exp_dir}/forces.jsonl row {row_no + 1}"
        require_keys(
            row,
            ("task_suite", "task_id", "exp_idx", "action_idx", "replan_i", "frame"),
            row_label,
            errors,
        )
        if row.get("frame") != row_no:
            errors.append(f"{row_label} frame={row.get('frame')!r}, expected {row_no}")
        for key in ("task_suite", "task_id", "exp_idx"):
            if row.get(key) != exp_meta.get(key):
                errors.append(
                    f"{row_label} {key}={row.get(key)!r}, expected {exp_meta.get(key)!r}"
                )
        if replan_steps is not None and replan_steps > 0:
            expected_action = row_no // replan_steps
            expected_replan = row_no % replan_steps
            if row.get("action_idx") != expected_action:
                errors.append(
                    f"{row_label} action_idx={row.get('action_idx')!r}, expected {expected_action}"
                )
            if row.get("replan_i") != expected_replan:
                errors.append(
                    f"{row_label} replan_i={row.get('replan_i')!r}, expected {expected_replan}"
                )

    video_name = exp_meta.get("video_file")
    video_path = exp_dir / video_name if isinstance(video_name, str) and video_name else exp_dir / "preview.mp4"
    video = ffprobe_video(video_path, errors)
    decoded_frames = video.get("decoded_frame_count") if video is not None else None

    counts = {
        "env_steps": env_steps,
        "num_frames": num_frames,
        "force_rows": len(rows),
        "metadata_video_frames": meta_video_frames,
        "decoded_video_frames": decoded_frames,
    }
    known_counts = [value for value in counts.values() if value is not None]
    if known_counts and any(value != known_counts[0] for value in known_counts[1:]):
        errors.append(f"{label} frame-count mismatch: {counts}")
    if env_steps is not None and env_steps < 0:
        errors.append(f"{label} env_steps must be nonnegative")

    if video is not None:
        resolution = exp_meta.get("video_resolution")
        actual_resolution = [video.get("width"), video.get("height")]
        if isinstance(resolution, list) and resolution != actual_resolution:
            errors.append(
                f"{label} video resolution metadata={resolution!r}, decoded={actual_resolution!r}"
            )

    return {
        "path": str(exp_dir),
        "task_id": exp_meta.get("task_id"),
        "exp_idx": exp_idx,
        "success": exp_meta.get("success"),
        "end_reason": exp_meta.get("end_reason"),
        **counts,
    }


def validate_artifact_root(artifact_root: Path, errors: list[str]) -> dict[str, Any]:
    start_errors = len(errors)
    run_meta = load_json(artifact_root / "run_meta.json", errors)
    if run_meta is None:
        return {"path": str(artifact_root), "valid": False, "experiments": []}
    require_keys(
        run_meta,
        (
            "task_suite",
            "task_id",
            "task",
            "control_mode",
            "num_total_experiments",
            "replan_steps",
            "completed_experiments",
            "successful_experiments",
            "failed_experiments",
            "success_rate_percent",
            "end_reason_counts",
        ),
        f"{artifact_root}/run_meta.json",
        errors,
    )
    exp_dirs = sorted(path for path in artifact_root.glob("exp_*") if path.is_dir())
    experiments = [validate_experiment(artifact_root, run_meta, path, errors) for path in exp_dirs]

    completed = len(experiments)
    successful = sum(item.get("success") is True for item in experiments)
    end_reasons = Counter(item.get("end_reason") for item in experiments)
    declared_completed = as_int(
        run_meta.get("completed_experiments"), f"{artifact_root} completed_experiments", errors
    )
    declared_successful = as_int(
        run_meta.get("successful_experiments"), f"{artifact_root} successful_experiments", errors
    )
    declared_failed = as_int(
        run_meta.get("failed_experiments"), f"{artifact_root} failed_experiments", errors
    )
    declared_total = as_int(
        run_meta.get("num_total_experiments"), f"{artifact_root} num_total_experiments", errors
    )
    if declared_completed != completed:
        errors.append(
            f"{artifact_root} completed_experiments={declared_completed}, discovered={completed}"
        )
    if declared_successful != successful:
        errors.append(
            f"{artifact_root} successful_experiments={declared_successful}, computed={successful}"
        )
    if declared_failed != completed - successful:
        errors.append(
            f"{artifact_root} failed_experiments={declared_failed}, computed={completed - successful}"
        )
    if declared_total is not None and completed != declared_total:
        errors.append(f"{artifact_root} completed={completed}, requested={declared_total}")

    declared_reasons = run_meta.get("end_reason_counts")
    if isinstance(declared_reasons, dict):
        for reason in END_REASONS:
            if declared_reasons.get(reason) != end_reasons.get(reason, 0):
                errors.append(
                    f"{artifact_root} end_reason_counts[{reason!r}]={declared_reasons.get(reason)!r}, "
                    f"computed={end_reasons.get(reason, 0)}"
                )
    else:
        errors.append(f"{artifact_root} end_reason_counts must be an object")

    computed_rate = 100.0 * successful / completed if completed else 0.0
    try:
        declared_rate = float(run_meta.get("success_rate_percent"))
    except (TypeError, ValueError):
        declared_rate = None
        errors.append(f"{artifact_root} success_rate_percent must be numeric")
    if declared_rate is not None and abs(declared_rate - computed_rate) > 1e-9:
        errors.append(
            f"{artifact_root} success_rate_percent={declared_rate}, computed={computed_rate}"
        )

    return {
        "path": str(artifact_root),
        "task_suite": run_meta.get("task_suite"),
        "task_id": run_meta.get("task_id"),
        "environment": run_meta.get("task"),
        "control_mode": run_meta.get("control_mode"),
        "completed_experiments": completed,
        "successful_experiments": successful,
        "success_rate_percent": computed_rate,
        "valid": len(errors) == start_errors,
        "experiments": experiments,
    }


def build_summary(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not root.exists():
        errors.append(f"run directory does not exist: {root}")
        artifact_roots: list[Path] = []
    else:
        artifact_roots = discover_artifact_roots(root)
        if not artifact_roots:
            errors.append(f"no mode-6 artifacts found under: {root}")
    runs = [validate_artifact_root(path, errors) for path in artifact_roots]
    experiments = [experiment for run in runs for experiment in run.get("experiments", [])]
    completed = len(experiments)
    successful = sum(item.get("success") is True for item in experiments)
    return {
        "valid": not errors,
        "root": str(root),
        "artifact_roots": len(runs),
        "completed_experiments": completed,
        "successful_experiments": successful,
        "success_rate_percent": 100.0 * successful / completed if completed else 0.0,
        "total_env_steps": sum(item.get("env_steps") or 0 for item in experiments),
        "total_force_rows": sum(item.get("force_rows") or 0 for item in experiments),
        "total_video_frames": sum(item.get("decoded_video_frames") or 0 for item in experiments),
        "runs": runs,
        "errors": errors,
    }


def print_text(summary: dict[str, Any]) -> None:
    status = "PASS" if summary["valid"] else "FAIL"
    print(
        f"{status}: {summary['artifact_roots']} artifact roots, "
        f"{summary['successful_experiments']}/{summary['completed_experiments']} successes, "
        f"{summary['total_env_steps']} env steps, "
        f"{summary['total_force_rows']} force rows, "
        f"{summary['total_video_frames']} decoded video frames"
    )
    for run in summary["runs"]:
        print(
            f"  task {run.get('task_id')}: "
            f"{run.get('successful_experiments')}/{run.get('completed_experiments')} "
            f"valid={run.get('valid')} path={run.get('path')}"
        )
    for error in summary["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory or a single artifacts directory")
    parser.add_argument("--json", action="store_true", help="Print only a JSON validation summary")
    args = parser.parse_args()

    root = args.run_dir.expanduser().resolve()
    summary = build_summary(root)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_text(summary)
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
