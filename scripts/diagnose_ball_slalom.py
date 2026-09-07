"""Collect stratified BallSlalom diagnostic rollouts and render selected videos."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from PIL import Image, ImageDraw, ImageFont

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_dribble_env_cfg import (
    DRIBBLE_CONTROL_RADIUS,
)
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    DUAL_BALL_TARGET_SPEED,
    VELOCITY_PUSH_INTERVAL_S,
)
from mjlab_microduck.tasks.microduck_ball_slalom_env_cfg import (
    SLALOM_COM_RANDOMIZATION_RANGE,
    SLALOM_FINAL_LATERAL_OFFSET,
    SLALOM_GOAL_RADIUS,
    SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
    SLALOM_PROGRESS_REWARD_WEIGHT,
    SLALOM_PUSH_RANGE,
    SLALOM_TERMINATION_COST_WEIGHT,
)

TASK_ID = "Mjlab-BallSlalom-Flat-MicroDuck"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/ball_slalom"
    / "2026-09-07_16-10-59_slalom_spacing_45cm_consolidation_v5"
    / "model_9996.pt"
)
TARGET_CATEGORIES = ("success", "fall_wp3", "ball_lost_wp3")
OUTCOME_CATEGORIES = (
    "success",
    "fall_wp3",
    "ball_lost_wp3",
    "simultaneous",
    "timeout",
    "other_failure",
)
MAX_SCAN_EPISODES = 256
SAMPLE_LIMIT = 8
FRAME_RATE = 25
FRAME_STRIDE = 2
BALANCE_RISK_DEG = 55.0
STALL_SPEED = 0.03
STALL_DURATION_S = 1.0


@dataclass(frozen=True)
class TerminalState:
    completed: bool
    terminated: bool
    timed_out: bool
    fell_over: bool
    ball_lost: bool
    waypoint_index: int


@dataclass(frozen=True)
class BallLostThresholds:
    min_forward: float
    max_distance: float
    max_lateral: float


@dataclass(frozen=True)
class Telemetry:
    step: int
    time_s: float
    waypoint: int
    course_side: str
    target_distance: float
    robot_ball_distance: float
    target_speed: float
    lateral_speed: float
    tilt_deg: float
    ball_marker_contact: bool
    robot_marker_contact: bool
    fell_over: bool
    ball_forward: float
    ball_lateral: float


@dataclass(frozen=True)
class SimFrame:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    telemetry_index: int


@dataclass(frozen=True)
class CameraPose:
    lookat: tuple[float, float, float]
    azimuth: float


@dataclass
class SelectedRollout:
    episode: int
    category: str
    terminal: TerminalState
    telemetry: list[Telemetry]
    frames: list[SimFrame]
    camera: CameraPose
    analysis: dict[str, Any]


def classify_episode(state: TerminalState) -> str:
    """Return one mutually exclusive terminal category."""
    if state.completed and not state.terminated:
        return "success"
    if state.fell_over and state.ball_lost:
        return "simultaneous"
    if state.waypoint_index == 2 and state.fell_over:
        return "fall_wp3"
    if state.waypoint_index == 2 and state.ball_lost:
        return "ball_lost_wp3"
    if state.timed_out:
        return "timeout"
    return "other_failure"


def should_select(
    category: str,
    selected_counts: dict[str, int],
    samples_per_category: int,
) -> bool:
    """Select the earliest requested episodes in each diagnostic category."""
    return (
        category in TARGET_CATEGORIES
        and selected_counts[category] < samples_per_category
    )


def quotas_met(selected_counts: dict[str, int], samples_per_category: int) -> bool:
    return all(
        selected_counts[category] >= samples_per_category
        for category in TARGET_CATEGORIES
    )


def configure_diagnostic_env_cfg(
    cfg: ManagerBasedRlEnvCfg,
    *,
    seed: int,
) -> ManagerBasedRlEnvCfg:
    """Freeze BallSlalom at its final training distribution with full DR."""
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False

    command = cfg.commands["body_pose"]
    command.lateral_offset = SLALOM_FINAL_LATERAL_OFFSET
    command.goal_radius = SLALOM_GOAL_RADIUS
    cfg.rewards["ball_target_progress"].weight = SLALOM_PROGRESS_REWARD_WEIGHT
    cfg.rewards["termination"].weight = SLALOM_TERMINATION_COST_WEIGHT

    cfg.events["randomize_com"].params["ranges"] = (
        -SLALOM_COM_RANDOMIZATION_RANGE,
        SLALOM_COM_RANDOMIZATION_RANGE,
    )
    cfg.events["randomize_head_com"].params["ranges"] = (
        -SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
        SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
    )
    cfg.events["reset_ball"].params["distribution"] = "continuous"
    push = cfg.events["push_robot"]
    push.interval_range_s = VELOCITY_PUSH_INTERVAL_S
    push.params["velocity_range"] = {
        "x": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
        "y": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
    }
    cfg.curriculum.clear()

    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.entity_name = None
    cfg.viewer.body_name = None
    cfg.viewer.lookat = (0.60, 0.0, 0.10)
    cfg.viewer.distance = 3.6
    cfg.viewer.azimuth = 140.0
    cfg.viewer.elevation = -28.0
    cfg.viewer.width = 960
    cfg.viewer.height = 720
    cfg.viewer.max_extra_envs = 0
    return cfg


def _scalar(value: torch.Tensor) -> float:
    return float(value[0].detach().cpu().item())


def _contact_found(raw_env: ManagerBasedRlEnv, sensor_name: str) -> bool:
    found = raw_env.scene[sensor_name].data.found
    if found is None:
        raise ValueError(f"Contact sensor '{sensor_name}' does not expose found")
    return bool((found[0] > 0).any().detach().cpu().item())


def _read_telemetry(raw_env: ManagerBasedRlEnv, step: int) -> Telemetry:
    command = raw_env.command_manager.get_term("body_pose")
    ball = raw_env.scene["ball"]
    robot = raw_env.scene["robot"]
    velocity = ball.data.root_link_lin_vel_w[0, :2]
    direction = command.direction_w[0]
    target_speed = torch.dot(velocity, direction)
    lateral_speed = direction[0] * velocity[1] - direction[1] * velocity[0]
    ball_in_base = microduck_mdp.ball_pos_in_base(raw_env, "ball")[0]
    gravity_z = robot.data.projected_gravity_b[0, 2].clamp(-1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(-gravity_z))
    side = "left" if _scalar(command.course_side) > 0.0 else "right"
    return Telemetry(
        step=step,
        time_s=step * raw_env.step_dt,
        waypoint=int(command.waypoint_index[0].detach().cpu().item()) + 1,
        course_side=side,
        target_distance=_scalar(command.distance),
        robot_ball_distance=float(
            torch.linalg.vector_norm(ball_in_base[:2]).detach().cpu().item()
        ),
        target_speed=float(target_speed.detach().cpu().item()),
        lateral_speed=float(lateral_speed.detach().cpu().item()),
        tilt_deg=float(tilt_deg.detach().cpu().item()),
        ball_marker_contact=_contact_found(raw_env, "ball_marker_contact"),
        robot_marker_contact=_contact_found(raw_env, "robot_marker_contact"),
        fell_over=bool(
            raw_env.termination_manager.get_term("fell_over")[0].detach().cpu().item()
        ),
        ball_forward=float(ball_in_base[0].detach().cpu().item()),
        ball_lateral=float(ball_in_base[1].detach().cpu().item()),
    )


def _capture_frame(raw_env: ManagerBasedRlEnv, telemetry_index: int) -> SimFrame:
    data = raw_env.sim.data
    return SimFrame(
        qpos=data.qpos[0].detach().cpu().numpy().copy(),
        qvel=data.qvel[0].detach().cpu().numpy().copy(),
        mocap_pos=data.mocap_pos[0].detach().cpu().numpy().copy(),
        mocap_quat=data.mocap_quat[0].detach().cpu().numpy().copy(),
        telemetry_index=telemetry_index,
    )


def course_camera_pose(start_xy: tuple[float, float], yaw: float) -> CameraPose:
    """Place one fixed world camera around the center of the rotated course."""
    center_distance = 0.60
    lookat = (
        start_xy[0] + center_distance * math.cos(yaw),
        start_xy[1] + center_distance * math.sin(yaw),
        0.10,
    )
    return CameraPose(lookat=lookat, azimuth=140.0 + math.degrees(yaw))


def _read_course_camera(raw_env: ManagerBasedRlEnv) -> CameraPose:
    command = raw_env.command_manager.get_term("body_pose")
    start = command._start_ball_pos_w[0].detach().cpu()
    quat = raw_env.scene["robot"].data.root_link_quat_w[0]
    qw, qx, qy, qz = (float(value.detach().cpu().item()) for value in quat)
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return course_camera_pose((float(start[0]), float(start[1])), yaw)


def _first_index(values: Iterable[bool], start: int) -> int | None:
    for index, value in enumerate(values, start=start):
        if value:
            return index
    return None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _format_percentage(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def analyze_episode(
    telemetry: list[Telemetry],
    *,
    step_dt: float,
    ball_lost_thresholds: BallLostThresholds,
) -> dict[str, Any]:
    """Summarize the third-segment window and identify its first precursor."""
    third_entry = next(
        (index for index, sample in enumerate(telemetry) if sample.waypoint == 3),
        None,
    )
    samples_per_second = max(1, round(1.0 / step_dt))
    window_start = (
        max(0, third_entry - samples_per_second)
        if third_entry is not None
        else max(0, len(telemetry) - samples_per_second)
    )
    window = telemetry[window_start:]
    first_events: dict[str, float | None] = {}
    predicates = {
        "fell_over": [sample.fell_over for sample in window],
        "balance_risk": [sample.tilt_deg >= BALANCE_RISK_DEG for sample in window],
        "control_separation": [
            sample.robot_ball_distance > DRIBBLE_CONTROL_RADIUS for sample in window
        ],
        "overspeed": [
            sample.target_speed > DUAL_BALL_TARGET_SPEED for sample in window
        ],
        "obstacle_contact": [
            sample.ball_marker_contact or sample.robot_marker_contact
            for sample in window
        ],
        "ball_lost_distance": [
            sample.robot_ball_distance > ball_lost_thresholds.max_distance
            for sample in window
        ],
        "ball_lost_lateral": [
            abs(sample.ball_lateral) > ball_lost_thresholds.max_lateral
            for sample in window
        ],
        "ball_lost_backward": [
            sample.ball_forward < ball_lost_thresholds.min_forward for sample in window
        ],
    }
    event_indices: dict[str, int] = {}
    for name, values in predicates.items():
        relative = _first_index(values, start=0)
        if relative is None:
            first_events[name] = None
        else:
            absolute = window_start + relative
            event_indices[name] = absolute
            first_events[name] = telemetry[absolute].time_s

    stall_steps = max(1, round(STALL_DURATION_S / step_dt))
    stall_run = 0
    stall_index: int | None = None
    stall_start = third_entry if third_entry is not None else window_start
    for index in range(stall_start, len(telemetry)):
        sample = telemetry[index]
        stalled = (
            sample.target_speed < STALL_SPEED
            and sample.target_distance > SLALOM_GOAL_RADIUS
        )
        stall_run = stall_run + 1 if stalled else 0
        if stall_run >= stall_steps:
            stall_index = index - stall_steps + 1
            break
    first_events["stalled"] = (
        telemetry[stall_index].time_s if stall_index is not None else None
    )
    if stall_index is not None:
        event_indices["stalled"] = stall_index

    precursor: str | None = None
    if event_indices:
        earliest_index = min(event_indices.values())
        earliest = sorted(
            name for name, index in event_indices.items() if index == earliest_index
        )
        precursor = earliest[0] if len(earliest) == 1 else "simultaneous_precursors"
    if precursor is None:
        precursor = "no_observable_precursor"

    before = (
        telemetry[max(0, third_entry - samples_per_second) : third_entry]
        if third_entry is not None
        else []
    )
    after = telemetry[third_entry:] if third_entry is not None else []

    def window_metrics(samples: list[Telemetry]) -> dict[str, float | None]:
        return {
            "mean_target_speed": _mean([sample.target_speed for sample in samples]),
            "mean_robot_ball_distance": _mean(
                [sample.robot_ball_distance for sample in samples]
            ),
            "max_tilt_deg": (
                max((sample.tilt_deg for sample in samples), default=None)
            ),
        }

    return {
        "third_segment_entry_s": (
            telemetry[third_entry].time_s if third_entry is not None else None
        ),
        "first_events_s": first_events,
        "primary_precursor": precursor,
        "one_second_before_third": window_metrics(before),
        "third_segment": window_metrics(after),
    }


def _terminal_state(raw_env: ManagerBasedRlEnv) -> TerminalState:
    command = raw_env.command_manager.get_term("body_pose")
    return TerminalState(
        completed=bool(command.completed[0].detach().cpu().item()),
        terminated=bool(
            raw_env.termination_manager.terminated[0].detach().cpu().item()
        ),
        timed_out=bool(raw_env.termination_manager.time_outs[0].detach().cpu().item()),
        fell_over=bool(
            raw_env.termination_manager.get_term("fell_over")[0].detach().cpu().item()
        ),
        ball_lost=bool(
            raw_env.termination_manager.get_term("ball_lost")[0].detach().cpu().item()
        ),
        waypoint_index=int(command.waypoint_index[0].detach().cpu().item()),
    )


def _episode_record(
    episode: int,
    category: str,
    terminal: TerminalState,
    telemetry: list[Telemetry],
    analysis: dict[str, Any],
    *,
    finite: bool,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "category": category,
        "terminal": asdict(terminal),
        "duration_s": telemetry[-1].time_s,
        "course_side": telemetry[-1].course_side,
        "finite": finite,
        "analysis": analysis,
    }


def _restore_frame(raw_env: ManagerBasedRlEnv, frame: SimFrame) -> None:
    data = raw_env.sim.data
    data.qpos[0].copy_(torch.as_tensor(frame.qpos, device=raw_env.device))
    data.qvel[0].copy_(torch.as_tensor(frame.qvel, device=raw_env.device))
    data.mocap_pos[0].copy_(torch.as_tensor(frame.mocap_pos, device=raw_env.device))
    data.mocap_quat[0].copy_(torch.as_tensor(frame.mocap_quat, device=raw_env.device))


def _event_lines(analysis: dict[str, Any]) -> list[str]:
    labels = {
        "fell_over": "fall",
        "balance_risk": "tilt",
        "control_separation": "control",
        "overspeed": "speed",
        "obstacle_contact": "contact",
        "ball_lost_distance": "lost_dist",
        "ball_lost_lateral": "lost_lat",
        "ball_lost_backward": "lost_back",
        "stalled": "stall",
    }
    parts: list[str] = []
    for name, value in analysis["first_events_s"].items():
        if value is not None:
            parts.append(f"{labels[name]}={value:.2f}s")
    if not parts:
        return ["first: none"]
    midpoint = (len(parts) + 1) // 2
    lines = ["first: " + ", ".join(parts[:midpoint])]
    if midpoint < len(parts):
        lines.append("       " + ", ".join(parts[midpoint:]))
    return lines


def _overlay(
    frame: np.ndarray,
    rollout: SelectedRollout,
    sample: Telemetry,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    lines = [
        (
            f"episode={rollout.episode:04d}  outcome={rollout.category}  "
            f"t={sample.time_s:5.2f}s  waypoint={sample.waypoint}/3  "
            f"side={sample.course_side}"
        ),
        (
            f"target_dist={sample.target_distance:.3f}m  "
            f"robot_ball={sample.robot_ball_distance:.3f}m  "
            f"tilt={sample.tilt_deg:5.1f}deg"
        ),
        (
            f"v_target={sample.target_speed:+.3f}m/s  "
            f"v_lateral={sample.lateral_speed:+.3f}m/s  "
            f"ball_xy=({sample.ball_forward:+.3f},{sample.ball_lateral:+.3f})m"
        ),
        (
            f"contact ball={int(sample.ball_marker_contact)} "
            f"robot={int(sample.robot_marker_contact)}  "
            f"primary={rollout.analysis['primary_precursor']}"
        ),
    ]
    lines.extend(_event_lines(rollout.analysis))
    line_height = 23
    box_height = 12 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((10, 7 + index * line_height), line, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _render_rollout(
    raw_env: ManagerBasedRlEnv,
    rollout: SelectedRollout,
    output_path: Path,
) -> None:
    renderer = raw_env._offline_renderer
    if renderer is None:
        raise RuntimeError("rgb_array renderer is not initialized")
    renderer._cam.lookat[:] = rollout.camera.lookat
    renderer._cam.azimuth = rollout.camera.azimuth

    def frames() -> Iterable[np.ndarray]:
        for frame in rollout.frames:
            _restore_frame(raw_env, frame)
            rendered = raw_env.render()
            if rendered is None:
                raise RuntimeError("rgb_array renderer returned no frame")
            sample = rollout.telemetry[frame.telemetry_index]
            yield _overlay(rendered, rollout, sample)

    media.write_video(output_path, frames(), fps=FRAME_RATE, qp=20)


def _averaged_window_metrics(
    records: list[dict[str, Any]], category: str, window: str
) -> dict[str, float | None]:
    matching = [record for record in records if record["category"] == category]
    result: dict[str, float | None] = {}
    for metric in (
        "mean_target_speed",
        "mean_robot_ball_distance",
        "max_tilt_deg",
    ):
        values = [
            record["analysis"][window][metric]
            for record in matching
            if record["analysis"][window][metric] is not None
        ]
        result[metric] = float(np.mean(values)) if values else None
    return result


def _diagnosis(records: list[dict[str, Any]]) -> dict[str, Any]:
    third_failures = [
        record
        for record in records
        if record["category"] in ("fall_wp3", "ball_lost_wp3")
    ]
    terminal_counts = {
        category: sum(record["category"] == category for record in records)
        for category in OUTCOME_CATEGORIES
    }
    target_terminal_counts = {
        category: terminal_counts.get(category, 0)
        for category in ("fall_wp3", "ball_lost_wp3")
    }
    primary_terminal = max(target_terminal_counts, key=target_terminal_counts.get)
    if target_terminal_counts["fall_wp3"] == target_terminal_counts["ball_lost_wp3"]:
        primary_terminal = None

    precursor_counts: dict[str, int] = {}
    for record in third_failures:
        precursor = record["analysis"]["primary_precursor"]
        precursor_counts[precursor] = precursor_counts.get(precursor, 0) + 1

    success_records = [record for record in records if record["category"] == "success"]
    success_precursors: dict[str, int] = {}
    for record in success_records:
        precursor = record["analysis"]["primary_precursor"]
        success_precursors[precursor] = success_precursors.get(precursor, 0) + 1

    dominant = None
    primary_precursors: dict[str, int] = {}
    if primary_terminal is not None:
        primary_records = [
            record
            for record in third_failures
            if record["category"] == primary_terminal
        ]
        for record in primary_records:
            precursor = record["analysis"]["primary_precursor"]
            primary_precursors[precursor] = primary_precursors.get(precursor, 0) + 1
        if primary_precursors and precursor_counts:
            category_top = max(primary_precursors, key=primary_precursors.get)
            overall_top = max(precursor_counts, key=precursor_counts.get)
            if (
                category_top == overall_top
                and category_top
                not in ("no_observable_precursor", "simultaneous_precursors")
                and primary_precursors[category_top] > len(primary_records) / 2
                and precursor_counts[category_top] > len(third_failures) / 2
                and (
                    precursor_counts[category_top] / len(third_failures)
                    > success_precursors.get(category_top, 0) / len(success_records)
                    if success_records
                    else True
                )
            ):
                dominant = category_top

    mechanism_ranking = [
        {
            "mechanism": mechanism,
            "third_failure_count": count,
            "third_failure_rate": count / len(third_failures),
            "primary_failure_count": primary_precursors.get(mechanism, 0),
            "primary_failure_rate": (
                primary_precursors.get(mechanism, 0)
                / target_terminal_counts[primary_terminal]
                if primary_terminal is not None
                else None
            ),
            "success_count": success_precursors.get(mechanism, 0),
            "success_rate": (
                success_precursors.get(mechanism, 0) / len(success_records)
                if success_records
                else None
            ),
        }
        for mechanism, count in sorted(
            precursor_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    targets = {
        "balance_risk": "Improve balance and recovery during the third segment.",
        "control_separation": "Improve close ball control before the final turn.",
        "fell_over": "Improve balance and recovery during the third segment.",
        "ball_lost_distance": "Improve close ball control before the final endpoint.",
        "ball_lost_lateral": "Improve lateral ball retention on the final turn.",
        "ball_lost_backward": "Prevent backward ball loss during final-turn recovery.",
        "overspeed": "Improve ball-speed regulation when entering the third segment.",
        "obstacle_contact": "Improve clearance around the final marker.",
        "stalled": "Improve sustained progress through the final segment.",
    }
    return {
        "terminal_counts": terminal_counts,
        "primary_terminal_category": primary_terminal,
        "third_failure_precursor_counts": dict(sorted(precursor_counts.items())),
        "success_precursor_counts": dict(sorted(success_precursors.items())),
        "mechanism_ranking": mechanism_ranking,
        "dominant_mechanism": dominant,
        "conclusion": (
            "dominant mechanism identified"
            if dominant is not None
            else "mixed or insufficient evidence"
        ),
        "next_optimization_target": (
            targets[dominant]
            if dominant is not None
            else (
                "Resolve the failure-specific third-segment precursor before "
                "changing training design."
            )
        ),
    }


def _write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    diagnosis = manifest["diagnosis"]
    counts = manifest["category_counts"]
    scanned = manifest["episodes_scanned"]
    lines = [
        "# BallSlalom diagnostic report",
        "",
        f"- Checkpoint: `{manifest['checkpoint']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Episodes scanned: `{scanned}`",
        f"- Requested samples per category: `{manifest['samples_per_category']}`",
        f"- Quotas met: `{manifest['quotas_met']}`",
        "",
        "## Population outcomes",
        "",
    ]
    for category in OUTCOME_CATEGORIES:
        count = counts[category]
        lines.append(f"- `{category}`: {count} ({count / scanned:.2%})")
    lines.extend(
        [
            "",
            "## Third-marker diagnosis",
            "",
            f"- Primary terminal category: `{diagnosis['primary_terminal_category']}`",
            f"- Precursor counts: `{diagnosis['third_failure_precursor_counts']}`",
            f"- Success-control counts: `{diagnosis['success_precursor_counts']}`",
            f"- Conclusion: **{diagnosis['conclusion']}**",
            f"- Dominant mechanism: `{diagnosis['dominant_mechanism']}`",
            f"- Next optimization target: {diagnosis['next_optimization_target']}",
            "",
            (
                "A mechanism is declared dominant only when it explains more than "
                "half of the largest third-marker failure category and all exclusive "
                "third-marker failures, is their most common precursor, and is more "
                "frequent in failures than in successful controls."
            ),
            "",
            "## Mechanism ranking",
            "",
        ]
    )
    for item in diagnosis["mechanism_ranking"]:
        lines.append(
            f"- `{item['mechanism']}`: third failures "
            f"{item['third_failure_count']} "
            f"({_format_percentage(item['third_failure_rate'])}), "
            f"primary category {item['primary_failure_count']} "
            f"({_format_percentage(item['primary_failure_rate'])}), successful "
            f"controls {item['success_count']} "
            f"({_format_percentage(item['success_rate'])})"
        )
    lines.extend(["", "## Third-segment controls", ""])
    for category in TARGET_CATEGORIES:
        before = _averaged_window_metrics(records, category, "one_second_before_third")
        after = _averaged_window_metrics(records, category, "third_segment")
        lines.append(f"### {category}")
        lines.append("")
        lines.append(f"- One second before third segment: `{before}`")
        lines.append(f"- Third segment: `{after}`")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnosis(
    checkpoint: Path,
    output_dir: Path,
    *,
    seed: int,
    samples_per_category: int,
    max_episodes: int,
    device: str,
) -> bool:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if samples_per_category <= 0:
        raise ValueError("samples_per_category must be positive")
    if max_episodes < samples_per_category:
        raise ValueError("max_episodes must be at least samples_per_category")
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir()
    for category in TARGET_CATEGORIES:
        (output_dir / category).mkdir()

    env_cfg = configure_diagnostic_env_cfg(load_env_cfg(TASK_ID), seed=seed)
    ball_lost_params = env_cfg.terminations["ball_lost"].params
    ball_lost_thresholds = BallLostThresholds(
        min_forward=float(ball_lost_params["min_forward"]),
        max_distance=float(ball_lost_params["max_distance"]),
        max_lateral=float(ball_lost_params["max_lateral"]),
    )
    agent_cfg = load_rl_cfg(TASK_ID)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    selected_counts = {category: 0 for category in TARGET_CATEGORIES}
    selected: list[SelectedRollout] = []
    records: list[dict[str, Any]] = []
    actor_dim = 0
    action_dim = 0

    try:
        obs = env.get_observations()
        with torch.inference_mode():
            for episode in range(1, max_episodes + 1):
                telemetry = [_read_telemetry(raw_env, step=0)]
                frames = [_capture_frame(raw_env, telemetry_index=0)]
                camera = _read_course_camera(raw_env)
                finite = True
                step = 0
                while True:
                    actions = policy(obs)
                    actor_dim = obs["actor"].shape[-1]
                    action_dim = actions.shape[-1]
                    finite &= bool(
                        torch.isfinite(obs["actor"]).all()
                        and torch.isfinite(actions).all()
                    )
                    obs, rewards, dones, _ = env.step(actions)
                    step += 1
                    finite &= bool(torch.isfinite(rewards).all())
                    telemetry.append(_read_telemetry(raw_env, step=step))
                    if step % FRAME_STRIDE == 0:
                        frames.append(
                            _capture_frame(raw_env, telemetry_index=len(telemetry) - 1)
                        )
                    if bool(dones[0].detach().cpu().item()):
                        if frames[-1].telemetry_index != len(telemetry) - 1:
                            frames.append(
                                _capture_frame(
                                    raw_env, telemetry_index=len(telemetry) - 1
                                )
                            )
                        break

                terminal = _terminal_state(raw_env)
                category = classify_episode(terminal)
                analysis = analyze_episode(
                    telemetry,
                    step_dt=raw_env.step_dt,
                    ball_lost_thresholds=ball_lost_thresholds,
                )
                record = _episode_record(
                    episode,
                    category,
                    terminal,
                    telemetry,
                    analysis,
                    finite=finite,
                )
                records.append(record)
                (episodes_dir / f"episode_{episode:04d}.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
                )
                if should_select(category, selected_counts, samples_per_category):
                    selected_counts[category] += 1
                    selected.append(
                        SelectedRollout(
                            episode=episode,
                            category=category,
                            terminal=terminal,
                            telemetry=telemetry,
                            frames=frames,
                            camera=camera,
                            analysis=analysis,
                        )
                    )

                print(
                    f"episode={episode:04d} category={category} "
                    f"selected={selected_counts}",
                    flush=True,
                )
                if quotas_met(selected_counts, samples_per_category):
                    break
                raw_env.reset()
                obs = env.get_observations()

        if actor_dim != 61 or action_dim != 14:
            raise RuntimeError(
                f"Policy contract changed: actor={actor_dim}, action={action_dim}"
            )

        for rollout in selected:
            index = sum(
                other.category == rollout.category and other.episode <= rollout.episode
                for other in selected
            )
            stem = f"{index:02d}_episode_{rollout.episode:04d}"
            category_dir = output_dir / rollout.category
            _render_rollout(raw_env, rollout, category_dir / f"{stem}.mp4")
            sidecar = {
                "episode": rollout.episode,
                "category": rollout.category,
                "terminal": asdict(rollout.terminal),
                "camera": asdict(rollout.camera),
                "analysis": rollout.analysis,
                "telemetry": [asdict(sample) for sample in rollout.telemetry],
            }
            (category_dir / f"{stem}.json").write_text(
                json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
            )
    finally:
        env.close()

    category_counts = {category: 0 for category in OUTCOME_CATEGORIES}
    for record in records:
        category = record["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    complete = quotas_met(selected_counts, samples_per_category)
    missing = {
        category: max(0, samples_per_category - selected_counts[category])
        for category in TARGET_CATEGORIES
    }
    manifest = {
        "task": TASK_ID,
        "checkpoint": str(checkpoint.resolve()),
        "seed": seed,
        "episodes_scanned": len(records),
        "max_episodes": max_episodes,
        "samples_per_category": samples_per_category,
        "selected_counts": selected_counts,
        "missing_counts": missing,
        "quotas_met": complete,
        "category_counts": category_counts,
        "video": {"width": 960, "height": 720, "fps": FRAME_RATE},
        "ball_lost_thresholds": asdict(ball_lost_thresholds),
        "actor_dim": actor_dim,
        "action_dim": action_dim,
        "all_finite": all(record["finite"] for record in records),
        "diagnosis": _diagnosis(records),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(output_dir, manifest, records)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--samples-per-category", type=int, default=SAMPLE_LIMIT)
    parser.add_argument("--max-episodes", type=int, default=MAX_SCAN_EPISODES)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (
        args.checkpoint.parent / "diagnostics" / f"seed_{args.seed}"
    )
    configure_torch_backends()
    complete = run_diagnosis(
        args.checkpoint,
        output_dir,
        seed=args.seed,
        samples_per_category=args.samples_per_category,
        max_episodes=args.max_episodes,
        device=args.device,
    )
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
