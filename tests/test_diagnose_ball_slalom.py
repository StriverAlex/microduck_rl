import importlib.util
import math
import sys
from pathlib import Path

import pytest

from mjlab_microduck.tasks.microduck_ball_slalom_env_cfg import (
    SLALOM_COM_RANDOMIZATION_RANGE,
    SLALOM_FINAL_LATERAL_OFFSET,
    SLALOM_GOAL_RADIUS,
    SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
    SLALOM_PROGRESS_REWARD_WEIGHT,
    SLALOM_PUSH_RANGE,
    SLALOM_TERMINATION_COST_WEIGHT,
    make_microduck_ball_slalom_env_cfg,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_ball_slalom", REPO / "scripts" / "diagnose_ball_slalom.py"
)
assert SPEC is not None and SPEC.loader is not None
diag = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diag
SPEC.loader.exec_module(diag)


def test_default_checkpoint_is_the_validated_preview_model():
    assert diag.DEFAULT_CHECKPOINT.parts[-2:] == (
        "2026-09-07_16-41-42_slalom_preview_40cm_v6",
        "model_10250.pt",
    )


def _terminal(**overrides):
    values = {
        "completed": False,
        "terminated": True,
        "timed_out": False,
        "fell_over": False,
        "ball_lost": False,
        "waypoint_index": 0,
    }
    values.update(overrides)
    return diag.TerminalState(**values)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (_terminal(completed=True, terminated=False, timed_out=True), "success"),
        (
            _terminal(fell_over=True, ball_lost=True, waypoint_index=2),
            "simultaneous",
        ),
        (_terminal(fell_over=True, waypoint_index=2), "fall_wp3"),
        (_terminal(ball_lost=True, waypoint_index=2), "ball_lost_wp3"),
        (_terminal(fell_over=True, waypoint_index=1), "other_failure"),
        (_terminal(timed_out=True, terminated=False), "timeout"),
    ],
)
def test_terminal_classification_is_mutually_exclusive(state, expected):
    assert diag.classify_episode(state) == expected


def test_selection_keeps_only_the_earliest_requested_samples():
    selected = {category: 0 for category in diag.TARGET_CATEGORIES}
    sequence = ["success"] * 10 + ["fall_wp3"] * 10 + ["ball_lost_wp3"] * 10
    kept = []

    for episode, category in enumerate(sequence, start=1):
        if diag.should_select(category, selected, samples_per_category=8):
            selected[category] += 1
            kept.append(episode)

    assert kept == list(range(1, 9)) + list(range(11, 19)) + list(range(21, 29))
    assert selected == {category: 8 for category in diag.TARGET_CATEGORIES}
    assert diag.quotas_met(selected, samples_per_category=8)


def test_simultaneous_terminal_is_never_selected_as_a_failure_sample():
    selected = {category: 0 for category in diag.TARGET_CATEGORIES}

    assert not diag.should_select("simultaneous", selected, samples_per_category=8)


def test_course_camera_rotates_once_with_the_randomized_course_heading():
    forward = diag.course_camera_pose((1.0, 2.0), yaw=0.0)
    left = diag.course_camera_pose((1.0, 2.0), yaw=math.pi / 2.0)

    assert forward.lookat == pytest.approx((1.60, 2.0, 0.10))
    assert forward.azimuth == pytest.approx(140.0)
    assert left.lookat == pytest.approx((1.0, 2.60, 0.10))
    assert left.azimuth == pytest.approx(230.0)


def test_diagnostic_cfg_freezes_final_course_and_keeps_full_dr():
    cfg = diag.configure_diagnostic_env_cfg(
        make_microduck_ball_slalom_env_cfg(), seed=789
    )

    command = cfg.commands["body_pose"]
    assert cfg.scene.num_envs == 1
    assert cfg.seed == 789
    assert not cfg.auto_reset
    assert command.lateral_offset == SLALOM_FINAL_LATERAL_OFFSET
    assert command.goal_radius == SLALOM_GOAL_RADIUS
    assert cfg.rewards["ball_target_progress"].weight == SLALOM_PROGRESS_REWARD_WEIGHT
    assert cfg.rewards["termination"].weight == SLALOM_TERMINATION_COST_WEIGHT
    assert cfg.curriculum == {}
    assert cfg.events["reset_ball"].params["distribution"] == "continuous"
    assert cfg.events["randomize_com"].params["ranges"] == (
        -SLALOM_COM_RANDOMIZATION_RANGE,
        SLALOM_COM_RANDOMIZATION_RANGE,
    )
    assert cfg.events["randomize_head_com"].params["ranges"] == (
        -SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
        SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
    )
    assert cfg.events["push_robot"].interval_range_s == diag.VELOCITY_PUSH_INTERVAL_S
    assert cfg.events["push_robot"].params["velocity_range"] == {
        "x": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
        "y": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
    }
    assert cfg.viewer.origin_type == cfg.viewer.OriginType.WORLD
    assert cfg.viewer.body_name is None
    assert cfg.viewer.max_extra_envs == 0
    assert (cfg.viewer.width, cfg.viewer.height) == (960, 720)
    assert cfg.viewer.distance == 3.6


def _telemetry(
    step,
    *,
    waypoint=3,
    target_speed=0.1,
    tilt=5.0,
    distance=0.2,
    fell_over=False,
    ball_forward=0.1,
    ball_lateral=0.0,
):
    return diag.Telemetry(
        step=step,
        time_s=step * 0.02,
        waypoint=waypoint,
        course_side="left",
        target_distance=0.3,
        robot_ball_distance=distance,
        target_speed=target_speed,
        lateral_speed=0.0,
        tilt_deg=tilt,
        ball_marker_contact=False,
        robot_marker_contact=False,
        fell_over=fell_over,
        ball_forward=ball_forward,
        ball_lateral=ball_lateral,
    )


def test_episode_analysis_uses_first_third_segment_precursor():
    telemetry = [_telemetry(0, waypoint=2)]
    telemetry += [_telemetry(step) for step in range(1, 5)]
    telemetry += [_telemetry(5, tilt=56.0, distance=0.31)]

    analysis = diag.analyze_episode(
        telemetry,
        step_dt=0.02,
        ball_lost_thresholds=diag.BallLostThresholds(-0.3, 0.75, 0.4),
    )

    assert analysis["third_segment_entry_s"] == pytest.approx(0.02)
    assert analysis["first_events_s"]["balance_risk"] == pytest.approx(0.10)
    assert analysis["first_events_s"]["control_separation"] == pytest.approx(0.10)
    assert analysis["primary_precursor"] == "simultaneous_precursors"


def test_episode_analysis_records_exact_terminal_boundaries_without_inference():
    thresholds = diag.BallLostThresholds(-0.3, 0.75, 0.4)
    telemetry = [_telemetry(0)]
    telemetry.append(
        _telemetry(
            1,
            distance=0.76,
            fell_over=True,
            ball_forward=-0.31,
            ball_lateral=0.41,
        )
    )

    analysis = diag.analyze_episode(
        telemetry,
        step_dt=0.02,
        ball_lost_thresholds=thresholds,
    )

    assert analysis["first_events_s"]["fell_over"] == pytest.approx(0.02)
    assert analysis["first_events_s"]["ball_lost_distance"] == pytest.approx(0.02)
    assert analysis["first_events_s"]["ball_lost_lateral"] == pytest.approx(0.02)
    assert analysis["first_events_s"]["ball_lost_backward"] == pytest.approx(0.02)


def test_episode_analysis_reports_no_observable_precursor_without_a_fallback():
    analysis = diag.analyze_episode(
        [_telemetry(0, target_speed=0.1)],
        step_dt=0.02,
        ball_lost_thresholds=diag.BallLostThresholds(-0.3, 0.75, 0.4),
    )

    assert analysis["primary_precursor"] == "no_observable_precursor"


def test_dominant_mechanism_requires_majority_in_largest_terminal_category():
    records = []
    for category, precursor, count in (
        ("fall_wp3", "balance_risk", 6),
        ("fall_wp3", "overspeed", 2),
        ("ball_lost_wp3", "balance_risk", 3),
        ("ball_lost_wp3", "control_separation", 2),
    ):
        records.extend(
            {
                "category": category,
                "analysis": {"primary_precursor": precursor},
            }
            for _ in range(count)
        )

    diagnosis = diag._diagnosis(records)

    assert diagnosis["primary_terminal_category"] == "fall_wp3"
    assert diagnosis["dominant_mechanism"] == "balance_risk"
    assert diagnosis["next_optimization_target"] is not None


def test_diagnosis_stays_inconclusive_without_a_strict_majority():
    records = [
        {"category": "fall_wp3", "analysis": {"primary_precursor": precursor}}
        for precursor in ("balance_risk", "balance_risk", "overspeed", "overspeed")
    ]
    records += [
        {
            "category": "ball_lost_wp3",
            "analysis": {"primary_precursor": "control_separation"},
        }
        for _ in range(2)
    ]

    diagnosis = diag._diagnosis(records)

    assert diagnosis["dominant_mechanism"] is None
    assert diagnosis["conclusion"] == "mixed or insufficient evidence"


def test_diagnosis_rejects_a_precursor_more_common_in_successful_controls():
    records = [
        {"category": "fall_wp3", "analysis": {"primary_precursor": "overspeed"}}
        for _ in range(6)
    ]
    records += [
        {
            "category": "ball_lost_wp3",
            "analysis": {"primary_precursor": precursor},
        }
        for precursor in ("overspeed", "obstacle_contact")
    ]
    records += [
        {"category": "success", "analysis": {"primary_precursor": "overspeed"}}
        for _ in range(9)
    ]
    records.append(
        {
            "category": "success",
            "analysis": {"primary_precursor": "obstacle_contact"},
        }
    )

    diagnosis = diag._diagnosis(records)

    assert diagnosis["dominant_mechanism"] is None
    assert diagnosis["conclusion"] == "mixed or insufficient evidence"
    assert diagnosis["mechanism_ranking"][0] == {
        "mechanism": "overspeed",
        "third_failure_count": 7,
        "third_failure_rate": 0.875,
        "primary_failure_count": 6,
        "primary_failure_rate": 1.0,
        "success_count": 9,
        "success_rate": 0.9,
    }


def test_unobserved_precursors_can_never_be_declared_dominant():
    records = [
        {
            "category": "fall_wp3",
            "analysis": {"primary_precursor": "no_observable_precursor"},
        }
        for _ in range(6)
    ]
    records += [
        {
            "category": "ball_lost_wp3",
            "analysis": {"primary_precursor": "overspeed"},
        }
        for _ in range(2)
    ]

    assert diag._diagnosis(records)["dominant_mechanism"] is None


def test_diagnosis_reports_zero_count_terminal_categories():
    records = [{"category": "success", "analysis": {"primary_precursor": "overspeed"}}]

    diagnosis = diag._diagnosis(records)

    assert diagnosis["terminal_counts"] == {
        "success": 1,
        "fall_wp3": 0,
        "ball_lost_wp3": 0,
        "simultaneous": 0,
        "timeout": 0,
        "other_failure": 0,
    }


def test_default_scan_limits_match_the_diagnostic_protocol():
    assert diag.MAX_SCAN_EPISODES == 256
    assert diag.SAMPLE_LIMIT == 8
