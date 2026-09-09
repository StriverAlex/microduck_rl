import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab_microduck.tasks.microduck_ball_slalom_env_cfg import (
    make_microduck_ball_slalom_env_cfg,
)

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_ball_slalom", REPO / "scripts" / "evaluate_ball_slalom.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def _command(num_envs: int = 1):
    return SimpleNamespace(
        course_side=torch.ones(num_envs),
        cfg=SimpleNamespace(
            cone_x=(0.45, 0.90, 1.35),
            waypoint_clearance=0.04,
            lateral_offset=0.16,
        ),
        _waypoint_pos_w=torch.tensor(
            [[[0.49, 0.16], [0.94, -0.16], [1.39, 0.16]]]
        ).repeat(num_envs, 1, 1),
        _start_ball_pos_w=torch.zeros(num_envs, 2),
        waypoint_index=torch.zeros(num_envs, dtype=torch.long),
        completed=torch.zeros(num_envs, dtype=torch.bool),
    )


def _tracker() -> evaluation.StrictRouteTracker:
    return evaluation.StrictRouteTracker(
        _command(), robot_xy=torch.tensor([[-0.10, 0.0]]), ball_xy=torch.zeros(1, 2)
    )


def _step(
    tracker: evaluation.StrictRouteTracker,
    x: float,
    y: float,
    *,
    ball_contact: bool = False,
    robot_contact: bool = False,
) -> None:
    position = torch.tensor([[x, y]])
    tracker.update(
        robot_xy=position,
        ball_xy=position,
        ball_contact=torch.tensor([ball_contact]),
        robot_contact=torch.tensor([robot_contact]),
        active=torch.tensor([True]),
    )


def test_strict_route_requires_ordered_alternating_crossings_for_ball_and_robot():
    tracker = _tracker()

    _step(tracker, 0.46, 0.10)
    _step(tracker, 0.91, -0.10)
    _step(tracker, 1.36, 0.10)

    assert tracker.completed.item()
    assert tracker.ball_index.item() == 3
    assert tracker.robot_index.item() == 3


def test_strict_route_rejects_wrong_side_even_if_later_position_is_correct():
    tracker = _tracker()

    _step(tracker, 0.46, -0.10)
    _step(tracker, 0.50, 0.10)

    assert tracker.ball_wrong_side.item()
    assert tracker.robot_wrong_side.item()
    assert tracker.ball_index.item() == 0
    assert tracker.robot_index.item() == 0
    assert not tracker.completed.item()


def test_strict_route_rejects_any_ball_or_robot_marker_contact():
    tracker = _tracker()

    _step(tracker, 0.46, 0.10, robot_contact=True)
    _step(tracker, 0.91, -0.10)
    _step(tracker, 1.36, 0.10)

    assert tracker.robot_contact.item()
    assert not tracker.completed.item()


def test_strict_route_requires_duck_as_well_as_ball_to_complete_course():
    tracker = _tracker()
    active = torch.tensor([True])

    for x, y in ((0.46, 0.10), (0.91, -0.10), (1.36, 0.10)):
        tracker.update(
            robot_xy=torch.tensor([[-0.10, 0.0]]),
            ball_xy=torch.tensor([[x, y]]),
            ball_contact=torch.tensor([False]),
            robot_contact=torch.tensor([False]),
            active=active,
        )

    assert tracker.ball_index.item() == 3
    assert tracker.robot_index.item() == 0
    assert not tracker.completed.item()


def test_generalization_scenario_rebuilds_the_physical_course_and_contact_matchers():
    cfg = make_microduck_ball_slalom_env_cfg()
    scenario = evaluation.SlalomScenario(
        name="test_five",
        cone_x=(0.40, 0.80, 1.20, 1.60, 2.00),
        lateral_offset=0.18,
        episode_length_s=24.0,
    )

    evaluation.configure_slalom_scenario(cfg, scenario)

    command = cfg.commands["body_pose"]
    assert command.cone_x == scenario.cone_x
    assert command.active_waypoints == len(scenario.cone_x)
    assert command.start_waypoint_probs == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert command.lateral_offset == scenario.lateral_offset
    assert cfg.episode_length_s == scenario.episode_length_s
    assert command.resampling_time_range == (48.0, 48.0)
    assert cfg.commands["twist"].resampling_time_range == (48.0, 48.0)
    course = cfg.scene.entities["slalom_course"].build()
    assert course.geom_names == tuple(f"slalom_cone_{i}" for i in range(1, 6))
    marker_sensors = {
        sensor.name: sensor
        for sensor in cfg.scene.sensors
        if "marker_contact" in sensor.name
    }
    assert marker_sensors["ball_marker_contact"].primary.pattern == (
        r"^slalom_cone_(?:1|2|3|4|5)$"
    )
    assert marker_sensors["robot_marker_contact"].primary.pattern == (
        r"^slalom_cone_(?:1|2|3|4|5)$"
    )


def test_generalization_battery_isolates_course_geometry_changes():
    scenarios = evaluation.GENERALIZATION_SCENARIOS

    assert tuple(scenarios) == (
        "five_standard",
        "five_tight",
        "five_wide",
        "five_irregular",
    )
    standard = scenarios["five_standard"]
    tight = scenarios["five_tight"]
    wide = scenarios["five_wide"]
    irregular = scenarios["five_irregular"]
    assert len(standard.cone_x) == 5
    assert tight.lateral_offset == standard.lateral_offset
    assert tight.cone_x[-1] < standard.cone_x[-1]
    assert wide.cone_x == standard.cone_x
    assert wide.lateral_offset > standard.lateral_offset
    irregular_gaps = tuple(
        round(next_x - current_x, 2)
        for current_x, next_x in zip(irregular.cone_x, irregular.cone_x[1:])
    )
    assert len(set(irregular_gaps)) > 1
