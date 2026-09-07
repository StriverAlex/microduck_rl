import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


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
