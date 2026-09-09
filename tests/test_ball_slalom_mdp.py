from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp


def _identity_quaternions(num_envs: int) -> torch.Tensor:
    quat = torch.zeros(num_envs, 4)
    quat[:, 0] = 1.0
    return quat


class _Course:
    def __init__(self):
        self.last_pose = None

    def write_mocap_pose_to_sim(self, pose, env_ids):
        self.last_pose = (pose.clone(), env_ids.clone())


def _slalom_term() -> mdp.BallSlalomCommand:
    term = object.__new__(mdp.BallSlalomCommand)
    term._env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        scene=SimpleNamespace(terrain=SimpleNamespace(env_origins=torch.zeros(1, 3))),
        step_dt=0.02,
    )
    term.cfg = SimpleNamespace(
        cone_x=(0.45, 0.90, 1.35, 1.80, 2.25),
        active_waypoints=3,
        start_waypoint_probs=(1.0, 0.0, 0.0, 0.0, 0.0),
        lateral_offset=0.16,
        waypoint_clearance=0.04,
        route_lateral_margin=0.06,
        ball_position_scale=0.30,
        preview_distance=0.25,
        distance_scale=1.0,
        goal_radius=0.08,
    )
    term._robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=_identity_quaternions(1),
            root_link_pos_w=torch.zeros(1, 3),
        )
    )
    term._ball = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.10, 0.00, 0.035]]))
    )
    term._course = _Course()
    term._command = torch.zeros(1, 6)
    term._target_pos_w = torch.zeros(1, 2)
    term._start_ball_pos_w = torch.zeros(1, 2)
    term._course_origin_w = torch.zeros(1, 2)
    term._initial_direction_w = torch.zeros(1, 2)
    term._direction_w = torch.tensor([[1.0, 0.0]])
    term._distance = torch.full((1,), torch.inf)
    term._sample_angle = torch.zeros(1)
    term._sample_distance = torch.zeros(1)
    term._target_epoch = torch.zeros(1, dtype=torch.long)
    term._pending = torch.ones(1, dtype=torch.bool)
    term._waypoint_pos_w = torch.zeros(1, 5, 2)
    term._waypoint_index = torch.zeros(1, dtype=torch.long)
    term._start_waypoint_index = torch.zeros(1, dtype=torch.long)
    term._total_waypoints = torch.full((1,), 3, dtype=torch.long)
    term._course_side = torch.ones(1)
    term._completed = torch.zeros(1, dtype=torch.bool)
    term._invalid = torch.zeros(1, dtype=torch.bool)
    term._course_forward_w = torch.zeros(1, 2)
    term._course_lateral_w = torch.zeros(1, 2)
    return term


def test_slalom_waypoints_alternate_and_course_is_anchored_to_ball():
    term = _slalom_term()
    term._update_command()

    assert torch.allclose(term.target_pos_w, torch.tensor([[0.59, 0.16]]))
    assert torch.allclose(
        term._waypoint_pos_w[0],
        torch.tensor(
            [
                [0.59, 0.16],
                [1.04, -0.16],
                [1.49, 0.16],
                [1.94, -0.16],
                [2.39, 0.16],
            ]
        ),
    )
    pose, env_ids = term._course.last_pose
    assert torch.equal(env_ids, torch.tensor([0]))
    assert torch.allclose(pose, torch.tensor([[0.10, 0.00, 0.00, 1, 0, 0, 0]]))
    assert torch.allclose(term.current_cone_pos_w, torch.tensor([[0.55, 0.00]]))


def test_slalom_command_exposes_ball_position_in_mirror_compatible_slots():
    term = _slalom_term()
    term._ball.data.root_link_pos_w[0, :2] = torch.tensor([0.15, -0.06])

    term._update_command()

    assert torch.allclose(term.command[0, 3:5], torch.tensor([-0.20, 0.50]))


def test_slalom_command_exposes_the_next_turn_before_reaching_the_waypoint():
    term = _slalom_term()
    term._update_command()

    assert term.command[0, 5].item() == -1.0

    term._course_side[:] = -1.0
    term._pending[:] = True
    term._update_command()

    assert term.command[0, 5].item() == 1.0


def test_slalom_next_turn_is_zero_on_the_final_waypoint():
    term = _slalom_term()
    term._update_command()
    term._waypoint_index[:] = 2
    term._update_command()

    assert term.command[0, 5].item() == 0.0


def test_slalom_direction_does_not_preview_before_passing_the_cone():
    term = _slalom_term()
    term._update_command()

    target = term.target_pos_w[0]
    term._ball.data.root_link_pos_w[0, :2] = torch.tensor([0.54, target[1]])
    term._update_command()

    assert term.waypoint_index.item() == 0
    assert torch.allclose(term.direction_w[0], torch.tensor([1.0, 0.0]))


def test_slalom_direction_previews_after_passing_the_cone():
    term = _slalom_term()
    term._update_command()
    term._ball.data.root_link_pos_w[0, :2] = torch.tensor([0.56, 0.16])

    term._update_command()

    assert term.waypoint_index.item() == 0
    assert term.direction_w[0, 1] < 0.0


def test_slalom_advances_in_order_and_finishes_after_the_last_cone():
    term = _slalom_term()
    term._update_command()

    expected_targets = (
        torch.tensor([1.04, -0.16]),
        torch.tensor([1.49, 0.16]),
    )
    robot_route_points = (
        torch.tensor([0.56, 0.10]),
        torch.tensor([1.01, -0.10]),
        torch.tensor([1.46, 0.10]),
    )
    for index, expected in enumerate(expected_targets, start=1):
        term._ball.data.root_link_pos_w[0, :2] = term.target_pos_w[0]
        term._robot.data.root_link_pos_w[0, :2] = robot_route_points[index - 1]
        term._update_command()
        assert term.waypoint_index.item() == index
        assert not term.completed.item()
        assert torch.allclose(term.target_pos_w[0], expected)

    term._ball.data.root_link_pos_w[0, :2] = term.target_pos_w[0]
    term._robot.data.root_link_pos_w[0, :2] = robot_route_points[2]
    term._update_command()
    assert term.completed.item()
    assert term.target_epoch.item() == 4


def test_slalom_does_not_advance_when_robot_has_not_passed_the_cone_on_route_side():
    term = _slalom_term()
    term._update_command()
    term._ball.data.root_link_pos_w[0, :2] = term.target_pos_w[0]
    term._robot.data.root_link_pos_w[0, :2] = torch.tensor([0.56, -0.10])

    term._update_command()

    assert term.waypoint_index.item() == 0


def test_slalom_always_requires_all_three_waypoints():
    term = _slalom_term()
    term._update_command()

    assert term.total_waypoints.item() == 3
    term._ball.data.root_link_pos_w[0, :2] = term.target_pos_w[0]
    term._robot.data.root_link_pos_w[0, :2] = torch.tensor([0.56, 0.10])
    term._update_command()

    assert not term.completed.item()
    assert term.waypoint_index.item() == 1


def test_slalom_resample_latches_the_curriculum_waypoint_count():
    term = _slalom_term()
    term.cfg.active_waypoints = 5

    term._resample_command(torch.tensor([0]))

    assert term.total_waypoints.item() == 5
    term.cfg.active_waypoints = 4
    assert term.total_waypoints.item() == 5

    term._resample_command(torch.tensor([0]))
    assert term.total_waypoints.item() == 4


def test_slalom_resample_latches_a_reverse_curriculum_start():
    term = _slalom_term()
    term.cfg.active_waypoints = 5
    term.cfg.start_waypoint_probs = (0.0, 0.0, 0.0, 1.0, 0.0)

    term._resample_command(torch.tensor([0]))

    assert term.start_waypoint_index.item() == 3
    assert term.waypoint_index.item() == 3
    term.cfg.start_waypoint_probs = (1.0, 0.0, 0.0, 0.0, 0.0)
    assert term.start_waypoint_index.item() == 3


def test_reverse_curriculum_moves_the_course_to_start_at_a_later_segment():
    term = _slalom_term()
    term.cfg.active_waypoints = 5
    term.cfg.start_waypoint_probs = (0.0, 0.0, 0.0, 1.0, 0.0)
    term._resample_command(torch.tensor([0]))
    term._course_side[:] = 1.0

    term._update_command()

    assert torch.allclose(term.target_pos_w, torch.tensor([[0.55, -0.32]]))
    assert torch.allclose(term._waypoint_pos_w[0, 2], torch.tensor([0.10, 0.00]))
    pose, env_ids = term._course.last_pose
    assert torch.equal(env_ids, torch.tensor([0]))
    assert torch.allclose(
        pose, torch.tensor([[-1.29, -0.16, 0.00, 1, 0, 0, 0]])
    )
    assert torch.isclose(
        term.initial_distance,
        torch.tensor([2.0 * (0.45**2 + 0.32**2) ** 0.5]),
    )


def test_waypoint_epoch_change_cannot_create_progress_reward(monkeypatch):
    target = SimpleNamespace(
        distance=torch.tensor([0.4]), target_epoch=torch.tensor([2])
    )
    term = object.__new__(mdp.directional_ball_dribble)
    term.target_command = target
    term.best_target_distance = torch.tensor([0.05])
    term.has_best_target_distance = torch.ones(1, dtype=torch.bool)
    term.last_target_epoch = torch.tensor([1])
    term.possession = torch.ones(1, dtype=torch.bool)
    term.effective_kick_event = torch.ones(1, dtype=torch.bool)
    term.remaining_progress_credit = torch.zeros(1)
    term.step_dt = 0.02

    monkeypatch.setattr(
        mdp.continuous_ball_dribble,
        "__call__",
        lambda self, *args, **kwargs: torch.zeros(1),
    )
    monkeypatch.setattr(
        mdp,
        "ball_pos_in_base",
        lambda env, asset_name: torch.tensor([[0.20, 0.0, 0.0]]),
    )

    value = term(
        SimpleNamespace(),
        command_name="twist",
        target_command_name="body_pose",
        feet_sensor_name="feet_ball_contact",
        nonfoot_sensor_name="nonfoot_ball_contact",
        asset_name="ball",
        max_speed=0.30,
        target_distance=0.12,
        longitudinal_std=0.25,
        lateral_std=0.15,
        clear_steps=2,
        min_speed_gain=0.05,
        max_control_distance=0.30,
        progress_credit_per_kick=0.15,
    )

    assert value.item() == 0.0
    assert torch.allclose(term.best_target_distance, torch.tensor([0.4]))
    assert term.last_target_epoch.item() == 2


def test_ball_control_distance_cost_is_zero_inside_and_linear_outside(monkeypatch):
    monkeypatch.setattr(
        mdp,
        "ball_pos_in_base",
        lambda env, asset_name: torch.tensor([[0.20, 0.00, 0.0], [0.30, 0.40, 0.0]]),
    )

    value = mdp.ball_control_distance_cost(
        SimpleNamespace(), asset_name="ball", control_distance=0.30
    )

    assert torch.allclose(value, torch.tensor([0.0, 0.2]))


def test_slalom_route_clearance_cost_anticipates_the_current_cone():
    term = _slalom_term()
    term._update_command()
    robot = term._robot
    ball = term._ball
    env = SimpleNamespace(
        scene={"ball": ball, "robot": robot},
        command_manager=SimpleNamespace(get_term=lambda name: term),
    )

    ball.data.root_link_pos_w[0, :2] = torch.tensor([0.45, 0.05])
    robot.data.root_link_pos_w[0, :2] = torch.tensor([0.45, 0.12])
    assert torch.allclose(
        mdp.slalom_route_clearance_cost(
            env,
            asset_names=("ball", "robot"),
            approach_distance=0.20,
            clearance_margin=0.10,
        ),
        torch.tensor([0.25]),
    )

    ball.data.root_link_pos_w[0, :2] = torch.tensor([0.30, 0.00])
    robot.data.root_link_pos_w[0, :2] = torch.tensor([0.30, 0.00])
    assert mdp.slalom_route_clearance_cost(
        env,
        asset_names=("ball", "robot"),
        approach_distance=0.20,
        clearance_margin=0.10,
    ).item() == 0.0


def test_slalom_route_clearance_cost_uses_the_worst_entity_and_route_side():
    term = _slalom_term()
    term._update_command()
    term._course_side[:] = -1.0
    robot = term._robot
    ball = term._ball
    env = SimpleNamespace(
        scene={"ball": ball, "robot": robot},
        command_manager=SimpleNamespace(get_term=lambda name: term),
    )

    ball.data.root_link_pos_w[0, :2] = torch.tensor([0.45, -0.12])
    robot.data.root_link_pos_w[0, :2] = torch.tensor([0.45, -0.05])
    assert torch.allclose(
        mdp.slalom_route_clearance_cost(
            env,
            asset_names=("ball", "robot"),
            approach_distance=0.20,
            clearance_margin=0.10,
        ),
        torch.tensor([0.25]),
    )

    term._completed[:] = True
    assert mdp.slalom_route_clearance_cost(
        env,
        asset_names=("ball", "robot"),
        approach_distance=0.20,
        clearance_margin=0.10,
    ).item() == 0.0


def test_slalom_metrics_report_ordered_completion_and_side_mass():
    term = _slalom_term()
    term._waypoint_index[:] = 2
    term._course_side[:] = -1.0
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: term),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False])),
    )

    assert mdp.ball_slalom_waypoints_completed(env).item() == 2.0
    assert torch.allclose(
        mdp.ball_slalom_course_completion(env), torch.tensor([2.0 / 3.0])
    )
    assert mdp.ball_slalom_side_mass(env, "left").item() == 0.0
    assert mdp.ball_slalom_side_mass(env, "right").item() == 1.0
    term._completed[:] = True
    assert mdp.ball_slalom_success(env).item() == 1.0

    term._invalid = torch.ones(1, dtype=torch.bool)
    assert mdp.ball_slalom_success(env).item() == 0.0


def test_slalom_metrics_count_only_the_sampled_course_suffix():
    term = _slalom_term()
    term._total_waypoints[:] = 5
    term._start_waypoint_index[:] = 3
    term._waypoint_index[:] = 4
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: term),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False])),
    )

    assert mdp.ball_slalom_waypoints_completed(env).item() == 1.0
    assert mdp.ball_slalom_course_completion(env).item() == 0.5
    term._completed[:] = True
    assert mdp.ball_slalom_waypoints_completed(env).item() == 2.0
    assert mdp.ball_slalom_course_completion(env).item() == 1.0


def test_slalom_curriculum_updates_goal_radius_with_course_geometry():
    term = _slalom_term()
    env = SimpleNamespace(
        common_step_counter=7200,
        command_manager=SimpleNamespace(get_term=lambda name: term),
    )
    stages = [
        {
            "step": 0,
            "active_waypoints": 3,
            "start_waypoint_probs": (1.0, 0.0, 0.0, 0.0, 0.0),
            "lateral_offset": 0.08,
            "goal_radius": 0.15,
        },
        {
            "step": 7200,
            "active_waypoints": 4,
            "start_waypoint_probs": (0.5, 0.0, 0.25, 0.25, 0.0),
            "lateral_offset": 0.12,
            "goal_radius": 0.12,
        },
    ]

    value = mdp.slalom_course_curriculum(
        env, torch.tensor([0]), command_name="body_pose", stages=stages
    )

    assert torch.isclose(value, torch.tensor(4.0))
    assert term.cfg.lateral_offset == 0.12
    assert term.cfg.goal_radius == 0.12
    assert term.cfg.active_waypoints == 4
    assert term.cfg.start_waypoint_probs == (0.5, 0.0, 0.25, 0.25, 0.0)
