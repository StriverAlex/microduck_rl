from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry


def _identity_quaternions(num_envs: int) -> torch.Tensor:
    quat = torch.zeros(num_envs, 4)
    quat[:, 0] = 1.0
    return quat


def _target_term(
    ball_xy: torch.Tensor,
    *,
    angle: float = 0.0,
    distance: float = 1.0,
) -> mdp.BallTargetCommand:
    num_envs = len(ball_xy)
    term = object.__new__(mdp.BallTargetCommand)
    term._env = SimpleNamespace(device="cpu", num_envs=num_envs)
    term.cfg = SimpleNamespace(
        ranges=((angle, angle), (distance, distance)),
        distance_scale=2.0,
        goal_radius=0.15,
    )
    term._robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_quat_w=_identity_quaternions(num_envs),
        )
    )
    term._ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.cat(
                (ball_xy, torch.full((num_envs, 1), 0.035)), dim=1
            )
        )
    )
    term._command = torch.zeros(num_envs, 6)
    term._target_pos_w = torch.zeros(num_envs, 2)
    term._start_ball_pos_w = torch.zeros(num_envs, 2)
    term._initial_direction_w = torch.zeros(num_envs, 2)
    term._direction_w = torch.zeros(num_envs, 2)
    term._direction_w[:, 0] = 1.0
    term._distance = torch.full((num_envs,), torch.inf)
    term._sample_angle = torch.zeros(num_envs)
    term._sample_distance = torch.zeros(num_envs)
    term._target_epoch = torch.zeros(num_envs, dtype=torch.long)
    term._pending = torch.zeros(num_envs, dtype=torch.bool)
    return term


def test_target_command_anchors_goal_after_reset_and_updates_ball_to_goal_vector():
    term = _target_term(torch.tensor([[0.10, 0.00]]), angle=0.0, distance=1.0)
    term._resample_command(torch.tensor([0]))
    term._update_command()

    assert torch.allclose(term.target_pos_w, torch.tensor([[1.10, 0.00]]))
    assert torch.allclose(term.direction_w, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(term.command, torch.tensor([[1.0, 0.0, 0.5, 0, 0, 0]]))
    assert torch.allclose(term.distance, torch.tensor([1.0]))

    term._ball.data.root_link_pos_w[0, :2] = torch.tensor([0.60, 0.20])
    term._update_command()
    expected = torch.tensor([0.50, -0.20])
    assert torch.allclose(
        term.direction_w[0], expected / torch.linalg.vector_norm(expected)
    )


def test_target_command_samples_direction_relative_to_robot_heading():
    term = _target_term(torch.tensor([[0.0, 0.0]]), angle=torch.pi / 2.0, distance=1.0)
    term._resample_command(torch.tensor([0]))
    term._update_command()
    assert torch.allclose(term.target_pos_w, torch.tensor([[0.0, 1.0]]), atol=1e-6)
    assert torch.allclose(term.command[0, :2], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_target_command_reset_only_reanchors_requested_environments():
    term = _target_term(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    term._resample_command(torch.tensor([0, 1]))
    term._update_command()
    original_second = term.target_pos_w[1].clone()

    term._ball.data.root_link_pos_w[0, :2] = torch.tensor([2.0, 0.0])
    term._ball.data.root_link_pos_w[1, :2] = torch.tensor([3.0, 0.0])
    term._resample_command(torch.tensor([0]))
    term._update_command()

    assert torch.allclose(term.target_pos_w[0], torch.tensor([3.0, 0.0]))
    assert torch.allclose(term.target_pos_w[1], original_second)


def test_target_metrics_are_net_progress_distance_and_cross_track():
    term = _target_term(torch.tensor([[0.0, 0.0]]), distance=1.0)
    term._resample_command(torch.tensor([0]))
    term._update_command()
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: term),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False])),
    )

    term._ball.data.root_link_pos_w[0, :2] = torch.tensor([0.40, 0.20])
    term._update_command()
    remaining = torch.linalg.vector_norm(torch.tensor([0.6, -0.2]))
    assert torch.allclose(
        mdp.ball_target_progress(env), torch.tensor([1.0 - remaining]), atol=1e-6
    )
    assert torch.allclose(
        mdp.ball_target_distance(env),
        torch.tensor([remaining]),
    )
    assert torch.allclose(mdp.ball_target_cross_track_error(env), torch.tensor([0.2]))
    assert mdp.ball_target_success(env).item() == 0.0


def test_target_angle_bin_metrics_report_sample_and_success_mass():
    term = _target_term(torch.tensor([[0.0, 0.0]]), angle=0.4, distance=1.0)
    term._resample_command(torch.tensor([0]))
    term._update_command()
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: term),
        termination_manager=SimpleNamespace(terminated=torch.tensor([False])),
    )

    assert mdp.ball_target_angle_bin_mass(env, 0.2, 0.5).item() == 1.0
    assert mdp.ball_target_angle_bin_mass(env, 0.5, 0.8).item() == 0.0
    term._ball.data.root_link_pos_w[0, :2] = term.target_pos_w[0]
    term._update_command()
    assert mdp.ball_target_success_angle_bin_mass(env, 0.2, 0.5).item() == 1.0
    assert mdp.ball_target_reached(env).item()


def test_target_speed_overshoot_is_nonnegative_and_uses_live_goal_direction():
    target = _target_term(torch.tensor([[0.0, 0.0]]))
    target._resample_command(torch.tensor([0]))
    target._update_command()
    ball = target._ball
    ball.data.root_link_lin_vel_w = torch.tensor([[0.5, 0.5, 0.0]])
    env = SimpleNamespace(
        scene={"ball": ball},
        command_manager=SimpleNamespace(get_term=lambda name: target),
    )

    cost = mdp.ball_target_speed_overshoot_cost(
        env, target_command_name="body_pose", target_speed=0.30
    )
    assert torch.allclose(cost, torch.tensor([0.20]))
    assert torch.all(cost >= 0.0)


def test_directional_progress_is_controlled_monotonic_and_has_no_reentry_jackpot(
    monkeypatch,
):
    target = SimpleNamespace(
        distance=torch.tensor([1.0]), target_epoch=torch.zeros(1, dtype=torch.long)
    )
    term = object.__new__(mdp.directional_ball_dribble)
    term.target_command = target
    term.best_target_distance = torch.zeros(1)
    term.has_best_target_distance = torch.zeros(1, dtype=torch.bool)
    term.possession = torch.ones(1, dtype=torch.bool)
    term.effective_kick_event = torch.zeros(1, dtype=torch.bool)
    term.remaining_progress_credit = torch.zeros(1)
    term.last_target_epoch = torch.zeros(1, dtype=torch.long)
    term.step_dt = 0.02
    robot_ball_distance = {"value": 0.20}

    monkeypatch.setattr(
        mdp.continuous_ball_dribble,
        "__call__",
        lambda self, *args, **kwargs: torch.zeros(1),
    )
    monkeypatch.setattr(
        mdp,
        "ball_pos_in_base",
        lambda env, asset_name: torch.tensor(
            [[robot_ball_distance["value"], 0.0, 0.0]]
        ),
    )
    env = SimpleNamespace()

    def compute():
        return term(
            env,
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

    assert compute().item() == 0.0

    term.effective_kick_event.fill_(True)
    target.distance = torch.tensor([0.996])
    toward = compute()
    term.effective_kick_event.fill_(False)
    assert torch.allclose(toward, torch.tensor([0.20]), atol=1e-6)
    assert torch.allclose(toward * 12.0 * 0.02, torch.tensor([0.048]))

    target.distance = torch.tensor([1.0])
    away = compute()
    assert away.item() == 0.0
    robot_ball_distance["value"] = 0.31
    target.distance = torch.tensor([0.996])
    assert compute().item() == 0.0

    robot_ball_distance["value"] = 0.20
    target.distance = torch.tensor([0.992])
    assert torch.allclose(compute(), torch.tensor([0.20]), atol=1e-6)


def test_directional_progress_credit_does_not_stack_without_another_kick(monkeypatch):
    target = SimpleNamespace(
        distance=torch.tensor([0.8]), target_epoch=torch.zeros(1, dtype=torch.long)
    )
    term = object.__new__(mdp.directional_ball_dribble)
    term.target_command = target
    term.best_target_distance = torch.tensor([1.0])
    term.has_best_target_distance = torch.ones(1, dtype=torch.bool)
    term.possession = torch.ones(1, dtype=torch.bool)
    term.effective_kick_event = torch.ones(1, dtype=torch.bool)
    term.remaining_progress_credit = torch.zeros(1)
    term.last_target_epoch = torch.zeros(1, dtype=torch.long)
    term.step_dt = 1.0

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

    def compute():
        return term(
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

    assert torch.allclose(compute(), torch.tensor([0.15]))
    term.effective_kick_event.fill_(False)
    target.distance = torch.tensor([0.7])
    assert compute().item() == 0.0

    term.effective_kick_event.fill_(True)
    target.distance = torch.tensor([0.6])
    assert torch.allclose(compute(), torch.tensor([0.10]), atol=1e-6)


def test_directional_progress_only_rewards_a_new_best_distance(monkeypatch):
    target = SimpleNamespace(
        distance=torch.tensor([1.2]), target_epoch=torch.zeros(1, dtype=torch.long)
    )
    term = object.__new__(mdp.directional_ball_dribble)
    term.target_command = target
    term.best_target_distance = torch.tensor([1.0])
    term.has_best_target_distance = torch.ones(1, dtype=torch.bool)
    term.possession = torch.ones(1, dtype=torch.bool)
    term.effective_kick_event = torch.ones(1, dtype=torch.bool)
    term.remaining_progress_credit = torch.zeros(1)
    term.last_target_epoch = torch.zeros(1, dtype=torch.long)
    term.step_dt = 1.0

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

    def compute():
        return term(
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

    assert compute().item() == 0.0
    term.effective_kick_event.fill_(False)
    target.distance = torch.tensor([1.3])
    assert compute().item() == 0.0

    term.effective_kick_event.fill_(True)
    target.distance = torch.tensor([0.9])
    assert torch.allclose(compute(), torch.tensor([0.10]), atol=1e-6)


def test_61d_symmetry_mirrors_target_direction_without_layout_change():
    actor = torch.zeros(1, 61)
    actor[0, 48:51] = torch.tensor([0.4, 1.0, 0.3])
    actor[0, 55:61] = torch.tensor([0.8, 0.4, 0.5, 0.0, 0.0, 0.0])
    obs = TensorDict({"actor": actor, "critic": torch.zeros(1, 67)}, batch_size=[1])
    actions = torch.zeros(1, 14)
    mirrored_obs, _ = microduck_vel_symmetry(None, obs, actions)

    assert torch.equal(mirrored_obs["actor"][1, 48:51], torch.tensor([0.4, -1.0, -0.3]))
    assert torch.equal(mirrored_obs["actor"][1, 55:58], torch.tensor([0.8, -0.4, 0.5]))
