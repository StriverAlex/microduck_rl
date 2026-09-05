from types import SimpleNamespace

import torch
from tensordict import TensorDict

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry


class _Scene(dict):
    def __init__(self, *args, origins=None, **kwargs):
        super().__init__(*args, **kwargs)
        if origins is not None:
            self.terrain = SimpleNamespace(env_origins=origins)


class _Ball:
    def __init__(self, positions=None, velocities=None):
        self.data = SimpleNamespace(
            root_link_pos_w=positions,
            root_link_lin_vel_w=velocities,
        )
        self.pose = None
        self.velocity = None

    def write_root_link_pose_to_sim(self, pose, env_ids):
        self.pose = pose.clone()

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        self.velocity = velocity.clone()


class _Command:
    def __init__(self, selected_y):
        num_envs = len(selected_y)
        self.command = torch.stack(
            [
                torch.zeros(num_envs),
                torch.tensor(selected_y),
                torch.zeros(num_envs),
            ],
            dim=1,
        )
        self.reselection_requests = []

    def request_reselection(self, env_mask):
        self.reselection_requests.append(env_mask.clone())


def _identity_quaternions(num_envs):
    quaternions = torch.zeros(num_envs, 4)
    quaternions[:, 0] = 1.0
    return quaternions


def _selection_term(foot_xy, ball_xyz):
    num_envs = len(ball_xyz)
    term = object.__new__(mdp.BallKickSideCommand)
    term.cfg = SimpleNamespace(
        selection_margin=0.010,
        target_distance=0.12,
        longitudinal_scale=0.30,
        lateral_scale=0.15,
    )
    term._robot = SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=foot_xy,
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_quat_w=_identity_quaternions(num_envs),
        )
    )
    term._ball = _Ball(positions=ball_xyz)
    term._foot_site_ids = torch.tensor([0, 1])
    term._command = torch.zeros(num_envs, 3)
    term._pending = torch.ones(num_envs, dtype=torch.bool)
    return term


def test_side_command_encodes_ball_geometry_and_uses_strict_10mm_selection():
    feet = torch.tensor(
        [
            [[0.00, 0.05, 0.0], [0.00, -0.05, 0.0]],
            [[0.00, 0.05, 0.0], [0.00, -0.05, 0.0]],
            [[0.00, 0.05, 0.0], [0.00, -0.05, 0.0]],
        ]
    )
    balls = torch.tensor(
        [
            [0.004, 0.03, 0.035],
            [0.005, -0.03, 0.035],
            [0.60, -0.30, 0.035],
        ]
    )
    term = _selection_term(feet, balls)
    term._update_command()

    assert term.command[:, 1].tolist() == [1.0, -1.0, -1.0]
    assert torch.allclose(
        term.command[:2, 0],
        torch.tensor([(0.004 - 0.12) / 0.30, (0.005 - 0.12) / 0.30]),
    )
    assert torch.allclose(term.command[:2, 2], torch.tensor([0.2, -0.2]))
    assert term.command[2, 0] == 1.0
    assert term.command[2, 2] == -1.0


def test_side_command_latches_until_contact_tracker_requests_reselection():
    feet = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    ball = torch.tensor([[0.0, 0.0, 0.035]])
    term = _selection_term(feet, ball)
    term._update_command()
    assert term.command[0, 1] == 1.0

    term._ball.data.root_link_pos_w[0, 0] = 0.1
    term._update_command()
    assert term.command[0, 1] == 1.0

    term.request_reselection(torch.tensor([True]))
    term._update_command()
    assert term.command[0, 1] == -1.0


def _reset_env(num_envs):
    qpos = torch.zeros(num_envs, 7)
    qpos[:, 3] = 1.0
    robot = SimpleNamespace(
        indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7))
    )
    ball = _Ball()
    env = SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
        scene=_Scene(
            {"robot": robot, "ball": ball},
            origins=torch.zeros(num_envs, 3),
        ),
    )
    return env, ball


def test_dual_ball_reset_samples_initial_bimodal_region():
    torch.manual_seed(1)
    env, ball = _reset_env(20_000)
    mdp.reset_ball_for_dual_kick(env, torch.arange(env.num_envs))
    xy = ball.pose[:, :2]
    assert torch.all((xy[:, 0] >= 0.075) & (xy[:, 0] <= 0.105))
    assert torch.all((xy[:, 1].abs() >= 0.027) & (xy[:, 1].abs() <= 0.057))
    assert torch.any(xy[:, 1] < 0.0) and torch.any(xy[:, 1] > 0.0)


def test_dual_ball_reset_samples_expanded_bimodal_region():
    torch.manual_seed(2)
    env, ball = _reset_env(20_000)
    mdp.reset_ball_for_dual_kick(
        env,
        torch.arange(env.num_envs),
        distribution="lobes",
        noise_y=0.028,
    )
    xy = ball.pose[:, :2]
    assert torch.all((xy[:, 1].abs() >= 0.014) & (xy[:, 1].abs() <= 0.070))
    assert xy[:, 1].abs().amin() < 0.015
    assert xy[:, 1].abs().amax() > 0.069


def test_dual_ball_reset_samples_final_continuous_region():
    torch.manual_seed(3)
    env, ball = _reset_env(20_000)
    mdp.reset_ball_for_dual_kick(
        env,
        torch.arange(env.num_envs),
        distribution="continuous",
    )
    xy = ball.pose[:, :2]
    assert torch.all((xy[:, 0] >= 0.075) & (xy[:, 0] <= 0.105))
    assert torch.all((xy[:, 1] >= -0.070) & (xy[:, 1] <= 0.070))
    assert xy[:, 1].amin() < -0.069
    assert xy[:, 1].amax() > 0.069


def _force_history(num_envs, num_sources, history_len=4):
    return torch.zeros(num_envs, num_sources, history_len, 3)


def _tracker(feet_history, nonfoot_history, selected_y, speeds, positions=None):
    num_envs = len(selected_y)
    term = object.__new__(mdp.continuous_ball_dribble)
    term.command = _Command(selected_y)
    if positions is None:
        positions = [[0.12, 0.0, 0.035]] * num_envs
    term.ball = ball = _Ball(
        positions=torch.tensor(positions, dtype=torch.float32),
        velocities=torch.stack(
            [torch.tensor(speeds), torch.zeros(num_envs), torch.zeros(num_envs)],
            dim=1,
        ),
    )
    term.feet_sensor = SimpleNamespace(
        data=SimpleNamespace(force_history=feet_history)
    )
    term.nonfoot_sensor = SimpleNamespace(
        data=SimpleNamespace(force_history=nonfoot_history)
    )
    term.step_dt = 0.02
    term.awaiting_clear = torch.zeros(num_envs, dtype=torch.bool)
    term.clear_step_count = torch.zeros(num_envs, dtype=torch.long)
    term.possession = torch.zeros(num_envs, dtype=torch.bool)
    term.attempt_valid = torch.zeros(num_envs, dtype=torch.bool)
    term.attempt_effective = torch.zeros(num_envs, dtype=torch.bool)
    term.attempt_left = torch.zeros(num_envs, dtype=torch.bool)
    term.attempt_start_speed = torch.zeros(num_envs)
    term.previous_forward_speed = torch.zeros(num_envs)
    term.invalid_event = torch.zeros(num_envs, dtype=torch.bool)
    term.wrong_foot_event = torch.zeros(num_envs, dtype=torch.bool)
    term.nonfoot_event = torch.zeros(num_envs, dtype=torch.bool)
    term.tie_event = torch.zeros(num_envs, dtype=torch.bool)
    term.effective_kick_event = torch.zeros(num_envs, dtype=torch.bool)
    term.valid_kick_count = torch.zeros(num_envs, dtype=torch.long)
    term.left_kick_count = torch.zeros(num_envs, dtype=torch.long)
    term.right_kick_count = torch.zeros(num_envs, dtype=torch.long)
    term.wrong_foot_count = torch.zeros(num_envs, dtype=torch.long)
    term.nonfoot_count = torch.zeros(num_envs, dtype=torch.long)
    term.tie_count = torch.zeros(num_envs, dtype=torch.long)
    term.forward_distance = torch.zeros(num_envs)

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_quat_w=_identity_quaternions(num_envs),
        )
    )
    env = SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        scene={"robot": robot, "ball": ball},
        _ball_kick_dir_w=torch.tensor([[1.0, 0.0]]).repeat(num_envs, 1),
    )
    return term, env


def _compute(term, env, clear_steps=2):
    return term(
        env,
        command_name="twist",
        feet_sensor_name="feet_ball_contact",
        nonfoot_sensor_name="nonfoot_ball_contact",
        asset_name="ball",
        max_speed=0.30,
        target_distance=0.12,
        longitudinal_std=0.25,
        lateral_std=0.15,
        clear_steps=clear_steps,
        min_speed_gain=0.05,
    )


def test_contact_state_covers_valid_wrong_nonfoot_and_substep_ties():
    feet = _force_history(5, 2)
    nonfoot = _force_history(5, 3)
    feet[0, 0, 2, 0] = 1.0
    feet[1, 0, 1, 0] = 1.0
    feet[1, 1, 2, 0] = 1.0
    feet[2, 0, 1, 0] = 1.0
    nonfoot[2, 0, 2, 0] = 1.0
    feet[3, 0, 2, 0] = 1.0
    feet[3, 1, 2, 0] = 1.0
    feet[4, 0, 2, 0] = 1.0
    nonfoot[4, 1, 2, 0] = 1.0

    term, env = _tracker(feet, nonfoot, [1.0] * 5, [0.1] * 5)
    reward = _compute(term, env)

    assert term.possession.tolist() == [True, False, False, False, False]
    assert term.invalid_event.tolist() == [False, True, True, True, True]
    assert term.wrong_foot_count.tolist() == [0, 1, 0, 1, 0]
    assert term.nonfoot_count.tolist() == [0, 0, 1, 0, 1]
    assert term.tie_count.tolist() == [0, 0, 0, 1, 1]
    assert term.valid_kick_count.tolist() == [1, 0, 0, 0, 0]
    assert torch.allclose(reward, torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0]))


def test_contact_must_clear_before_the_same_ball_can_be_kicked_again():
    feet = _force_history(1, 2)
    nonfoot = _force_history(1, 1)
    feet[0, 0, 3, 0] = 1.0
    term, env = _tracker(feet, nonfoot, [1.0], [0.10])

    _compute(term, env)
    assert term.valid_kick_count.item() == 1
    assert term.awaiting_clear.item()

    term.ball.data.root_link_lin_vel_w[0, 0] = 0.15
    _compute(term, env)
    assert term.valid_kick_count.item() == 1

    feet.zero_()
    term.ball.data.root_link_lin_vel_w[0, 0] = 0.08
    _compute(term, env)
    assert term.awaiting_clear.item()
    assert not any(mask.any() for mask in term.command.reselection_requests)

    _compute(term, env)
    assert not term.awaiting_clear.item()
    assert term.command.reselection_requests[-1].tolist() == [True]

    term.command.command[0, 1] = -1.0
    feet[0, 1, 3, 0] = 1.0
    term.ball.data.root_link_lin_vel_w[0, 0] = 0.16
    _compute(term, env)
    assert term.valid_kick_count.item() == 2
    assert term.left_kick_count.item() == 1
    assert term.right_kick_count.item() == 1


def test_controlled_velocity_reward_fades_when_the_ball_is_not_close():
    feet = _force_history(1, 2)
    nonfoot = _force_history(1, 1)
    feet[0, 0, 3, 0] = 1.0
    term, env = _tracker(feet, nonfoot, [1.0], [0.20])
    near_reward = _compute(term, env).item()

    feet.zero_()
    term.ball.data.root_link_pos_w[0, 0] = 0.70
    far_reward = _compute(term, env).item()
    assert torch.isclose(torch.tensor(near_reward), torch.tensor(0.20))
    assert far_reward < near_reward * 0.01


def test_invalid_contact_is_binary_and_uses_standard_reward_dt_scaling():
    feet = _force_history(2, 2)
    nonfoot = _force_history(2, 1)
    feet[0, 1, 2, 0] = 1.0
    feet[1, 0, 2, 0] = 1.0
    term, env = _tracker(feet, nonfoot, [1.0, 1.0], [0.0, 0.1])
    _compute(term, env)
    env.reward_manager = SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(func=term)
    )

    raw = mdp.invalid_ball_contact_cost(env)
    assert raw.tolist() == [1.0, 0.0]
    assert torch.allclose(raw * -2.0 * 0.02, torch.tensor([-0.04, 0.0]))

    kick = mdp.effective_ball_kick_reward(env)
    assert kick.tolist() == [0.0, 1.0]
    assert torch.allclose(kick * 25.0 * 0.02, torch.tensor([0.0, 0.5]))


def test_state_reset_only_clears_requested_environments():
    feet = _force_history(2, 2)
    nonfoot = _force_history(2, 1)
    feet[:, 0, 3, 0] = 1.0
    term, env = _tracker(feet, nonfoot, [1.0, 1.0], [0.1, 0.2])
    _compute(term, env)
    term.reset(torch.tensor([0]))

    assert term.valid_kick_count.tolist() == [0, 1]
    assert term.possession.tolist() == [False, True]
    assert torch.allclose(term.forward_distance, torch.tensor([0.0, 0.004]))


def test_ball_speed_overshoot_uses_the_dribble_target_without_contact_gating():
    ball = _Ball(velocities=torch.tensor([[1.5, 0.0, 0.0]]))
    env = SimpleNamespace(
        scene={"ball": ball},
        _ball_kick_dir_w=torch.tensor([[1.0, 0.0]]),
    )
    cost = mdp.ball_speed_overshoot_penalty(env, target_speed=0.30)
    assert torch.allclose(cost, torch.tensor([1.20]))
    assert torch.all(cost >= 0.0)


def test_continuous_success_requires_three_kicks_distance_and_no_termination():
    term = object.__new__(mdp.continuous_ball_dribble)
    term.valid_kick_count = torch.tensor([3, 3, 2])
    term.forward_distance = torch.tensor([0.6, 0.6, 0.6])
    env = SimpleNamespace(
        reward_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(func=term)
        ),
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([False, True, False])
        ),
    )
    assert mdp.ball_dribble_success(env).tolist() == [1.0, 0.0, 0.0]


def test_ball_lost_uses_the_robot_relative_recoverable_region():
    positions = torch.tensor(
        [
            [-0.16, 0.00, 0.035],
            [0.76, 0.00, 0.035],
            [0.10, 0.41, 0.035],
            [0.10, 0.10, 0.035],
        ]
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(4, 3),
            root_link_quat_w=_identity_quaternions(4),
        )
    )
    env = SimpleNamespace(scene={"robot": robot, "ball": _Ball(positions=positions)})
    assert mdp.ball_is_lost(env).tolist() == [True, True, True, False]


def test_61d_symmetry_flips_selection_and_lateral_ball_error():
    actor = torch.zeros(1, 61)
    actor[0, 48:51] = torch.tensor([0.4, 1.0, 0.3])
    critic = torch.zeros(1, 67)
    obs = TensorDict({"actor": actor, "critic": critic}, batch_size=[1])
    actions = torch.arange(14, dtype=torch.float32).unsqueeze(0)
    mirrored_obs, mirrored_actions = microduck_vel_symmetry(None, obs, actions)

    assert torch.equal(
        mirrored_obs["actor"][1, 48:51], torch.tensor([0.4, -1.0, -0.3])
    )
    assert torch.equal(mirrored_actions[1, :5], -actions[0, 9:14])
    assert torch.equal(mirrored_actions[1, 9:14], -actions[0, :5])
