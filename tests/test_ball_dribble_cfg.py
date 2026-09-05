import math

from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import observations as obs_mdp
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_ball_dribble_env_cfg import (
    DRIBBLE_CONTROL_RADIUS,
    DRIBBLE_EPISODE_LENGTH_S,
    DRIBBLE_GOAL_RADIUS,
    DRIBBLE_PROGRESS_PER_KICK,
    DRIBBLE_PROGRESS_REWARD_WEIGHT,
    DRIBBLE_TARGET_DISTANCE_SCALE,
    DRIBBLE_TARGET_FINAL_DISTANCE_RANGE,
    DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE,
    DRIBBLE_TARGET_LONG_DISTANCE_RANGE,
    DRIBBLE_TARGET_MEDIUM_DISTANCE_RANGE,
    DRIBBLE_TARGET_NOMINAL_DISTANCE_RANGE,
    DRIBBLE_TILT_COST_WEIGHT,
    DRIBBLE_UPRIGHT_REWARD_WEIGHT,
    MicroduckBallDribbleRlCfg,
    make_microduck_ball_dribble_env_cfg,
)
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_dual_env_cfg,
)


def test_dribble_command_and_observation_contract():
    cfg = make_microduck_ball_dribble_env_cfg()
    target = cfg.commands["body_pose"]

    assert isinstance(target, mdp.BallTargetCommandCfg)
    assert target.ranges == (
        (0.0, 0.0),
        DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE,
    )
    assert target.distance_scale == DRIBBLE_TARGET_DISTANCE_SCALE
    assert target.goal_radius == DRIBBLE_GOAL_RADIUS
    assert target.resampling_time_range == (
        DRIBBLE_EPISODE_LENGTH_S * 2,
        DRIBBLE_EPISODE_LENGTH_S * 2,
    )

    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["body_command"]
        assert term.func is obs_mdp.generated_commands
        assert term.params == {"command_name": "body_pose"}
        assert cfg.observations[group].terms["head_command"].params["dim"] == 4


def test_dribble_reward_uses_target_progress_without_contact_jackpot():
    cfg = make_microduck_ball_dribble_env_cfg()

    assert "ball_forward_velocity" not in cfg.rewards
    progress = cfg.rewards["ball_target_progress"]
    assert progress.func is mdp.directional_ball_dribble
    assert progress.weight == DRIBBLE_PROGRESS_REWARD_WEIGHT
    assert progress.params["target_command_name"] == "body_pose"
    assert progress.params["max_control_distance"] == DRIBBLE_CONTROL_RADIUS
    assert progress.params["progress_credit_per_kick"] == DRIBBLE_PROGRESS_PER_KICK
    assert (
        cfg.rewards["ball_speed_overshoot"].func is mdp.ball_target_speed_overshoot_cost
    )
    assert cfg.rewards["ball_speed_overshoot"].weight < 0.0
    assert cfg.rewards["invalid_contact"].weight < 0.0
    assert cfg.rewards["invalid_contact"].params == {
        "reward_name": "ball_target_progress"
    }
    assert "effective_kick" not in cfg.rewards


def test_dribble_progress_is_primary_positive_reward_mass():
    cfg = make_microduck_ball_dribble_env_cfg()
    static_annuities = {
        "support_foot_grounded",
        "pose_stand_legs",
        "pose_stand_neck",
        "height_stand",
    }
    assert static_annuities.isdisjoint(cfg.rewards)
    assert cfg.rewards["upright"].weight == DRIBBLE_UPRIGHT_REWARD_WEIGHT
    initial_kicks = math.ceil(
        (min(DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE) - DRIBBLE_GOAL_RADIUS)
        / DRIBBLE_PROGRESS_PER_KICK
    )
    final_kicks = math.ceil(
        (max(DRIBBLE_TARGET_FINAL_DISTANCE_RANGE) - DRIBBLE_GOAL_RADIUS)
        / DRIBBLE_PROGRESS_PER_KICK
    )
    assert initial_kicks >= 2
    assert final_kicks >= 9
    assert min(DRIBBLE_TARGET_FINAL_DISTANCE_RANGE) == min(
        DRIBBLE_TARGET_LONG_DISTANCE_RANGE
    )
    assert max(DRIBBLE_TARGET_FINAL_DISTANCE_RANGE) > max(
        DRIBBLE_TARGET_LONG_DISTANCE_RANGE
    )


def test_dribble_stability_is_an_official_nonnegative_cost():
    cfg = make_microduck_ball_dribble_env_cfg()
    tilt = cfg.rewards["body_tilt"]

    assert tilt.func is base_mdp.flat_orientation_l2
    assert tilt.weight == DRIBBLE_TILT_COST_WEIGHT
    assert tilt.weight < 0.0


def test_dribble_goal_ends_as_successful_timeout():
    cfg = make_microduck_ball_dribble_env_cfg()
    term = cfg.terminations["target_reached"]

    assert term.func is mdp.ball_target_reached
    assert term.time_out
    assert term.params == {"command_name": "body_pose"}
    assert "target_reached" not in cfg.rewards


def test_dribble_curriculum_expands_one_difficulty_at_a_time():
    cfg = make_microduck_ball_dribble_env_cfg()
    stages = cfg.curriculum["target_range"].params["range_stages"]

    assert stages == [
        {
            "step": 0,
            "ranges": (
                (0.0, 0.0),
                DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE,
            ),
        },
        {
            "step": 300 * 24,
            "ranges": (
                (-math.radians(15.0), math.radians(15.0)),
                DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE,
            ),
        },
        {
            "step": 600 * 24,
            "ranges": (
                (-math.radians(15.0), math.radians(15.0)),
                DRIBBLE_TARGET_MEDIUM_DISTANCE_RANGE,
            ),
        },
        {
            "step": 900 * 24,
            "ranges": (
                (-math.radians(30.0), math.radians(30.0)),
                DRIBBLE_TARGET_MEDIUM_DISTANCE_RANGE,
            ),
        },
        {
            "step": 1200 * 24,
            "ranges": (
                (-math.radians(30.0), math.radians(30.0)),
                DRIBBLE_TARGET_NOMINAL_DISTANCE_RANGE,
            ),
        },
        {
            "step": 1500 * 24,
            "ranges": (
                (-math.radians(60.0), math.radians(60.0)),
                DRIBBLE_TARGET_NOMINAL_DISTANCE_RANGE,
            ),
        },
        {
            "step": 2100 * 24,
            "ranges": (
                (-math.radians(60.0), math.radians(60.0)),
                DRIBBLE_TARGET_LONG_DISTANCE_RANGE,
            ),
        },
        {
            "step": 2700 * 24,
            "ranges": (
                (-math.radians(90.0), math.radians(90.0)),
                DRIBBLE_TARGET_LONG_DISTANCE_RANGE,
            ),
        },
        {
            "step": 3300 * 24,
            "ranges": (
                (-math.radians(90.0), math.radians(90.0)),
                DRIBBLE_TARGET_FINAL_DISTANCE_RANGE,
            ),
        },
    ]


def test_dribble_play_uses_final_target_distribution():
    cfg = make_microduck_ball_dribble_env_cfg(play=True)
    assert cfg.commands["body_pose"].ranges == (
        (-math.pi / 2.0, math.pi / 2.0),
        DRIBBLE_TARGET_FINAL_DISTANCE_RANGE,
    )
    assert "target_range" not in cfg.curriculum
    assert "push_robot" not in cfg.events
    assert "push_magnitude" not in cfg.curriculum


def test_dribble_push_curriculum_stays_at_the_validated_magnitude():
    cfg = make_microduck_ball_dribble_env_cfg()

    assert cfg.curriculum["push_magnitude"].params["push_stages"] == [
        {
            "step": 0,
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
        },
        {
            "step": 2000 * 24,
            "velocity_range": {
                "x": (-0.08, 0.08),
                "y": (-0.08, 0.08),
            },
        },
    ]


def test_dribble_metrics_reference_the_directional_state_term():
    cfg = make_microduck_ball_dribble_env_cfg()
    state_metrics = {
        "valid_kick_count",
        "valid_kick_left_count",
        "valid_kick_right_count",
        "wrong_foot_contact_count",
        "nonfoot_contact_count",
        "contact_tie_count",
    }
    assert all(
        cfg.metrics[name].params["reward_name"] == "ball_target_progress"
        for name in state_metrics
    )
    assert cfg.metrics["target_progress"].func is mdp.ball_target_progress
    assert cfg.metrics["remaining_target_distance"].func is mdp.ball_target_distance
    assert cfg.metrics["target_success"].func is mdp.ball_target_success
    assert cfg.metrics["cross_track_error"].reduce == "mean"
    for label in ("00_15", "15_30", "30_60", "60_90"):
        assert (
            cfg.metrics[f"target_angle_{label}_mass"].func
            is mdp.ball_target_angle_bin_mass
        )
        assert (
            cfg.metrics[f"target_success_angle_{label}_mass"].func
            is mdp.ball_target_success_angle_bin_mass
        )


def test_dribble_preserves_61d_layout_and_old_dual_behavior():
    dual = make_microduck_ball_kick_dual_env_cfg()
    dribble = make_microduck_ball_dribble_env_cfg()
    assert list(dribble.observations["actor"].terms) == list(
        dual.observations["actor"].terms
    )
    assert "effective_kick" in dual.rewards
    assert dual.rewards["effective_kick"].weight == 25.0
    assert MicroduckBallDribbleRlCfg.algorithm.symmetry_cfg["use_mirror_loss"] is True
    assert MicroduckBallDribbleRlCfg.experiment_name == "ball_dribble"
    assert MicroduckBallDribbleRlCfg.max_iterations == 4_200


def test_only_base_dribble_task_is_registered():
    tasks = list_tasks()
    assert "Mjlab-BallDribble-Flat-MicroDuck" in tasks
    assert "Mjlab-BallDribble-Flat-Backlash-MicroDuck" not in tasks
