from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import observations as obs_mdp
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.robot.microduck_constants import MICRODUCK_SLALOM_COURSE_CFG
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_ball_dribble_env_cfg import (
    DRIBBLE_CONTROL_RADIUS,
    DRIBBLE_PROGRESS_REWARD_WEIGHT,
    DRIBBLE_TARGET_DISTANCE_SCALE,
    make_microduck_ball_dribble_env_cfg,
)
from mjlab_microduck.tasks.microduck_ball_slalom_env_cfg import (
    SLALOM_COM_RANDOMIZATION_RANGE,
    SLALOM_CONE_X,
    SLALOM_CONTROL_DISTANCE_COST_WEIGHT,
    SLALOM_ENTROPY_COEF,
    SLALOM_EPISODE_LENGTH_S,
    SLALOM_FINAL_LATERAL_OFFSET,
    SLALOM_GOAL_RADIUS,
    SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
    SLALOM_INITIAL_GOAL_RADIUS,
    SLALOM_INITIAL_LATERAL_OFFSET,
    SLALOM_INITIAL_PROGRESS_REWARD_WEIGHT,
    SLALOM_INITIAL_TERMINATION_COST_WEIGHT,
    SLALOM_INTERMEDIATE_LATERAL_OFFSET,
    SLALOM_LARGE_LATERAL_OFFSET,
    SLALOM_LEARNING_RATE,
    SLALOM_MEDIUM_GOAL_RADIUS,
    SLALOM_MIN_BALL_FORWARD,
    SLALOM_OBSTACLE_COST_WEIGHT,
    SLALOM_PROGRESS_REWARD_WEIGHT,
    SLALOM_PUSH_RANGE,
    SLALOM_SMALL_LATERAL_OFFSET,
    SLALOM_SUCCESS_REWARD_WEIGHT,
    SLALOM_TERMINATION_COST_WEIGHT,
    SLALOM_WAYPOINT_CLEARANCE,
    MicroduckBallSlalomRlCfg,
    make_microduck_ball_slalom_env_cfg,
)


def test_slalom_course_asset_contains_three_physical_markers():
    course = MICRODUCK_SLALOM_COURSE_CFG.build()

    assert course.is_fixed_base
    assert course.is_mocap
    assert course.geom_names == (
        "slalom_cone_1",
        "slalom_cone_2",
        "slalom_cone_3",
    )
    assert (
        tuple(course.spec.geom(f"slalom_cone_{index}").pos[0] for index in range(1, 4))
        == SLALOM_CONE_X
    )


def test_slalom_preserves_dribble_observation_and_reward_contract():
    dribble = make_microduck_ball_dribble_env_cfg()
    cfg = make_microduck_ball_slalom_env_cfg()

    assert cfg.episode_length_s == SLALOM_EPISODE_LENGTH_S
    assert tuple(cfg.scene.entities) == ("robot", "ball", "slalom_course")
    assert list(cfg.observations["actor"].terms) == list(
        dribble.observations["actor"].terms
    )
    for group in ("actor", "critic"):
        body = cfg.observations[group].terms["body_command"]
        assert body.func is obs_mdp.generated_commands
        assert body.params == {"command_name": "body_pose"}
    assert cfg.rewards["ball_target_progress"].func is mdp.directional_ball_dribble
    assert "effective_kick" not in cfg.rewards


def test_slalom_command_and_curriculum_reach_the_complete_course():
    cfg = make_microduck_ball_slalom_env_cfg()
    command = cfg.commands["body_pose"]

    assert isinstance(command, mdp.BallSlalomCommandCfg)
    assert command.preview_distance == 0.25
    assert command.waypoint_clearance == SLALOM_WAYPOINT_CLEARANCE
    assert command.lateral_offset == SLALOM_INITIAL_LATERAL_OFFSET
    assert command.distance_scale == DRIBBLE_TARGET_DISTANCE_SCALE
    assert command.ball_position_scale == DRIBBLE_CONTROL_RADIUS
    assert command.goal_radius == SLALOM_INITIAL_GOAL_RADIUS
    stages = cfg.curriculum["slalom_course"].params["stages"]
    assert [
        (
            stage["step"],
            stage["lateral_offset"],
            stage["goal_radius"],
        )
        for stage in stages
    ] == [
        (0, SLALOM_INITIAL_LATERAL_OFFSET, SLALOM_INITIAL_GOAL_RADIUS),
        (1200 * 24, SLALOM_SMALL_LATERAL_OFFSET, SLALOM_INITIAL_GOAL_RADIUS),
        (2200 * 24, SLALOM_INTERMEDIATE_LATERAL_OFFSET, SLALOM_INITIAL_GOAL_RADIUS),
        (3200 * 24, SLALOM_LARGE_LATERAL_OFFSET, SLALOM_INITIAL_GOAL_RADIUS),
        (4200 * 24, SLALOM_FINAL_LATERAL_OFFSET, SLALOM_INITIAL_GOAL_RADIUS),
        (5000 * 24, SLALOM_FINAL_LATERAL_OFFSET, SLALOM_MEDIUM_GOAL_RADIUS),
    ]
    assert "target_range" not in cfg.curriculum
    assert "com_range" not in cfg.curriculum
    assert "head_com_range" not in cfg.curriculum
    assert "ball_position" not in cfg.curriculum
    assert "push_magnitude" not in cfg.curriculum
    assert cfg.events["randomize_com"].params["ranges"] == (
        -SLALOM_COM_RANDOMIZATION_RANGE,
        SLALOM_COM_RANDOMIZATION_RANGE,
    )
    assert cfg.events["randomize_head_com"].params["ranges"] == (
        -SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
        SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
    )
    assert cfg.events["reset_ball"].params["distribution"] == "continuous"
    assert cfg.events["push_robot"].params["velocity_range"] == {
        "x": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
        "y": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
    }
    assert cfg.rewards["ball_target_progress"].weight == (
        SLALOM_INITIAL_PROGRESS_REWARD_WEIGHT
    )
    assert cfg.curriculum["slalom_progress_weight"].params == {
        "reward_name": "ball_target_progress",
        "weight_stages": [
            {"step": 0, "weight": SLALOM_INITIAL_PROGRESS_REWARD_WEIGHT},
            {"step": 6000 * 24, "weight": SLALOM_PROGRESS_REWARD_WEIGHT},
        ],
    }
    assert cfg.curriculum["action_rate_weight"].params == {
        "reward_name": "action_rate_l2",
        "weight_stages": [
            {"step": 0, "weight": -0.05},
        ],
    }
    assert cfg.rewards["termination"].weight == (SLALOM_INITIAL_TERMINATION_COST_WEIGHT)
    assert cfg.curriculum["slalom_termination_weight"].params == {
        "reward_name": "termination",
        "weight_stages": [
            {"step": 0, "weight": SLALOM_INITIAL_TERMINATION_COST_WEIGHT},
            {"step": 7000 * 24, "weight": SLALOM_TERMINATION_COST_WEIGHT},
        ],
    }
    assert cfg.rewards["course_success"].weight == SLALOM_SUCCESS_REWARD_WEIGHT


def test_slalom_play_uses_three_cones_without_training_curricula_or_pushes():
    cfg = make_microduck_ball_slalom_env_cfg(play=True)
    command = cfg.commands["body_pose"]

    assert command.waypoint_clearance == SLALOM_WAYPOINT_CLEARANCE
    assert command.lateral_offset == SLALOM_FINAL_LATERAL_OFFSET
    assert "slalom_course" not in cfg.curriculum
    assert "slalom_progress_weight" not in cfg.curriculum
    assert "slalom_termination_weight" not in cfg.curriculum
    assert cfg.rewards["ball_target_progress"].weight == (
        DRIBBLE_PROGRESS_REWARD_WEIGHT
    )
    assert cfg.rewards["termination"].weight == SLALOM_TERMINATION_COST_WEIGHT
    assert cfg.rewards["course_success"].weight == SLALOM_SUCCESS_REWARD_WEIGHT
    assert "push_robot" not in cfg.events
    assert "push_magnitude" not in cfg.curriculum


def test_slalom_obstacle_contact_is_a_nonnegative_cost_with_negative_weight():
    cfg = make_microduck_ball_slalom_env_cfg()
    cost = cfg.rewards["obstacle_contact"]

    assert cost.func is mdp.slalom_obstacle_contact_cost
    assert cost.weight == SLALOM_OBSTACLE_COST_WEIGHT
    assert cost.weight < 0.0
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert {"ball_marker_contact", "robot_marker_contact"} <= sensor_names


def test_slalom_uses_official_failure_termination_cost():
    cfg = make_microduck_ball_slalom_env_cfg()
    cost = cfg.rewards["termination"]

    assert cost.func is base_mdp.is_terminated
    assert cost.weight == SLALOM_INITIAL_TERMINATION_COST_WEIGHT
    assert cost.weight < 0.0


def test_slalom_completion_reward_is_a_positive_terminal_event():
    cfg = make_microduck_ball_slalom_env_cfg()
    reward = cfg.rewards["course_success"]

    assert reward.func is mdp.ball_slalom_success
    assert reward.weight == SLALOM_SUCCESS_REWARD_WEIGHT
    assert reward.weight > 0.0
    assert cfg.terminations["target_reached"].time_out


def test_slalom_control_distance_uses_a_nonnegative_cost_with_negative_weight():
    cfg = make_microduck_ball_slalom_env_cfg()
    cost = cfg.rewards["ball_control_distance"]

    assert cost.func is mdp.ball_control_distance_cost
    assert cost.weight == SLALOM_CONTROL_DISTANCE_COST_WEIGHT
    assert cost.weight < 0.0
    assert cost.params == {
        "asset_name": "ball",
        "control_distance": DRIBBLE_CONTROL_RADIUS,
    }
    assert cfg.terminations["ball_lost"].params["min_forward"] == (
        SLALOM_MIN_BALL_FORWARD
    )
    assert SLALOM_MIN_BALL_FORWARD == -DRIBBLE_CONTROL_RADIUS


def test_slalom_termination_and_metrics_use_completed_course_state():
    cfg = make_microduck_ball_slalom_env_cfg()

    target_reached = cfg.terminations["target_reached"]
    assert target_reached.func is mdp.ball_slalom_reached
    assert target_reached.time_out
    assert cfg.metrics["waypoints_completed"].reduce == "last"
    assert cfg.metrics["course_completion"].reduce == "last"
    assert cfg.metrics["course_success"].reduce == "last"
    assert cfg.metrics["obstacle_contact_rate"].reduce == "mean"
    assert "target_success" not in cfg.metrics


def test_only_base_slalom_task_is_registered():
    tasks = list_tasks()
    assert "Mjlab-BallSlalom-Flat-MicroDuck" in tasks
    assert "Mjlab-BallSlalom-Flat-Backlash-MicroDuck" not in tasks
    assert MicroduckBallSlalomRlCfg.algorithm.symmetry_cfg["use_mirror_loss"] is True
    assert MicroduckBallSlalomRlCfg.experiment_name == "ball_slalom"
    assert MicroduckBallSlalomRlCfg.max_iterations == 7_500
    assert MicroduckBallSlalomRlCfg.algorithm.entropy_coef == SLALOM_ENTROPY_COEF
    assert MicroduckBallSlalomRlCfg.algorithm.learning_rate == SLALOM_LEARNING_RATE
    assert MicroduckBallSlalomRlCfg.algorithm.schedule == "fixed"
