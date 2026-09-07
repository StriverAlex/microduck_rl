"""Sequential three-marker ball slalom for Microduck.

The actor keeps the BallDribble 61D observation contract.  Its body command
contains the ball-to-waypoint direction, the robot-to-ball position, and the
sign of the next turn.  The physical markers remain collision-enabled and
alternate to the left and right of the desired ball path.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers import (
    CurriculumTermCfg,
    MetricsTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_SLALOM_COURSE_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_dribble_env_cfg import (
    DRIBBLE_CONTROL_RADIUS,
    DRIBBLE_GOAL_RADIUS,
    DRIBBLE_TARGET_DISTANCE_SCALE,
    MicroduckBallDribbleRlCfg,
    make_microduck_ball_dribble_env_cfg,
)

SLALOM_EPISODE_LENGTH_S = 15.0
SLALOM_CONE_X = (0.45, 0.90, 1.35)
SLALOM_INITIAL_LATERAL_OFFSET = 0.08
SLALOM_SMALL_LATERAL_OFFSET = 0.10
SLALOM_INTERMEDIATE_LATERAL_OFFSET = 0.12
SLALOM_LARGE_LATERAL_OFFSET = 0.14
SLALOM_FINAL_LATERAL_OFFSET = 0.16
SLALOM_WAYPOINT_CLEARANCE = 0.04
SLALOM_ROUTE_LATERAL_MARGIN = 0.06
SLALOM_PREVIEW_DISTANCE = 0.40
SLALOM_GOAL_RADIUS = 0.06
SLALOM_INITIAL_GOAL_RADIUS = DRIBBLE_GOAL_RADIUS
SLALOM_MEDIUM_GOAL_RADIUS = 0.10
SLALOM_DISTANCE_SCALE = DRIBBLE_TARGET_DISTANCE_SCALE
SLALOM_OBSTACLE_COST_WEIGHT = -1.0
SLALOM_CONTROL_DISTANCE_COST_WEIGHT = -5.0
SLALOM_MIN_BALL_FORWARD = -DRIBBLE_CONTROL_RADIUS
SLALOM_INITIAL_TERMINATION_COST_WEIGHT = -300.0
SLALOM_TERMINATION_COST_WEIGHT = -500.0
SLALOM_SUCCESS_REWARD_WEIGHT = 100.0
SLALOM_INITIAL_PROGRESS_REWARD_WEIGHT = 40.0
SLALOM_PROGRESS_REWARD_WEIGHT = 24.0
SLALOM_ENTROPY_COEF = 0.001
SLALOM_LEARNING_RATE = 1.0e-4
SLALOM_COM_RANDOMIZATION_RANGE = 0.015
SLALOM_HEAD_COM_RANDOMIZATION_RANGE = 0.01
SLALOM_PUSH_RANGE = 0.08


def make_microduck_ball_slalom_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create ordered same-ball dribbling around three physical markers."""
    cfg = make_microduck_ball_dribble_env_cfg(play=play)
    cfg.episode_length_s = SLALOM_EPISODE_LENGTH_S
    cfg.scene.entities = {
        **cfg.scene.entities,
        "slalom_course": MICRODUCK_SLALOM_COURSE_CFG,
    }
    cfg.sim.nconmax = 60

    ball_marker_contact = ContactSensorCfg(
        name="ball_marker_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^slalom_cone_[123]$",
            entity="slalom_course",
        ),
        secondary=ContactMatch(mode="body", pattern=r"^ball$", entity="ball"),
        fields=("found",),
        reduce="none",
        num_slots=1,
        history_length=cfg.decimation,
    )
    robot_marker_contact = ContactSensorCfg(
        name="robot_marker_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^slalom_cone_[123]$",
            entity="slalom_course",
        ),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
        history_length=cfg.decimation,
    )
    cfg.scene.sensors = cfg.scene.sensors + (
        ball_marker_contact,
        robot_marker_contact,
    )

    command_duration = (
        SLALOM_EPISODE_LENGTH_S * 2,
        SLALOM_EPISODE_LENGTH_S * 2,
    )
    cfg.commands["twist"].resampling_time_range = command_duration
    cfg.commands["body_pose"] = microduck_mdp.BallSlalomCommandCfg(
        resampling_time_range=command_duration,
        debug_vis=False,
        cone_x=SLALOM_CONE_X,
        preview_distance=SLALOM_PREVIEW_DISTANCE,
        lateral_offset=(
            SLALOM_FINAL_LATERAL_OFFSET if play else SLALOM_INITIAL_LATERAL_OFFSET
        ),
        waypoint_clearance=SLALOM_WAYPOINT_CLEARANCE,
        route_lateral_margin=SLALOM_ROUTE_LATERAL_MARGIN,
        ball_position_scale=DRIBBLE_CONTROL_RADIUS,
        distance_scale=SLALOM_DISTANCE_SCALE,
        goal_radius=SLALOM_GOAL_RADIUS if play else SLALOM_INITIAL_GOAL_RADIUS,
    )

    cfg.rewards["obstacle_contact"] = RewardTermCfg(
        func=microduck_mdp.slalom_obstacle_contact_cost,
        weight=SLALOM_OBSTACLE_COST_WEIGHT,
        params={
            "ball_sensor_name": ball_marker_contact.name,
            "robot_sensor_name": robot_marker_contact.name,
        },
    )
    cfg.terminations["obstacle_contact"] = TerminationTermCfg(
        func=microduck_mdp.slalom_obstacle_contact,
        params={
            "ball_sensor_name": ball_marker_contact.name,
            "robot_sensor_name": robot_marker_contact.name,
        },
    )
    cfg.terminations["invalid_route"] = TerminationTermCfg(
        func=microduck_mdp.ball_slalom_invalid_route,
        params={"command_name": "body_pose"},
    )
    cfg.rewards["ball_control_distance"] = RewardTermCfg(
        func=microduck_mdp.ball_control_distance_cost,
        weight=SLALOM_CONTROL_DISTANCE_COST_WEIGHT,
        params={
            "asset_name": "ball",
            "control_distance": DRIBBLE_CONTROL_RADIUS,
        },
    )
    cfg.terminations["ball_lost"].params["min_forward"] = SLALOM_MIN_BALL_FORWARD
    cfg.rewards["termination"] = RewardTermCfg(
        func=base_mdp.is_terminated,
        weight=(
            SLALOM_TERMINATION_COST_WEIGHT
            if play
            else SLALOM_INITIAL_TERMINATION_COST_WEIGHT
        ),
    )
    cfg.rewards["course_success"] = RewardTermCfg(
        func=microduck_mdp.ball_slalom_success,
        weight=SLALOM_SUCCESS_REWARD_WEIGHT,
        params={"command_name": "body_pose"},
    )
    if not play:
        cfg.rewards[
            "ball_target_progress"
        ].weight = SLALOM_INITIAL_PROGRESS_REWARD_WEIGHT
    cfg.terminations["target_reached"] = TerminationTermCfg(
        func=microduck_mdp.ball_slalom_reached,
        time_out=True,
        params={"command_name": "body_pose"},
    )

    for name in (
        "target_progress",
        "remaining_target_distance",
        "cross_track_error",
        "target_success",
        "target_angle_00_15_mass",
        "target_success_angle_00_15_mass",
        "target_angle_15_30_mass",
        "target_success_angle_15_30_mass",
        "target_angle_30_60_mass",
        "target_success_angle_30_60_mass",
        "target_angle_60_90_mass",
        "target_success_angle_60_90_mass",
    ):
        cfg.metrics.pop(name)
    cfg.metrics.update(
        {
            "waypoints_completed": MetricsTermCfg(
                func=microduck_mdp.ball_slalom_waypoints_completed,
                params={"command_name": "body_pose"},
                reduce="last",
            ),
            "course_completion": MetricsTermCfg(
                func=microduck_mdp.ball_slalom_course_completion,
                params={"command_name": "body_pose"},
                reduce="last",
            ),
            "course_success": MetricsTermCfg(
                func=microduck_mdp.ball_slalom_success,
                params={"command_name": "body_pose"},
                reduce="last",
            ),
            "obstacle_contact_rate": MetricsTermCfg(
                func=microduck_mdp.slalom_obstacle_contact_cost,
                params={
                    "ball_sensor_name": ball_marker_contact.name,
                    "robot_sensor_name": robot_marker_contact.name,
                },
                reduce="mean",
            ),
        }
    )
    for side in ("left", "right"):
        cfg.metrics[f"course_{side}_mass"] = MetricsTermCfg(
            func=microduck_mdp.ball_slalom_side_mass,
            params={"command_name": "body_pose", "side": side},
            reduce="last",
        )
        cfg.metrics[f"course_success_{side}_mass"] = MetricsTermCfg(
            func=microduck_mdp.ball_slalom_success_side_mass,
            params={"command_name": "body_pose", "side": side},
            reduce="last",
        )
    cfg.curriculum.pop("target_range", None)
    if not play:
        cfg.curriculum.pop("com_range")
        cfg.curriculum.pop("head_com_range")
        cfg.curriculum.pop("ball_position")
        cfg.curriculum.pop("push_magnitude")
        cfg.events["randomize_com"].params["ranges"] = (
            -SLALOM_COM_RANDOMIZATION_RANGE,
            SLALOM_COM_RANDOMIZATION_RANGE,
        )
        cfg.events["randomize_head_com"].params["ranges"] = (
            -SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
            SLALOM_HEAD_COM_RANDOMIZATION_RANGE,
        )
        cfg.events["reset_ball"].params["distribution"] = "continuous"
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
            "y": (-SLALOM_PUSH_RANGE, SLALOM_PUSH_RANGE),
        }
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.05},
                ],
            },
        )
        cfg.curriculum["slalom_course"] = CurriculumTermCfg(
            func=microduck_mdp.slalom_course_curriculum,
            params={
                "command_name": "body_pose",
                "stages": [
                    {
                        "step": 0,
                        "lateral_offset": SLALOM_INITIAL_LATERAL_OFFSET,
                        "goal_radius": SLALOM_INITIAL_GOAL_RADIUS,
                    },
                    {
                        "step": 1200 * 24,
                        "lateral_offset": SLALOM_SMALL_LATERAL_OFFSET,
                        "goal_radius": SLALOM_INITIAL_GOAL_RADIUS,
                    },
                    {
                        "step": 2200 * 24,
                        "lateral_offset": SLALOM_INTERMEDIATE_LATERAL_OFFSET,
                        "goal_radius": SLALOM_INITIAL_GOAL_RADIUS,
                    },
                    {
                        "step": 3200 * 24,
                        "lateral_offset": SLALOM_LARGE_LATERAL_OFFSET,
                        "goal_radius": SLALOM_INITIAL_GOAL_RADIUS,
                    },
                    {
                        "step": 4200 * 24,
                        "lateral_offset": SLALOM_FINAL_LATERAL_OFFSET,
                        "goal_radius": SLALOM_INITIAL_GOAL_RADIUS,
                    },
                    {
                        "step": 5000 * 24,
                        "lateral_offset": SLALOM_FINAL_LATERAL_OFFSET,
                        "goal_radius": SLALOM_MEDIUM_GOAL_RADIUS,
                    },
                    {
                        "step": 6000 * 24,
                        "lateral_offset": SLALOM_FINAL_LATERAL_OFFSET,
                        "goal_radius": SLALOM_GOAL_RADIUS,
                    },
                ],
            },
        )
        cfg.curriculum["slalom_progress_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "ball_target_progress",
                "weight_stages": [
                    {
                        "step": 0,
                        "weight": SLALOM_INITIAL_PROGRESS_REWARD_WEIGHT,
                    },
                    {
                        "step": 6000 * 24,
                        "weight": SLALOM_PROGRESS_REWARD_WEIGHT,
                    },
                ],
            },
        )
        cfg.curriculum["slalom_termination_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "termination",
                "weight_stages": [
                    {
                        "step": 0,
                        "weight": SLALOM_INITIAL_TERMINATION_COST_WEIGHT,
                    },
                    {
                        "step": 7000 * 24,
                        "weight": SLALOM_TERMINATION_COST_WEIGHT,
                    },
                ],
            },
        )

    return cfg


MicroduckBallSlalomRlCfg = deepcopy(MicroduckBallDribbleRlCfg)
MicroduckBallSlalomRlCfg.experiment_name = "ball_slalom"
MicroduckBallSlalomRlCfg.run_name = "ball_slalom"
MicroduckBallSlalomRlCfg.max_iterations = 7_500
MicroduckBallSlalomRlCfg.algorithm.entropy_coef = SLALOM_ENTROPY_COEF
MicroduckBallSlalomRlCfg.algorithm.learning_rate = SLALOM_LEARNING_RATE
MicroduckBallSlalomRlCfg.algorithm.schedule = "fixed"
