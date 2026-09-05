"""Directional same-ball dribbling for Microduck.

The policy keeps the BallKickDual ball-relative twist command and receives a
second command in the existing 6D body slot: the live ball-to-target direction
and remaining distance.  Contact count is diagnostic only; the task reward is
best-so-far progress toward the target.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import observations as obs_mdp
from mjlab.managers import (
    CurriculumTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    DUAL_BALL_TARGET_SPEED,
    DUAL_COMMAND_TARGET_DISTANCE,
    DUAL_CONTACT_CLEAR_STEPS,
    DUAL_MIN_SPEED_GAIN,
    DUAL_REWARD_LATERAL_STD,
    DUAL_REWARD_LONGITUDINAL_STD,
    MicroduckBallKickDualRlCfg,
    make_microduck_ball_kick_dual_env_cfg,
)

DRIBBLE_EPISODE_LENGTH_S = 10.0
DRIBBLE_GOAL_RADIUS = 0.15
DRIBBLE_CONTROL_RADIUS = 0.30
DRIBBLE_PROGRESS_PER_KICK = 0.15
DRIBBLE_TARGET_DISTANCE_SCALE = 2.0
DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE = (0.35, 0.50)
DRIBBLE_TARGET_MEDIUM_DISTANCE_RANGE = (0.55, 0.75)
DRIBBLE_TARGET_NOMINAL_DISTANCE_RANGE = (0.80, 1.10)
DRIBBLE_TARGET_LONG_DISTANCE_RANGE = (1.0, 1.3)
DRIBBLE_TARGET_FINAL_DISTANCE_RANGE = (1.0, 1.5)
DRIBBLE_PROGRESS_REWARD_WEIGHT = 12.0
DRIBBLE_TILT_COST_WEIGHT = -5.0
DRIBBLE_UPRIGHT_REWARD_WEIGHT = 0.5


def _target_ranges(
    max_angle_deg: float,
    distance_range: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    max_angle = math.radians(max_angle_deg)
    return ((-max_angle, max_angle), distance_range)


def make_microduck_ball_dribble_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create target-conditioned continuous dribbling on flat terrain."""
    cfg = make_microduck_ball_kick_dual_env_cfg(play=play)
    cfg.episode_length_s = DRIBBLE_EPISODE_LENGTH_S

    command_duration = (
        DRIBBLE_EPISODE_LENGTH_S * 2,
        DRIBBLE_EPISODE_LENGTH_S * 2,
    )
    cfg.commands["twist"].resampling_time_range = command_duration
    cfg.commands["body_pose"] = microduck_mdp.BallTargetCommandCfg(
        resampling_time_range=command_duration,
        debug_vis=False,
        ranges=_target_ranges(
            90.0 if play else 0.0,
            DRIBBLE_TARGET_FINAL_DISTANCE_RANGE
            if play
            else DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE,
        ),
        distance_scale=DRIBBLE_TARGET_DISTANCE_SCALE,
        goal_radius=DRIBBLE_GOAL_RADIUS,
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=obs_mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    feet_sensor_name = "feet_ball_contact"
    nonfoot_sensor_name = "nonfoot_ball_contact"
    progress_reward = RewardTermCfg(
        func=microduck_mdp.directional_ball_dribble,
        weight=DRIBBLE_PROGRESS_REWARD_WEIGHT,
        params={
            "command_name": "twist",
            "target_command_name": "body_pose",
            "feet_sensor_name": feet_sensor_name,
            "nonfoot_sensor_name": nonfoot_sensor_name,
            "asset_name": "ball",
            "max_speed": DUAL_BALL_TARGET_SPEED,
            "target_distance": DUAL_COMMAND_TARGET_DISTANCE,
            "longitudinal_std": DUAL_REWARD_LONGITUDINAL_STD,
            "lateral_std": DUAL_REWARD_LATERAL_STD,
            "clear_steps": DUAL_CONTACT_CLEAR_STEPS,
            "min_speed_gain": DUAL_MIN_SPEED_GAIN,
            "max_control_distance": DRIBBLE_CONTROL_RADIUS,
            "progress_credit_per_kick": DRIBBLE_PROGRESS_PER_KICK,
        },
    )
    rewards = {}
    for name, term in cfg.rewards.items():
        if name == "ball_forward_velocity":
            rewards["ball_target_progress"] = progress_reward
        elif name != "effective_kick":
            rewards[name] = term
    cfg.rewards = rewards
    for reward_name in (
        "support_foot_grounded",
        "pose_stand_legs",
        "pose_stand_neck",
        "height_stand",
    ):
        cfg.rewards.pop(reward_name)
    cfg.rewards["upright"].weight = DRIBBLE_UPRIGHT_REWARD_WEIGHT
    cfg.rewards["body_tilt"] = RewardTermCfg(
        func=base_mdp.flat_orientation_l2,
        weight=DRIBBLE_TILT_COST_WEIGHT,
    )
    cfg.rewards[
        "ball_speed_overshoot"
    ].func = microduck_mdp.ball_target_speed_overshoot_cost
    cfg.rewards["ball_speed_overshoot"].params = {
        "target_command_name": "body_pose",
        "asset_name": "ball",
        "target_speed": DUAL_BALL_TARGET_SPEED,
    }
    cfg.rewards["invalid_contact"].params = {"reward_name": "ball_target_progress"}
    cfg.terminations["target_reached"] = TerminationTermCfg(
        func=microduck_mdp.ball_target_reached,
        time_out=True,
        params={"command_name": "body_pose"},
    )

    contact_metrics = (
        "valid_kick_count",
        "valid_kick_left_count",
        "valid_kick_right_count",
        "wrong_foot_contact_count",
        "nonfoot_contact_count",
        "contact_tie_count",
    )
    for name in contact_metrics:
        cfg.metrics[name].params = {"reward_name": "ball_target_progress"}
    cfg.metrics.pop("ball_forward_distance")
    cfg.metrics.pop("continuous_success")
    cfg.metrics.update(
        {
            "target_progress": MetricsTermCfg(
                func=microduck_mdp.ball_target_progress,
                params={"command_name": "body_pose"},
                reduce="last",
            ),
            "remaining_target_distance": MetricsTermCfg(
                func=microduck_mdp.ball_target_distance,
                params={"command_name": "body_pose"},
                reduce="last",
            ),
            "cross_track_error": MetricsTermCfg(
                func=microduck_mdp.ball_target_cross_track_error,
                params={"command_name": "body_pose"},
                reduce="mean",
            ),
            "target_success": MetricsTermCfg(
                func=microduck_mdp.ball_target_success,
                params={"command_name": "body_pose"},
                reduce="last",
            ),
        }
    )
    for label, min_angle_deg, max_angle_deg in (
        ("00_15", 0.0, 15.0),
        ("15_30", 15.0, 30.0),
        ("30_60", 30.0, 60.0),
        ("60_90", 60.0, 90.0),
    ):
        params = {
            "command_name": "body_pose",
            "min_abs_angle": math.radians(min_angle_deg),
            "max_abs_angle": math.radians(max_angle_deg),
        }
        cfg.metrics[f"target_angle_{label}_mass"] = MetricsTermCfg(
            func=microduck_mdp.ball_target_angle_bin_mass,
            params=params,
            reduce="last",
        )
        cfg.metrics[f"target_success_angle_{label}_mass"] = MetricsTermCfg(
            func=microduck_mdp.ball_target_success_angle_bin_mass,
            params=params,
            reduce="last",
        )

    if play:
        cfg.curriculum.pop("target_range", None)
    else:
        cfg.curriculum["target_range"] = CurriculumTermCfg(
            func=microduck_mdp.pose_command_range_curriculum,
            params={
                "command_name": "body_pose",
                "range_stages": [
                    {
                        "step": 0,
                        "ranges": _target_ranges(
                            0.0, DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 300 * 24,
                        "ranges": _target_ranges(
                            15.0, DRIBBLE_TARGET_INITIAL_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 600 * 24,
                        "ranges": _target_ranges(
                            15.0, DRIBBLE_TARGET_MEDIUM_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 900 * 24,
                        "ranges": _target_ranges(
                            30.0, DRIBBLE_TARGET_MEDIUM_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 1200 * 24,
                        "ranges": _target_ranges(
                            30.0, DRIBBLE_TARGET_NOMINAL_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 1500 * 24,
                        "ranges": _target_ranges(
                            60.0, DRIBBLE_TARGET_NOMINAL_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 2100 * 24,
                        "ranges": _target_ranges(
                            60.0, DRIBBLE_TARGET_LONG_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 2700 * 24,
                        "ranges": _target_ranges(
                            90.0, DRIBBLE_TARGET_LONG_DISTANCE_RANGE
                        ),
                    },
                    {
                        "step": 3300 * 24,
                        "ranges": _target_ranges(
                            90.0, DRIBBLE_TARGET_FINAL_DISTANCE_RANGE
                        ),
                    },
                ],
            },
        )
        cfg.curriculum["push_magnitude"].params["push_stages"] = [
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

    return cfg


MicroduckBallDribbleRlCfg = deepcopy(MicroduckBallKickDualRlCfg)
MicroduckBallDribbleRlCfg.experiment_name = "ball_dribble"
MicroduckBallDribbleRlCfg.run_name = "ball_dribble"
MicroduckBallDribbleRlCfg.max_iterations = 4_200
