import mujoco

from mjlab.tasks.registry import list_tasks

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_XML
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    BALL_FIELD_TEXTURE,
    BALL_OFFSET_ABS_Y,
    BALL_OFFSET_X,
    BALL_POS_NOISE_XY,
    BALL_TARGET_SPEED,
    DUAL_BALL_TARGET_SPEED,
    DUAL_COMMAND_LATERAL_SCALE,
    DUAL_COMMAND_LONGITUDINAL_SCALE,
    DUAL_COMMAND_TARGET_DISTANCE,
    DUAL_CONTACT_CLEAR_STEPS,
    DUAL_EPISODE_LENGTH_S,
    DUAL_MIN_SPEED_GAIN,
    DUAL_REWARD_LATERAL_STD,
    DUAL_REWARD_LONGITUDINAL_STD,
    DUAL_SUCCESS_DISTANCE,
    DUAL_SUCCESS_KICKS,
    MicroduckBallKickDualRlCfg,
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_dual_env_cfg,
    make_microduck_ball_kick_env_cfg,
)


def _sensor(cfg, name):
    return next(sensor for sensor in cfg.scene.sensors if sensor.name == name)


def test_ball_tasks_use_green_field_and_soccer_ball_without_physics_changes():
    cfg = make_microduck_ball_kick_dual_env_cfg()
    texture = cfg.scene.terrain.textures[0]
    material = cfg.scene.terrain.materials[0]
    assert texture is BALL_FIELD_TEXTURE
    assert material.texture == "groundplane"
    assert material.texrepeat == (1.0, 1.0)
    assert material.reflectance == 0.0

    model = mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML)).compile()
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    material_id = model.geom_matid[geom_id]
    assert model.ntex == 1
    assert model.geom_size[geom_id, 0] == 0.035
    assert tuple(model.geom_friction[geom_id]) == (0.5, 0.005, 0.0001)
    assert model.body_mass[body_id] == 0.015
    assert material_id >= 0
    assert model.mat_texuniform[material_id] == 0
    assert 0 in model.mat_texid[material_id]


def test_fixed_foot_ball_kick_behavior_is_unchanged():
    right = make_microduck_ball_kick_env_cfg(kick_foot="right")
    left = make_microduck_ball_kick_env_cfg(kick_foot="left")

    assert right.events["reset_ball"].func is mdp.reset_ball_in_front_of_foot
    assert right.events["reset_ball"].params == {
        "offset": (BALL_OFFSET_X, -BALL_OFFSET_ABS_Y),
        "noise_xy": BALL_POS_NOISE_XY,
        "ball_radius": 0.035,
        "asset_name": "ball",
    }
    assert left.events["reset_ball"].params["offset"] == (
        BALL_OFFSET_X,
        BALL_OFFSET_ABS_Y,
    )
    assert _sensor(right, "support_foot_ground_contact").primary.pattern == (
        r"^left_foot_collision$"
    )
    assert _sensor(left, "support_foot_ground_contact").primary.pattern == (
        r"^right_foot_collision$"
    )
    assert isinstance(right.commands["twist"], mdp.VelocityCommandCommandOnlyCfg)
    assert right.rewards["ball_forward_velocity"].func is mdp.ball_forward_velocity
    assert right.rewards["ball_forward_velocity"].weight == 12.0
    assert right.rewards["ball_speed_overshoot"].params["target_speed"] == BALL_TARGET_SPEED
    assert right.rewards["support_foot_grounded"].weight == 2.0
    assert "invalid_contact" not in right.rewards
    assert MicroduckBallKickRlCfg.algorithm.symmetry_cfg is None


def test_dual_cfg_wires_same_ball_contact_cycle_and_guidance():
    cfg = make_microduck_ball_kick_dual_env_cfg()
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    command = cfg.commands["twist"]

    assert cfg.episode_length_s == DUAL_EPISODE_LENGTH_S == 8.0
    assert isinstance(command, mdp.BallKickSideCommandCfg)
    assert command.selection_margin == 0.010
    assert command.target_distance == DUAL_COMMAND_TARGET_DISTANCE
    assert command.longitudinal_scale == DUAL_COMMAND_LONGITUDINAL_SCALE
    assert command.lateral_scale == DUAL_COMMAND_LATERAL_SCALE
    assert "support_foot_ground_contact" not in sensors
    assert sensors["feet_ball_contact"].history_length == cfg.decimation
    assert sensors["nonfoot_ball_contact"].history_length == cfg.decimation
    assert sensors["nonfoot_ball_contact"].primary.mode == "body"
    assert sensors["nonfoot_ball_contact"].primary.pattern == r".+"
    assert sensors["nonfoot_ball_contact"].primary.exclude == (
        "ankle_left",
        "ankle_right",
    )
    assert all(event.mode != "interval" or name != "reset_ball" for name, event in cfg.events.items())

    reward = cfg.rewards["ball_forward_velocity"]
    assert reward.func is mdp.continuous_ball_dribble
    assert reward.weight == 4.0
    assert reward.params == {
        "command_name": "twist",
        "feet_sensor_name": "feet_ball_contact",
        "nonfoot_sensor_name": "nonfoot_ball_contact",
        "asset_name": "ball",
        "max_speed": DUAL_BALL_TARGET_SPEED,
        "target_distance": DUAL_COMMAND_TARGET_DISTANCE,
        "longitudinal_std": DUAL_REWARD_LONGITUDINAL_STD,
        "lateral_std": DUAL_REWARD_LATERAL_STD,
        "clear_steps": DUAL_CONTACT_CLEAR_STEPS,
        "min_speed_gain": DUAL_MIN_SPEED_GAIN,
    }
    assert cfg.rewards["ball_speed_overshoot"].weight == -4.0
    assert (
        cfg.rewards["ball_speed_overshoot"].params["target_speed"]
        == DUAL_BALL_TARGET_SPEED
    )
    assert cfg.rewards["support_foot_grounded"].weight == 1.0
    assert cfg.rewards["pose_stand_legs"].weight == 0.5
    assert cfg.rewards["invalid_contact"].func is mdp.invalid_ball_contact_cost
    assert cfg.rewards["invalid_contact"].weight == -2.0
    assert cfg.rewards["effective_kick"].func is mdp.effective_ball_kick_reward
    assert cfg.rewards["effective_kick"].weight == 25.0
    assert cfg.terminations["ball_lost"].func is mdp.ball_is_lost


def test_dual_curricula_delay_motion_taxes_and_disturbances():
    cfg = make_microduck_ball_kick_dual_env_cfg()
    action_stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert action_stages == [
        {"step": 0, "weight": -0.05},
        {"step": 1500 * 24, "weight": -0.1},
        {"step": 2000 * 24, "weight": -0.2},
        {"step": 2500 * 24, "weight": -0.4},
        {"step": 3000 * 24, "weight": -0.6},
    ]
    push_stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert push_stages[0]["velocity_range"] == {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    assert push_stages[1]["step"] == 2000 * 24
    assert push_stages[2]["step"] == 3000 * 24


def test_dual_ball_position_curriculum_matches_iteration_boundaries():
    cfg = make_microduck_ball_kick_dual_env_cfg()
    reset = cfg.events["reset_ball"]
    assert reset.func is mdp.reset_ball_for_dual_kick
    assert reset.params["distribution"] == "lobes"
    assert reset.params["offset_x"] == 0.09
    assert reset.params["offset_abs_y"] == 0.042
    assert reset.params["noise_x"] == 0.015
    assert reset.params["noise_y"] == 0.015
    assert reset.params["x_range"] == (0.075, 0.105)
    assert reset.params["y_range"] == (-0.070, 0.070)

    stages = cfg.curriculum["ball_position"].params["param_stages"]
    assert stages == [
        {
            "step": 0,
            "params": {
                "distribution": "lobes",
                "noise_x": 0.015,
                "noise_y": 0.015,
            },
        },
        {
            "step": 2000 * 24,
            "params": {
                "distribution": "lobes",
                "noise_x": 0.015,
                "noise_y": 0.028,
            },
        },
        {"step": 3000 * 24, "params": {"distribution": "continuous"}},
    ]


def test_dual_play_uses_final_continuous_region_without_curriculum():
    cfg = make_microduck_ball_kick_dual_env_cfg(play=True)
    assert cfg.events["reset_ball"].params["distribution"] == "continuous"
    assert "ball_position" not in cfg.curriculum
    assert "push_magnitude" not in cfg.curriculum
    assert "push_robot" not in cfg.events
    assert cfg.scene.env_spacing == 0.40
    assert cfg.viewer.distance == 2.5
    assert cfg.viewer.elevation == -35.0
    assert cfg.viewer.azimuth == 120.0
    assert cfg.viewer.max_extra_envs == 7


def test_dual_metrics_use_last_reduction_for_cumulative_state():
    cfg = make_microduck_ball_kick_dual_env_cfg()
    expected = {
        "selected_left",
        "selected_right",
        "valid_kick_count",
        "valid_kick_left_count",
        "valid_kick_right_count",
        "wrong_foot_contact_count",
        "nonfoot_contact_count",
        "contact_tie_count",
        "ball_forward_distance",
        "robot_ball_distance",
        "fell_over",
        "ball_lost",
        "continuous_success",
    }
    assert expected <= cfg.metrics.keys()
    assert all(cfg.metrics[name].reduce == "last" for name in expected)
    assert cfg.metrics["continuous_success"].params == {
        "reward_name": "ball_forward_velocity",
        "min_kicks": DUAL_SUCCESS_KICKS,
        "min_distance": DUAL_SUCCESS_DISTANCE,
    }


def test_dual_preserves_61d_actor_layout_and_enables_mirror_loss():
    fixed = make_microduck_ball_kick_env_cfg()
    dual = make_microduck_ball_kick_dual_env_cfg()
    assert list(dual.observations["actor"].terms) == list(
        fixed.observations["actor"].terms
    )
    assert "ball_position" not in dual.observations["actor"].terms
    assert "ball_velocity" not in dual.observations["actor"].terms
    assert dual.observations["actor"].terms["head_command"].params["dim"] == 4
    assert dual.observations["actor"].terms["body_command"].params["dim"] == 6
    assert MicroduckBallKickDualRlCfg.algorithm.symmetry_cfg["use_mirror_loss"] is True
    assert MicroduckBallKickDualRlCfg.experiment_name == "ball_kick_dual_continuous"


def test_only_base_dual_task_is_registered():
    tasks = list_tasks()
    assert "Mjlab-BallKickDual-Flat-MicroDuck" in tasks
    assert "Mjlab-BallKickDual-Flat-Backlash-MicroDuck" not in tasks
