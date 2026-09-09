"""Compare two BallSlalom checkpoints with fixed-seed first-episode rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_slalom_env_cfg import (
    SLALOM_FINAL_LATERAL_OFFSET,
    SLALOM_GOAL_RADIUS,
    SLALOM_ROUTE_LATERAL_MARGIN,
)

TASK_ID = "Mjlab-BallSlalom-Flat-MicroDuck"
STRICT_ROUTE_LATERAL_MARGIN = SLALOM_ROUTE_LATERAL_MARGIN


@dataclass(frozen=True)
class SlalomScenario:
    """Physical course geometry used for an evaluation-only rollout."""

    name: str
    cone_x: tuple[float, ...]
    lateral_offset: float
    episode_length_s: float

    def __post_init__(self) -> None:
        if len(self.cone_x) < 3:
            raise ValueError("a slalom scenario requires at least three cones")
        if self.cone_x[0] <= 0.0 or any(
            next_x <= current_x
            for current_x, next_x in zip(self.cone_x, self.cone_x[1:])
        ):
            raise ValueError("cone_x must contain positive increasing positions")
        if self.lateral_offset < STRICT_ROUTE_LATERAL_MARGIN:
            raise ValueError("lateral_offset must satisfy the strict route margin")
        if self.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")


GENERALIZATION_SCENARIOS = {
    "five_standard": SlalomScenario(
        name="five_standard",
        cone_x=(0.45, 0.90, 1.35, 1.80, 2.25),
        lateral_offset=0.16,
        episode_length_s=25.0,
    ),
    "five_tight": SlalomScenario(
        name="five_tight",
        cone_x=(0.38, 0.76, 1.14, 1.52, 1.90),
        lateral_offset=0.16,
        episode_length_s=25.0,
    ),
    "five_wide": SlalomScenario(
        name="five_wide",
        cone_x=(0.45, 0.90, 1.35, 1.80, 2.25),
        lateral_offset=0.20,
        episode_length_s=25.0,
    ),
    "five_irregular": SlalomScenario(
        name="five_irregular",
        cone_x=(0.42, 0.91, 1.29, 1.83, 2.22),
        lateral_offset=0.16,
        episode_length_s=25.0,
    ),
}


def _build_slalom_course_spec(cone_x: tuple[float, ...]) -> mujoco.MjSpec:
    geoms = "\n".join(
        f'<geom type="cylinder" name="slalom_cone_{index}" '
        f'pos="{x:.6f} 0 0.05" size="0.025 0.05" '
        'rgba="0.15 0.45 1 1" friction="0.8 0.005 0.0001"/>'
        for index, x in enumerate(cone_x, start=1)
    )
    return mujoco.MjSpec.from_string(
        '<mujoco model="microduck_slalom_course_eval">'
        '<compiler angle="radian" autolimits="true"/>'
        '<worldbody><body name="slalom_course">'
        f"{geoms}"
        "</body></worldbody></mujoco>"
    )


def configure_slalom_scenario(
    cfg: ManagerBasedRlEnvCfg,
    scenario: SlalomScenario,
) -> None:
    """Replace only the physical course and matching command geometry."""
    cfg.scene.entities["slalom_course"] = EntityCfg(
        spec_fn=partial(_build_slalom_course_spec, scenario.cone_x)
    )
    command = cfg.commands["body_pose"]
    command.cone_x = scenario.cone_x
    command.active_waypoints = len(scenario.cone_x)
    command.start_waypoint_probs = (1.0,) + (0.0,) * (
        len(scenario.cone_x) - 1
    )
    command.lateral_offset = scenario.lateral_offset
    cfg.episode_length_s = scenario.episode_length_s
    command_duration = (scenario.episode_length_s * 2,) * 2
    command.resampling_time_range = command_duration
    cfg.commands["twist"].resampling_time_range = command_duration

    marker_pattern = (
        r"^slalom_cone_(?:"
        + "|".join(str(index) for index in range(1, len(scenario.cone_x) + 1))
        + r")$"
    )
    for sensor in cfg.scene.sensors:
        if sensor.name in ("ball_marker_contact", "robot_marker_contact"):
            sensor.primary.pattern = marker_pattern


class StrictRouteTracker:
    """Track ordered, collision-free crossings of every cone plane."""

    def __init__(self, command, robot_xy: torch.Tensor, ball_xy: torch.Tensor):
        self.course_side = command.course_side.clone()
        self.cone_x = torch.as_tensor(
            command.cfg.cone_x, device=ball_xy.device, dtype=ball_xy.dtype
        )
        first_local_x = command.cfg.cone_x[0] + command.cfg.waypoint_clearance
        first_local_y = self.course_side * command.cfg.lateral_offset
        first_world = command._waypoint_pos_w[:, 0] - command._start_ball_pos_w
        denominator = first_local_x**2 + first_local_y**2
        cos_yaw = (
            first_world[:, 0] * first_local_x + first_world[:, 1] * first_local_y
        ) / denominator
        sin_yaw = (
            first_world[:, 1] * first_local_x - first_world[:, 0] * first_local_y
        ) / denominator
        self.forward_w = torch.stack((cos_yaw, sin_yaw), dim=1)
        self.lateral_w = torch.stack((-sin_yaw, cos_yaw), dim=1)
        self.origin_w = command._start_ball_pos_w.clone()
        self.ball_index = torch.zeros_like(command.waypoint_index)
        self.robot_index = torch.zeros_like(command.waypoint_index)
        self.ball_wrong_side = torch.zeros_like(command.completed, dtype=torch.bool)
        self.robot_wrong_side = torch.zeros_like(command.completed, dtype=torch.bool)
        self.ball_contact = torch.zeros_like(command.completed, dtype=torch.bool)
        self.robot_contact = torch.zeros_like(command.completed, dtype=torch.bool)
        self.previous_ball = self.course_coordinates(ball_xy)
        self.previous_robot = self.course_coordinates(robot_xy)

    def course_coordinates(self, position_w: torch.Tensor) -> torch.Tensor:
        offset = position_w - self.origin_w
        return torch.stack(
            (
                (offset * self.forward_w).sum(dim=1),
                (offset * self.lateral_w).sum(dim=1),
            ),
            dim=1,
        )

    def _update_entity(
        self,
        index: torch.Tensor,
        wrong_side: torch.Tensor,
        previous: torch.Tensor,
        current: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        incomplete = index < len(self.cone_x)
        lookup = index.clamp(max=len(self.cone_x) - 1)
        plane_x = self.cone_x[lookup]
        crossed = (
            active
            & incomplete
            & (previous[:, 0] < plane_x)
            & (current[:, 0] >= plane_x)
        )
        alternating = torch.where(
            lookup.remainder(2) == 0,
            torch.ones_like(self.course_side),
            -torch.ones_like(self.course_side),
        )
        correct_side = (
            self.course_side * alternating * current[:, 1]
            >= STRICT_ROUTE_LATERAL_MARGIN
        )
        wrong_side |= crossed & ~correct_side
        index += (crossed & correct_side).long()
        previous.copy_(current)

    def update(
        self,
        robot_xy: torch.Tensor,
        ball_xy: torch.Tensor,
        ball_contact: torch.Tensor,
        robot_contact: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        current_ball = self.course_coordinates(ball_xy)
        current_robot = self.course_coordinates(robot_xy)
        self._update_entity(
            self.ball_index,
            self.ball_wrong_side,
            self.previous_ball,
            current_ball,
            active,
        )
        self._update_entity(
            self.robot_index,
            self.robot_wrong_side,
            self.previous_robot,
            current_robot,
            active,
        )
        self.ball_contact |= active & ball_contact
        self.robot_contact |= active & robot_contact

    @property
    def completed(self) -> torch.Tensor:
        return (
            (self.ball_index == len(self.cone_x))
            & (self.robot_index == len(self.cone_x))
            & ~self.ball_wrong_side
            & ~self.robot_wrong_side
            & ~self.ball_contact
            & ~self.robot_contact
        )


@dataclass(frozen=True)
class RolloutResult:
    checkpoint: str
    episodes: int
    success_rate: float
    strict_success_rate: float
    completion_rate: float
    fall_rate: float
    ball_lost_rate: float
    obstacle_contact_termination_rate: float
    invalid_route_termination_rate: float
    left_success_rate: float
    right_success_rate: float
    strict_left_success_rate: float
    strict_right_success_rate: float
    ball_route_completion_rate: float
    robot_route_completion_rate: float
    ball_wrong_side_rate: float
    robot_wrong_side_rate: float
    ball_marker_contact_rate: float
    robot_marker_contact_rate: float
    obstacle_contact_waypoint_counts: tuple[int, ...]
    fall_waypoint_counts: tuple[int, ...]
    ball_lost_waypoint_counts: tuple[int, ...]
    ball_lost_backward_count: int
    ball_lost_distance_count: int
    ball_lost_lateral_count: int
    simultaneous_fall_and_ball_lost_count: int
    timeout_count: int
    mean_peak_target_speed: float
    mean_peak_target_speed_success: float
    mean_peak_target_speed_fall: float
    mean_peak_target_speed_ball_lost: float
    mean_max_robot_ball_distance: float
    mean_max_robot_ball_distance_success: float
    mean_max_robot_ball_distance_fall: float
    mean_max_robot_ball_distance_ball_lost: float
    mean_episode_length: float
    actor_dim: int
    action_dim: int
    finite: bool


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    num_envs: int,
    episodes: int,
    seed: int,
    device: str,
    goal_radius: float | None = None,
    scenario: SlalomScenario | None = None,
) -> RolloutResult:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not 0 < episodes <= num_envs:
        raise ValueError("episodes must satisfy 0 < episodes <= num_envs")

    env_cfg = load_env_cfg(TASK_ID)
    if scenario is None:
        env_cfg.commands["body_pose"].lateral_offset = SLALOM_FINAL_LATERAL_OFFSET
        env_cfg.commands["body_pose"].active_waypoints = len(
            env_cfg.commands["body_pose"].cone_x
        )
    else:
        configure_slalom_scenario(env_cfg, scenario)
    env_cfg.commands["body_pose"].goal_radius = SLALOM_GOAL_RADIUS
    env_cfg.curriculum.clear()
    if goal_radius is not None:
        if goal_radius <= 0.0:
            raise ValueError("goal_radius must be positive")
        env_cfg.commands["body_pose"].goal_radius = goal_radius
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.auto_reset = False
    ball_lost_params = env_cfg.terminations["ball_lost"].params
    min_ball_forward = ball_lost_params["min_forward"]
    max_ball_distance = ball_lost_params["max_distance"]
    max_ball_lateral = ball_lost_params["max_lateral"]

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    active = torch.zeros(num_envs, dtype=torch.bool, device=device)
    active[:episodes] = True
    successes = torch.zeros(num_envs, dtype=torch.bool, device=device)
    completions = torch.zeros(num_envs, device=device)
    falls = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ball_losses = torch.zeros(num_envs, dtype=torch.bool, device=device)
    obstacle_contact_terminations = torch.zeros(
        num_envs, dtype=torch.bool, device=device
    )
    invalid_route_terminations = torch.zeros(num_envs, dtype=torch.bool, device=device)
    course_sides = torch.zeros(num_envs, device=device)
    terminal_waypoints = torch.zeros(num_envs, dtype=torch.long, device=device)
    terminal_ball_position = torch.zeros(num_envs, 3, device=device)
    peak_target_speed = torch.zeros(num_envs, device=device)
    max_robot_ball_distance = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    finite = True
    actor_dim = 0
    action_dim = 0

    try:
        obs = env.get_observations()
        command = raw_env.command_manager.get_term("body_pose")
        tracker = StrictRouteTracker(
            command,
            raw_env.scene["robot"].data.root_link_pos_w[:, :2],
            raw_env.scene["ball"].data.root_link_pos_w[:, :2],
        )
        with torch.inference_mode():
            while active.any():
                actor_obs = obs["actor"]
                actions = policy(obs)
                actor_dim = actor_obs.shape[-1]
                action_dim = actions.shape[-1]
                finite &= bool(
                    torch.isfinite(actor_obs).all() and torch.isfinite(actions).all()
                )

                obs, rewards, dones, _ = env.step(actions)
                finite &= bool(torch.isfinite(rewards).all())
                lengths[active] += 1

                command = raw_env.command_manager.get_term("body_pose")
                ball = raw_env.scene["ball"]
                robot = raw_env.scene["robot"]
                ball_contact = (
                    (raw_env.scene["ball_marker_contact"].data.found > 0)
                    .flatten(start_dim=1)
                    .any(dim=1)
                )
                robot_contact = (
                    (raw_env.scene["robot_marker_contact"].data.found > 0)
                    .flatten(start_dim=1)
                    .any(dim=1)
                )
                tracker.update(
                    robot.data.root_link_pos_w[:, :2],
                    ball.data.root_link_pos_w[:, :2],
                    ball_contact,
                    robot_contact,
                    active,
                )
                target_speed = (
                    ball.data.root_link_lin_vel_w[:, :2] * command.direction_w
                ).sum(dim=1)
                peak_target_speed[active] = torch.maximum(
                    peak_target_speed[active], target_speed[active]
                )
                robot_ball_distance = torch.linalg.vector_norm(
                    microduck_mdp.ball_pos_in_base(raw_env, "ball")[:, :2], dim=1
                )
                max_robot_ball_distance[active] = torch.maximum(
                    max_robot_ball_distance[active], robot_ball_distance[active]
                )

                newly_done = active & dones.bool()
                if newly_done.any():
                    completed = command.completed
                    terminated = raw_env.termination_manager.terminated
                    successes[newly_done] = (completed & ~terminated)[newly_done]
                    completions[newly_done] = (
                        command.waypoint_index.float() + completed.float()
                    )[newly_done] / command.total_waypoints.float()[newly_done]
                    falls[newly_done] = raw_env.termination_manager.get_term(
                        "fell_over"
                    )[newly_done]
                    ball_losses[newly_done] = raw_env.termination_manager.get_term(
                        "ball_lost"
                    )[newly_done]
                    obstacle_contact_terminations[newly_done] = (
                        raw_env.termination_manager.get_term("obstacle_contact")
                    )[newly_done]
                    invalid_route_terminations[newly_done] = (
                        raw_env.termination_manager.get_term("invalid_route")
                    )[newly_done]
                    course_sides[newly_done] = command.course_side[newly_done]
                    terminal_waypoints[newly_done] = command.waypoint_index[newly_done]
                    terminal_ball_position[newly_done] = microduck_mdp.ball_pos_in_base(
                        raw_env, "ball"
                    )[newly_done]
                    active[newly_done] = False

                reset_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    raw_env.reset(env_ids=reset_ids)
                    obs = env.get_observations()
    finally:
        env.close()

    selected = slice(0, episodes)
    selected_successes = successes[selected]
    strict_successes = selected_successes & tracker.completed[selected]
    selected_falls = falls[selected]
    selected_ball_losses = ball_losses[selected]
    selected_obstacle_contacts = obstacle_contact_terminations[selected]
    selected_invalid_routes = invalid_route_terminations[selected]
    selected_sides = course_sides[selected]
    selected_waypoints = terminal_waypoints[selected]
    selected_peak_speed = peak_target_speed[selected]
    selected_max_distance = max_robot_ball_distance[selected]
    selected_terminal_ball_position = terminal_ball_position[selected]
    left = selected_sides > 0.0
    right = selected_sides < 0.0
    return RolloutResult(
        checkpoint=str(checkpoint.resolve()),
        episodes=episodes,
        success_rate=selected_successes.float().mean().item(),
        strict_success_rate=strict_successes.float().mean().item(),
        completion_rate=completions[selected].mean().item(),
        fall_rate=selected_falls.float().mean().item(),
        ball_lost_rate=selected_ball_losses.float().mean().item(),
        obstacle_contact_termination_rate=selected_obstacle_contacts.float()
        .mean()
        .item(),
        invalid_route_termination_rate=selected_invalid_routes.float().mean().item(),
        left_success_rate=selected_successes[left].float().mean().item(),
        right_success_rate=selected_successes[right].float().mean().item(),
        strict_left_success_rate=strict_successes[left].float().mean().item(),
        strict_right_success_rate=strict_successes[right].float().mean().item(),
        ball_route_completion_rate=(tracker.ball_index[selected] == len(tracker.cone_x))
        .float()
        .mean()
        .item(),
        robot_route_completion_rate=(
            tracker.robot_index[selected] == len(tracker.cone_x)
        )
        .float()
        .mean()
        .item(),
        ball_wrong_side_rate=tracker.ball_wrong_side[selected].float().mean().item(),
        robot_wrong_side_rate=tracker.robot_wrong_side[selected].float().mean().item(),
        ball_marker_contact_rate=tracker.ball_contact[selected].float().mean().item(),
        robot_marker_contact_rate=tracker.robot_contact[selected].float().mean().item(),
        obstacle_contact_waypoint_counts=tuple(
            int(
                (selected_obstacle_contacts & (selected_waypoints == index))
                .sum()
                .item()
            )
            for index in range(len(tracker.cone_x))
        ),
        fall_waypoint_counts=tuple(
            int((selected_falls & (selected_waypoints == index)).sum().item())
            for index in range(len(tracker.cone_x))
        ),
        ball_lost_waypoint_counts=tuple(
            int((selected_ball_losses & (selected_waypoints == index)).sum().item())
            for index in range(len(tracker.cone_x))
        ),
        ball_lost_backward_count=int(
            (
                selected_ball_losses
                & (selected_terminal_ball_position[:, 0] < min_ball_forward)
            )
            .sum()
            .item()
        ),
        ball_lost_distance_count=int(
            (
                selected_ball_losses
                & (
                    torch.linalg.vector_norm(
                        selected_terminal_ball_position[:, :2], dim=1
                    )
                    > max_ball_distance
                )
            )
            .sum()
            .item()
        ),
        ball_lost_lateral_count=int(
            (
                selected_ball_losses
                & (selected_terminal_ball_position[:, 1].abs() > max_ball_lateral)
            )
            .sum()
            .item()
        ),
        simultaneous_fall_and_ball_lost_count=int(
            (selected_falls & selected_ball_losses).sum().item()
        ),
        timeout_count=int(
            (
                ~selected_successes
                & ~selected_falls
                & ~selected_ball_losses
                & ~selected_obstacle_contacts
                & ~selected_invalid_routes
            )
            .sum()
            .item()
        ),
        mean_peak_target_speed=selected_peak_speed.mean().item(),
        mean_peak_target_speed_success=selected_peak_speed[selected_successes]
        .mean()
        .item(),
        mean_peak_target_speed_fall=selected_peak_speed[selected_falls].mean().item(),
        mean_peak_target_speed_ball_lost=selected_peak_speed[selected_ball_losses]
        .mean()
        .item(),
        mean_max_robot_ball_distance=selected_max_distance.mean().item(),
        mean_max_robot_ball_distance_success=selected_max_distance[selected_successes]
        .mean()
        .item(),
        mean_max_robot_ball_distance_fall=selected_max_distance[selected_falls]
        .mean()
        .item(),
        mean_max_robot_ball_distance_ball_lost=selected_max_distance[
            selected_ball_losses
        ]
        .mean()
        .item(),
        mean_episode_length=lengths[selected].float().mean().item(),
        actor_dim=actor_dim,
        action_dim=action_dim,
        finite=finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-success-gain", type=float, default=0.03)
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--goal-radius", type=float)
    parser.add_argument("--max-ball-lost-increase", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    configure_torch_backends()
    baseline = evaluate_checkpoint(
        args.baseline,
        num_envs=args.num_envs,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        goal_radius=args.goal_radius,
    )
    candidate = evaluate_checkpoint(
        args.candidate,
        num_envs=args.num_envs,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        goal_radius=args.goal_radius,
    )
    passed = (
        candidate.strict_success_rate >= args.min_success_rate
        and candidate.strict_success_rate
        >= baseline.strict_success_rate + args.min_success_gain
        and candidate.fall_rate <= baseline.fall_rate
        and candidate.ball_lost_rate
        <= baseline.ball_lost_rate + args.max_ball_lost_increase
        and candidate.finite
        and candidate.actor_dim == 61
        and candidate.action_dim == 14
    )
    result = {
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "strict_success_gain": (
            candidate.strict_success_rate - baseline.strict_success_rate
        ),
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
