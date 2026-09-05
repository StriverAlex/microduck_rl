"""Compare two BallSlalom checkpoints with fixed-seed first-episode rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-BallSlalom-Flat-MicroDuck"


@dataclass(frozen=True)
class RolloutResult:
    checkpoint: str
    episodes: int
    success_rate: float
    completion_rate: float
    fall_rate: float
    ball_lost_rate: float
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
) -> RolloutResult:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not 0 < episodes <= num_envs:
        raise ValueError("episodes must satisfy 0 < episodes <= num_envs")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.auto_reset = False

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
    lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    finite = True
    actor_dim = 0
    action_dim = 0

    try:
        obs = env.get_observations()
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

                newly_done = active & dones.bool()
                if newly_done.any():
                    command = raw_env.command_manager.get_term("body_pose")
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
                    active[newly_done] = False

                reset_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    raw_env.reset(env_ids=reset_ids)
                    obs = env.get_observations()
    finally:
        env.close()

    selected = slice(0, episodes)
    return RolloutResult(
        checkpoint=str(checkpoint.resolve()),
        episodes=episodes,
        success_rate=successes[selected].float().mean().item(),
        completion_rate=completions[selected].mean().item(),
        fall_rate=falls[selected].float().mean().item(),
        ball_lost_rate=ball_losses[selected].float().mean().item(),
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
    parser.add_argument("--max-ball-lost-increase", type=float, default=0.01)
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
    )
    candidate = evaluate_checkpoint(
        args.candidate,
        num_envs=args.num_envs,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )
    passed = (
        candidate.success_rate >= baseline.success_rate + args.min_success_gain
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
        "success_gain": candidate.success_rate - baseline.success_rate,
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
