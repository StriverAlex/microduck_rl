"""Render a multi-environment BallSlalom generalization replay."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import numpy as np
import torch
from evaluate_ball_slalom import (
    GENERALIZATION_SCENARIOS,
    TASK_ID,
    SlalomScenario,
    configure_slalom_scenario,
)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/ball_slalom/2026-09-07_16-41-42_slalom_preview_40cm_v6"
    / "model_10250.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "logs/diagnostics/slalom_generalization_videos_model_10250"
    / "five_irregular_multi_env.mp4"
)
FRAME_RATE = 25
FRAME_STRIDE = 2


def configure_multi_env_cfg(
    cfg: ManagerBasedRlEnvCfg,
    scenario: SlalomScenario,
    *,
    num_envs: int,
    seed: int,
) -> ManagerBasedRlEnvCfg:
    """Configure a spaced grid and fixed world camera for one replay."""
    if num_envs < 2:
        raise ValueError("num_envs must be at least two")
    configure_slalom_scenario(cfg, scenario)
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 3.0
    cfg.seed = seed
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.entity_name = None
    cfg.viewer.body_name = None
    cfg.viewer.lookat = (0.0, 0.0, 0.10)
    cfg.viewer.distance = 11.0
    cfg.viewer.azimuth = 140.0
    cfg.viewer.elevation = -50.0
    cfg.viewer.width = 1280
    cfg.viewer.height = 720
    cfg.viewer.max_extra_envs = num_envs - 1
    return cfg


def _overlay(frame: np.ndarray, scenario: SlalomScenario, step: int, num_envs: int):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    text = (
        f"BallSlalom generalization | {scenario.name} | "
        f"envs={num_envs} | t={step / 50.0:5.2f}s"
    )
    draw.rectangle((0, 0, image.width, 40), fill=(0, 0, 0))
    draw.text((12, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def render_multi_env_replay(
    checkpoint: Path,
    output: Path,
    *,
    scenario: SlalomScenario,
    num_envs: int,
    duration_s: float,
    seed: int,
    device: str,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    env_cfg = configure_multi_env_cfg(
        load_env_cfg(TASK_ID, play=True),
        scenario,
        num_envs=num_envs,
        seed=seed,
    )
    agent_cfg = load_rl_cfg(TASK_ID)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
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
    total_steps = round(duration_s / raw_env.step_dt)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        obs = env.get_observations()

        def frames() -> Iterable[np.ndarray]:
            nonlocal obs
            with torch.inference_mode():
                for step in range(total_steps):
                    actions = policy(obs)
                    if obs["actor"].shape[-1] != 61 or actions.shape[-1] != 14:
                        raise RuntimeError("Policy contract must remain 61D -> 14D")
                    if (
                        not torch.isfinite(obs["actor"]).all()
                        or not torch.isfinite(actions).all()
                    ):
                        raise RuntimeError("Non-finite policy input or output")
                    obs, rewards, _, _ = env.step(actions)
                    if not torch.isfinite(rewards).all():
                        raise RuntimeError("Non-finite reward during replay")
                    if step % FRAME_STRIDE == 0:
                        rendered = raw_env.render()
                        if rendered is None:
                            raise RuntimeError("rgb_array renderer returned no frame")
                        yield _overlay(rendered, scenario, step, num_envs)

        media.write_video(output, frames(), fps=FRAME_RATE, qp=20)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--scenario",
        choices=tuple(GENERALIZATION_SCENARIOS),
        default="five_irregular",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    configure_torch_backends()
    render_multi_env_replay(
        args.checkpoint,
        args.output,
        scenario=GENERALIZATION_SCENARIOS[args.scenario],
        num_envs=args.num_envs,
        duration_s=args.duration_s,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
