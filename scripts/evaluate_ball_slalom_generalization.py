"""Evaluate one BallSlalom checkpoint on fixed out-of-distribution courses."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from evaluate_ball_slalom import GENERALIZATION_SCENARIOS, evaluate_checkpoint
from mjlab.utils.torch import configure_torch_backends

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/ball_slalom/2026-09-07_16-41-42_slalom_preview_40cm_v6"
    / "model_10250.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs/diagnostics/slalom_generalization_model_10250" / "results.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(GENERALIZATION_SCENARIOS),
        default=tuple(GENERALIZATION_SCENARIOS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    configure_torch_backends()
    results = {}
    for name in args.scenarios:
        scenario = GENERALIZATION_SCENARIOS[name]
        rollout = evaluate_checkpoint(
            args.checkpoint,
            num_envs=args.num_envs,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            scenario=scenario,
        )
        results[name] = {
            "scenario": asdict(scenario),
            "rollout": asdict(rollout),
        }

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "episodes_per_scenario": args.episodes,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
