"""Evaluate or render a native PyTorch tube-navigation policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rlrl_env import TubeCarEnv
from rlrl_env.agents.ppo import load_model
from rlrl_env.tube_curriculum import TUBE_STAGES, tube_stage_with_overrides
from train_tube_ppo import evaluate_tube_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--stage", choices=sorted(TUBE_STAGES), default="short_straight")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument(
        "--tube-length",
        type=float,
        help="Override the selected stage's tube length (must be greater than 5)",
    )
    parser.add_argument(
        "--tube-radius",
        type=float,
        help="Override the selected stage's tube radius (must be greater than 1)",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        help="Override the selected stage's curvature in [0, 1]",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--graphics",
        choices=("enhanced", "legacy"),
        default="enhanced",
        help="Enhanced playback graphics or the original training-era wireframe",
    )
    parser.add_argument(
        "--playback-fps",
        type=int,
        default=50,
        help="Rendered policy steps per second; use 3-8 for slow playback",
    )
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.playback_fps < 1:
        raise ValueError("playback-fps must be positive")
    try:
        stage = tube_stage_with_overrides(
            TUBE_STAGES[args.stage],
            tube_length=args.tube_length,
            tube_radius=args.tube_radius,
            curvature=args.curvature,
        )
    except ValueError as error:
        parser.error(str(error))

    device = torch.device("cpu")
    probe_env = TubeCarEnv()
    observation_size = probe_env.observation_space.shape[0]
    probe_env.close()
    model, _ = load_model(args.model, device, observation_size)
    metrics = evaluate_tube_policy(
        model,
        stage,
        args.episodes,
        args.seed,
        device,
        render=args.render,
        deterministic=not args.stochastic,
        render_fps=args.playback_fps,
        render_style=args.graphics,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
