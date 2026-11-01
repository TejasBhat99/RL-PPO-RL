"""Pretrain the PyTorch tube actor from successful geometric-expert flights."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from rlrl_env import TubeCarEnv
from rlrl_env.agents.ppo import PPOConfig, load_model, save_checkpoint
from rlrl_env.agents.tube_expert import TubeExpertController
from rlrl_env.tube_curriculum import TUBE_STAGES, tube_env_kwargs


def collect_dataset(
    stage_names: list[str],
    episodes_per_stage: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    controller = TubeExpertController()
    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    metrics: dict[str, dict[str, float]] = {}
    for stage_index, stage_name in enumerate(stage_names):
        stage = TUBE_STAGES[stage_name]
        env = TubeCarEnv(**tube_env_kwargs(stage))
        stage_observations: list[np.ndarray] = []
        stage_actions: list[np.ndarray] = []
        successes = 0
        attempts = 0
        max_attempts = max(episodes_per_stage * 4, episodes_per_stage + 20)
        while successes < episodes_per_stage and attempts < max_attempts:
            observation, _ = env.reset(seed=seed + 100_000 * stage_index + attempts)
            episode_observations: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            terminated = truncated = False
            info = {"is_success": False}
            while not (terminated or truncated):
                action = controller.action(env)
                episode_observations.append(observation.copy())
                episode_actions.append(action)
                observation, _, terminated, truncated, info = env.step(action)
            attempts += 1
            if info["is_success"]:
                successes += 1
                stage_observations.extend(episode_observations)
                stage_actions.extend(episode_actions)
        env.close()
        if successes < episodes_per_stage:
            raise RuntimeError(
                f"Expert produced only {successes}/{episodes_per_stage} successful "
                f"{stage_name} episodes in {attempts} attempts"
            )
        all_observations.extend(stage_observations)
        all_actions.extend(stage_actions)
        metrics[stage_name] = {
            "successful_episodes": float(successes),
            "attempts": float(attempts),
            "success_rate": successes / attempts,
            "transitions": float(len(stage_observations)),
        }
        print(
            f"{stage_name}: retained {successes}/{attempts} successful episodes "
            f"({len(stage_observations):,} transitions)",
            flush=True,
        )
    return (
        np.asarray(all_observations, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stages",
        default="procedural_tube,procedural_robust,procedural_challenging",
    )
    parser.add_argument("--episodes-per-stage", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.episodes_per_stage, args.epochs, args.batch_size) < 1:
        raise ValueError("Episode, epoch, and batch counts must be positive")
    if args.learning_rate <= 0 or not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("Invalid learning rate or validation fraction")
    stage_names = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = sorted(set(stage_names) - TUBE_STAGES.keys())
    if not stage_names or unknown:
        raise ValueError(f"Unknown tube stages: {unknown}")

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    observations, actions, collection_metrics = collect_dataset(
        stage_names, args.episodes_per_stage, args.seed
    )
    permutation = np.random.permutation(len(observations))
    validation_size = max(1, int(len(observations) * args.validation_fraction))
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]

    probe_env = TubeCarEnv()
    observation_size = int(np.prod(probe_env.observation_space.shape))
    action_size = int(np.prod(probe_env.action_space.shape))
    probe_env.close()
    model, checkpoint = load_model(args.resume, device, observation_size)
    model.critic.requires_grad_(False)
    model.log_standard_deviation.requires_grad_(False)
    with torch.no_grad():
        model.log_standard_deviation.fill_(-1.5)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    training_observations = torch.from_numpy(observations[training_indices])
    training_actions = torch.from_numpy(actions[training_indices])
    validation_observations = torch.from_numpy(observations[validation_indices]).to(device)
    validation_actions = torch.from_numpy(actions[validation_indices]).to(device)
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_permutation = torch.randperm(len(training_observations))
        model.train()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(epoch_permutation), args.batch_size):
            indices = epoch_permutation[start : start + args.batch_size]
            batch_observations = training_observations[indices].to(device)
            batch_actions = training_actions[indices].to(device)
            outputs = model.actor(batch_observations)
            continuous_loss = F.smooth_l1_loss(outputs[:, :3], batch_actions[:, :3])
            binary_loss = F.binary_cross_entropy_with_logits(
                outputs[:, 3:], batch_actions[:, 3:]
            )
            loss = continuous_loss + 0.5 * binary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        model.eval()
        with torch.no_grad():
            validation_outputs = model.actor(validation_observations)
            validation_continuous = F.smooth_l1_loss(
                validation_outputs[:, :3], validation_actions[:, :3]
            )
            validation_binary = F.binary_cross_entropy_with_logits(
                validation_outputs[:, 3:], validation_actions[:, 3:]
            )
            validation_loss = float(
                (validation_continuous + 0.5 * validation_binary).item()
            )
            boost_accuracy = float(
                (
                    (torch.sigmoid(validation_outputs[:, 4]) >= 0.5)
                    == (validation_actions[:, 4] >= 0.5)
                )
                .float()
                .mean()
                .item()
            )
        epoch_metrics = {
            "epoch": float(epoch),
            "training_loss": total_loss / batches,
            "validation_loss": validation_loss,
            "boost_accuracy": boost_accuracy,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No behavior-cloning update completed")
    model.load_state_dict(best_state)
    model.critic.requires_grad_(True)
    model.log_standard_deviation.requires_grad_(True)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        hidden_sizes=tuple(checkpoint["config"]["hidden_sizes"]),
        min_log_standard_deviation=-2.0,
        max_log_standard_deviation=-1.3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.output,
        model,
        optimizer,
        config,
        int(checkpoint["global_step"]),
        TUBE_STAGES[stage_names[0]],
        observation_size,
        action_size,
    )
    metadata = {
        "source_checkpoint": str(args.resume),
        "output_checkpoint": str(args.output),
        "device": str(device),
        "stages": stage_names,
        "transitions": int(len(observations)),
        "collection": collection_metrics,
        "best_validation_loss": best_validation_loss,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved expert-pretrained actor: {args.output.resolve()}")


if __name__ == "__main__":
    main()
