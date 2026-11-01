"""Refine a cloned tube policy with DAgger recovery-state supervision."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from pretrain_tube_expert import collect_dataset
from rlrl_env import TubeCarEnv
from rlrl_env.agents.ppo import PPOConfig, load_model, save_checkpoint
from rlrl_env.agents.tube_expert import TubeExpertController
from rlrl_env.tube_curriculum import TUBE_STAGES, tube_env_kwargs
from train_tube_ppo import evaluate_tube_policy


def collect_dagger_dataset(
    model: torch.nn.Module,
    stage_names: list[str],
    episodes_per_stage: int,
    seed: int,
    expert_probability: float,
    intervention_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    controller = TubeExpertController()
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metrics: dict[str, dict[str, float]] = {}
    model.to("cpu")
    model.eval()
    for stage_index, stage_name in enumerate(stage_names):
        stage = TUBE_STAGES[stage_name]
        env = TubeCarEnv(**tube_env_kwargs(stage))
        successes = 0
        interventions = 0
        expert_steps = 0
        transitions = 0
        for episode in range(episodes_per_stage):
            observation, _ = env.reset(seed=seed + 100_000 * stage_index + episode)
            terminated = truncated = False
            info = {"is_success": False}
            while not (terminated or truncated):
                expert_action = controller.action(env)
                with torch.no_grad():
                    learner_action = model.predict(
                        torch.from_numpy(observation).unsqueeze(0), deterministic=True
                    ).numpy()[0]
                radial_ratio = env.radial_distance / max(
                    env.tube_radius - env.car_radius, 1e-6
                )
                intervene = radial_ratio >= intervention_ratio
                use_expert = intervene or rng.random() < expert_probability
                executed_action = expert_action if use_expert else learner_action
                observations.append(observation.copy())
                labels.append(expert_action)
                transitions += 1
                expert_steps += int(use_expert)
                interventions += int(intervene)
                observation, _, terminated, truncated, info = env.step(executed_action)
            successes += int(info["is_success"])
        env.close()
        metrics[stage_name] = {
            "episodes": float(episodes_per_stage),
            "success_rate": successes / episodes_per_stage,
            "transitions": float(transitions),
            "expert_action_fraction": expert_steps / transitions,
            "intervention_fraction": interventions / transitions,
        }
        print(f"DAgger {stage_name}: {json.dumps(metrics[stage_name])}", flush=True)
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        metrics,
    )


def train_actor(
    model: torch.nn.Module,
    observations: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    model.to(device)
    model.critic.requires_grad_(False)
    model.log_standard_deviation.requires_grad_(False)
    optimizer = Adam(model.actor.parameters(), lr=learning_rate, eps=1e-5)
    observation_tensor = torch.from_numpy(observations)
    action_tensor = torch.from_numpy(actions)
    final_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(observations))
        loss_sum = continuous_sum = binary_sum = 0.0
        boost_correct = samples = batches = 0
        model.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_observations = observation_tensor[indices].to(device)
            batch_actions = action_tensor[indices].to(device)
            outputs = model.actor(batch_observations)
            continuous_loss = F.mse_loss(outputs[:, :3], batch_actions[:, :3])
            binary_loss = F.binary_cross_entropy_with_logits(
                outputs[:, 3:], batch_actions[:, 3:]
            )
            loss = continuous_loss + 0.5 * binary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                boost_correct += int(
                    (
                        (torch.sigmoid(outputs[:, 4]) >= 0.5)
                        == (batch_actions[:, 4] >= 0.5)
                    ).sum()
                )
                samples += len(indices)
            loss_sum += float(loss.item())
            continuous_sum += float(continuous_loss.item())
            binary_sum += float(binary_loss.item())
            batches += 1
        final_metrics = {
            "epoch": float(epoch),
            "loss": loss_sum / batches,
            "continuous_mse": continuous_sum / batches,
            "binary_bce": binary_sum / batches,
            "boost_accuracy": boost_correct / samples,
        }
        print(f"supervised={json.dumps(final_metrics)}", flush=True)
    model.critic.requires_grad_(True)
    model.log_standard_deviation.requires_grad_(True)
    return copy.deepcopy(model.state_dict()), final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stages",
        default="procedural_tube,procedural_robust,procedural_challenging,procedural_hard_tail",
    )
    parser.add_argument("--anchor-episodes", type=int, default=100)
    parser.add_argument("--episodes-per-stage", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--initial-expert-probability", type=float, default=0.5)
    parser.add_argument("--intervention-ratio", type=float, default=0.72)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = (
        args.anchor_episodes,
        args.episodes_per_stage,
        args.iterations,
        args.epochs,
        args.batch_size,
        args.eval_episodes,
    )
    if min(counts) < 1:
        raise ValueError("All count arguments must be positive")
    if (
        not 0.0 <= args.intervention_ratio <= 1.0
        or not 0.0 <= args.initial_expert_probability <= 1.0
        or args.learning_rate <= 0
    ):
        raise ValueError("Invalid intervention ratio or learning rate")
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
    args.run_dir.mkdir(parents=True, exist_ok=True)

    probe_env = TubeCarEnv()
    observation_size = int(np.prod(probe_env.observation_space.shape))
    action_size = int(np.prod(probe_env.action_space.shape))
    probe_env.close()
    model, checkpoint = load_model(args.resume, torch.device("cpu"), observation_size)
    anchor_observations, anchor_actions, anchor_metrics = collect_dataset(
        stage_names, args.anchor_episodes, args.seed
    )
    aggregate_observations = [anchor_observations]
    aggregate_actions = [anchor_actions]
    history: list[dict[str, object]] = []
    best_score = (-1.0, -1.0, -float("inf"))
    best_state: dict[str, torch.Tensor] | None = None

    for iteration in range(1, args.iterations + 1):
        expert_probability = (
            args.initial_expert_probability * 0.5 ** (iteration - 1)
            if iteration < args.iterations
            else 0.0
        )
        dagger_observations, dagger_actions, collection_metrics = collect_dagger_dataset(
            model,
            stage_names,
            args.episodes_per_stage,
            args.seed + iteration * 1_000_000,
            expert_probability,
            args.intervention_ratio,
        )
        aggregate_observations.append(dagger_observations)
        aggregate_actions.append(dagger_actions)
        observations = np.concatenate(aggregate_observations)
        actions = np.concatenate(aggregate_actions)
        _, training_metrics = train_actor(
            model,
            observations,
            actions,
            device,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        model.to("cpu")
        evaluation = evaluate_tube_policy(
            model,
            TUBE_STAGES["procedural_tube"],
            args.eval_episodes,
            args.seed + 9_000_000,
            torch.device("cpu"),
        )
        print(f"iteration_evaluation={json.dumps(evaluation)}", flush=True)
        score = (
            evaluation["success_rate"],
            evaluation["mean_progress"],
            -evaluation["mean_radial_error"],
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
        config = PPOConfig(
            learning_rate=args.learning_rate,
            hidden_sizes=tuple(checkpoint["config"]["hidden_sizes"]),
            min_log_standard_deviation=-2.0,
            max_log_standard_deviation=-1.3,
        )
        save_checkpoint(
            args.run_dir / f"iteration_{iteration}.pt",
            model,
            optimizer,
            config,
            int(checkpoint["global_step"]),
            TUBE_STAGES["procedural_tube"],
            observation_size,
            action_size,
        )
        history.append(
            {
                "iteration": iteration,
                "expert_probability": expert_probability,
                "aggregate_transitions": len(observations),
                "collection": collection_metrics,
                "training": training_metrics,
                "evaluation": evaluation,
            }
        )

    if best_state is None:
        raise RuntimeError("No DAgger iteration completed")
    model.load_state_dict(best_state)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        hidden_sizes=tuple(checkpoint["config"]["hidden_sizes"]),
        min_log_standard_deviation=-2.0,
        max_log_standard_deviation=-1.3,
    )
    save_checkpoint(
        args.run_dir / "best_model.pt",
        model,
        optimizer,
        config,
        int(checkpoint["global_step"]),
        TUBE_STAGES["procedural_tube"],
        observation_size,
        action_size,
    )
    (args.run_dir / "metrics.json").write_text(
        json.dumps({"anchor": anchor_metrics, "iterations": history}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved best DAgger policy: {(args.run_dir / 'best_model.pt').resolve()}")


if __name__ == "__main__":
    main()
