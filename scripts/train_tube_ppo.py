"""Train native PyTorch PPO on the tube-navigation curriculum."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from rlrl_env import TubeCarEnv
from rlrl_env.agents.ppo import (
    ActorCritic,
    PPOConfig,
    RolloutBuffer,
    load_model,
    save_checkpoint,
    update_policy,
)
from rlrl_env.tube_curriculum import TUBE_STAGES, TubeStage, tube_env_kwargs


class TubeVectorRunner:
    """Synchronous vector runner with explicit terminal resets."""

    def __init__(self, stage: TubeStage, n_envs: int, seed: int) -> None:
        self.envs = [TubeCarEnv(**tube_env_kwargs(stage)) for _ in range(n_envs)]
        self.observations = np.stack(
            [env.reset(seed=seed + index)[0] for index, env in enumerate(self.envs)]
        )
        self.episode_rewards = np.zeros(n_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int64)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
        next_observations = []
        rewards = np.zeros(len(self.envs), dtype=np.float32)
        dones = np.zeros(len(self.envs), dtype=np.float32)
        completed: list[dict[str, float]] = []
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self.episode_rewards[index] += reward
            self.episode_lengths[index] += 1
            rewards[index] = reward
            dones[index] = float(done)
            if done:
                completed.append(
                    {
                        "reward": float(self.episode_rewards[index]),
                        "length": float(self.episode_lengths[index]),
                        "success": float(info["is_success"]),
                        "crashed": float(info["crashed"]),
                        "progress": float(info["progress_fraction"]),
                        "radial_error": float(info["mean_radial_error"]),
                        "control_change": float(info["mean_control_change"]),
                    }
                )
                observation, _ = env.reset()
                self.episode_rewards[index] = 0.0
                self.episode_lengths[index] = 0
            next_observations.append(observation)
        self.observations = np.stack(next_observations)
        return self.observations, rewards, dones, completed

    def close(self) -> None:
        for env in self.envs:
            env.close()


def evaluate_tube_policy(
    model: ActorCritic,
    stage: TubeStage,
    episodes: int,
    seed: int,
    device: torch.device,
    render: bool = False,
    deterministic: bool = True,
    render_fps: int | None = None,
    render_style: str = "legacy",
) -> dict[str, float]:
    env = TubeCarEnv(
        render_mode="human" if render else None,
        render_style=render_style,
        **tube_env_kwargs(stage),
    )
    if render and render_fps is not None:
        env.metadata = {**env.metadata, "render_fps": render_fps}
    rewards: list[float] = []
    lengths: list[int] = []
    progresses: list[float] = []
    radial_errors: list[float] = []
    control_changes: list[float] = []
    successes = 0
    crashes = 0
    playback_cancelled = False
    model.eval()
    for episode in range(episodes):
        episode_seed = seed + episode
        observation, info = env.reset(seed=episode_seed)
        episode_reward = 0.0
        episode_length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if render and env.consume_restart_request():
                observation, info = env.reset(seed=episode_seed)
                episode_reward = 0.0
                episode_length = 0
                terminated = truncated = False
                continue
            if render and env.paused:
                env.render()
                if env.quit_requested:
                    playback_cancelled = True
                    truncated = True
                continue
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            action = model.predict(
                observation_tensor, deterministic=deterministic
            ).cpu().numpy()[0]
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            if render and env.quit_requested:
                playback_cancelled = True
                truncated = True
        rewards.append(episode_reward)
        lengths.append(episode_length)
        progresses.append(float(info["progress_fraction"]))
        radial_errors.append(float(info["mean_radial_error"]))
        control_changes.append(float(info["mean_control_change"]))
        successes += int(info["is_success"])
        crashes += int(info["crashed"])
        if playback_cancelled:
            break
    env.close()
    model.train()
    completed_episodes = len(rewards)
    return {
        "episodes": float(completed_episodes),
        "success_rate": successes / completed_episodes,
        "crash_rate": crashes / completed_episodes,
        "mean_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_progress": float(np.mean(progresses)),
        "mean_radial_error": float(np.mean(radial_errors)),
        "mean_control_change": float(np.mean(control_changes)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default=(
            "short_straight,long_straight,gentle_curve,curved_tube,"
            "procedural_gentle,procedural_curved,procedural_hard,procedural_tube"
        ),
    )
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--min-log-std", type=float, default=-2.0)
    parser.add_argument("--max-log-std", type=float, default=-0.5)
    parser.add_argument(
        "--reset-log-std",
        type=float,
        help="Reset continuous-action exploration when resuming a checkpoint",
    )
    parser.add_argument("--stochastic-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.timesteps,
        args.n_envs,
        args.eval_episodes,
        args.eval_freq,
        args.checkpoint_freq,
        args.n_steps,
        args.batch_size,
        args.n_epochs,
    ) < 1:
        raise ValueError("Training counts must be positive")
    if args.learning_rate <= 0 or args.entropy_coefficient < 0:
        raise ValueError("Learning rate must be positive and entropy coefficient non-negative")
    if args.min_log_std > args.max_log_std:
        raise ValueError("min-log-std must not exceed max-log-std")
    stage_names = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = sorted(set(stage_names) - TUBE_STAGES.keys())
    if not stage_names or unknown:
        raise ValueError(f"Unknown tube stages: {unknown}; choices are {sorted(TUBE_STAGES)}")
    stages = [TUBE_STAGES[name] for name in stage_names]

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(1)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or Path("artifacts") / "torch_ppo" / f"tube-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        entropy_coefficient=args.entropy_coefficient,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        min_log_standard_deviation=args.min_log_std,
        max_log_standard_deviation=args.max_log_std,
    )

    probe_env = TubeCarEnv()
    observation_size = int(np.prod(probe_env.observation_space.shape))
    action_size = int(np.prod(probe_env.action_space.shape))
    probe_env.close()
    if args.resume:
        model, checkpoint = load_model(args.resume, device, observation_size)
        with torch.no_grad():
            if args.reset_log_std is not None:
                model.log_standard_deviation.fill_(args.reset_log_std)
            model.log_standard_deviation.clamp_(args.min_log_std, args.max_log_std)
        optimizer = Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        if checkpoint["optimizer_compatible"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
        global_step = int(checkpoint["global_step"])
    else:
        model = ActorCritic(observation_size, action_size, config.hidden_sizes).to(device)
        optimizer = Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        global_step = 0

    metadata = {
        "implementation": "native_pytorch_ppo",
        "environment": "TubeCar-v0",
        "seed": args.seed,
        "device": str(device),
        "n_envs": args.n_envs,
        "total_timesteps": args.timesteps,
        "stochastic_evaluation": args.stochastic_eval,
        "stages": [asdict(stage) for stage in stages],
        "ppo": asdict(config),
    }
    (run_dir / "configuration.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    writer = SummaryWriter(run_dir / "tensorboard")
    recent_episodes: deque[dict[str, float]] = deque(maxlen=100)
    timesteps_per_stage = args.timesteps // len(stages)
    all_metrics: dict[str, dict[str, float]] = {}

    try:
        for stage_index, stage in enumerate(stages):
            runner = TubeVectorRunner(stage, args.n_envs, args.seed + 100 * stage_index)
            buffer = RolloutBuffer(
                config.n_steps, args.n_envs, observation_size, action_size, device
            )
            stage_dir = run_dir / stage.name
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_steps = 0
            next_evaluation = args.eval_freq
            next_checkpoint = args.checkpoint_freq
            print(f"\n=== Tube PPO: {stage.name} ({timesteps_per_stage:,} steps) ===")

            initial_metrics = evaluate_tube_policy(
                model,
                stage,
                args.eval_episodes,
                args.seed + 10_000,
                device,
                deterministic=not args.stochastic_eval,
            )
            print(f"initial_evaluation={json.dumps(initial_metrics)}")
            best_score = (
                initial_metrics["success_rate"],
                initial_metrics["mean_progress"],
                -initial_metrics["mean_radial_error"],
                -initial_metrics["mean_control_change"],
            )
            save_checkpoint(
                stage_dir / "best_model.pt",
                model,
                optimizer,
                config,
                global_step,
                stage,
                observation_size,
                action_size,
            )
            for name in (
                "success_rate",
                "crash_rate",
                "mean_progress",
                "mean_radial_error",
                "mean_control_change",
                "mean_reward",
            ):
                writer.add_scalar(f"eval/{name}", initial_metrics[name], global_step)

            while stage_steps < timesteps_per_stage:
                buffer.reset()
                for _ in range(config.n_steps):
                    observations = torch.as_tensor(
                        runner.observations, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        actions, log_probabilities, _, values = model.action_and_value(observations)
                    next_observations, rewards, dones, completed = runner.step(
                        actions.cpu().numpy()
                    )
                    buffer.add(
                        observations,
                        actions,
                        log_probabilities,
                        torch.as_tensor(rewards, dtype=torch.float32, device=device),
                        torch.as_tensor(dones, dtype=torch.float32, device=device),
                        values,
                    )
                    recent_episodes.extend(completed)
                    runner.observations = next_observations
                    global_step += args.n_envs
                    stage_steps += args.n_envs

                with torch.no_grad():
                    next_values = model.value(
                        torch.as_tensor(runner.observations, dtype=torch.float32, device=device)
                    )
                buffer.compute_returns_and_advantages(
                    next_values, config.gamma, config.gae_lambda
                )
                train_metrics = update_policy(model, optimizer, buffer, config)

                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, global_step)
                if recent_episodes:
                    mean_reward = float(np.mean([item["reward"] for item in recent_episodes]))
                    success_rate = float(np.mean([item["success"] for item in recent_episodes]))
                    mean_progress = float(np.mean([item["progress"] for item in recent_episodes]))
                    mean_radial = float(np.mean([item["radial_error"] for item in recent_episodes]))
                    writer.add_scalar("rollout/mean_reward", mean_reward, global_step)
                    writer.add_scalar("rollout/success_rate", success_rate, global_step)
                    writer.add_scalar("rollout/mean_progress", mean_progress, global_step)
                    writer.add_scalar("rollout/mean_radial_error", mean_radial, global_step)
                else:
                    mean_reward = success_rate = mean_progress = mean_radial = float("nan")
                print(
                    f"step={global_step:>8,} stage={stage_steps:>8,} "
                    f"reward={mean_reward:>7.2f} progress={mean_progress:>5.1%} "
                    f"success={success_rate:>5.1%} radial={mean_radial:>4.2f} "
                    f"kl={train_metrics['kl']:.4f}"
                )

                if stage_steps >= next_checkpoint:
                    save_checkpoint(
                        stage_dir / "checkpoints" / f"step_{global_step}.pt",
                        model, optimizer, config, global_step, stage,
                        observation_size, action_size,
                    )
                    next_checkpoint += args.checkpoint_freq

                if stage_steps >= next_evaluation:
                    metrics = evaluate_tube_policy(
                        model, stage, args.eval_episodes, args.seed + 10_000,
                        device, deterministic=not args.stochastic_eval,
                    )
                    print(f"evaluation={json.dumps(metrics)}")
                    for name in (
                        "success_rate", "crash_rate", "mean_progress",
                        "mean_radial_error", "mean_control_change", "mean_reward",
                    ):
                        writer.add_scalar(f"eval/{name}", metrics[name], global_step)
                    score = (
                        metrics["success_rate"],
                        metrics["mean_progress"],
                        -metrics["mean_radial_error"],
                        -metrics["mean_control_change"],
                    )
                    if score > best_score:
                        best_score = score
                        save_checkpoint(
                            stage_dir / "best_model.pt", model, optimizer, config,
                            global_step, stage, observation_size, action_size,
                        )
                    next_evaluation += args.eval_freq

            final_metrics = evaluate_tube_policy(
                model, stage, args.eval_episodes, args.seed + 20_000, device,
                deterministic=not args.stochastic_eval,
            )
            all_metrics[stage.name] = final_metrics
            (stage_dir / "metrics.json").write_text(
                json.dumps(final_metrics, indent=2), encoding="utf-8"
            )
            save_checkpoint(
                stage_dir / "final_model.pt", model, optimizer, config,
                global_step, stage, observation_size, action_size,
            )
            print(f"Stage evaluation: {json.dumps(final_metrics, indent=2)}")
            runner.close()

        save_checkpoint(
            run_dir / "final_model.pt", model, optimizer, config,
            global_step, stages[-1], observation_size, action_size,
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(all_metrics, indent=2), encoding="utf-8"
        )
    finally:
        writer.close()

    print(f"\nTraining complete. Artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
