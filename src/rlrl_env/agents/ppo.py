"""Native PyTorch components for clipped Proximal Policy Optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal
from torch.nn import functional as F
from torch.optim import Adam


def _layer(in_features: int, out_features: int, gain: float = np.sqrt(2.0)) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Hybrid Gaussian/Bernoulli policy and value function with independent MLPs."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
        binary_action_count: int = 2,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("At least one hidden layer is required")
        if not 0 <= binary_action_count < action_size:
            raise ValueError("binary_action_count must be smaller than action_size")
        self.action_size = action_size
        self.binary_action_count = binary_action_count
        self.continuous_action_count = action_size - binary_action_count

        actor_layers: list[nn.Module] = []
        critic_layers: list[nn.Module] = []
        previous_size = observation_size
        for hidden_size in hidden_sizes:
            actor_layers.extend((_layer(previous_size, hidden_size), nn.Tanh()))
            critic_layers.extend((_layer(previous_size, hidden_size), nn.Tanh()))
            previous_size = hidden_size
        actor_layers.append(_layer(previous_size, action_size, gain=0.01))
        critic_layers.append(_layer(previous_size, 1, gain=1.0))

        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)
        self.log_standard_deviation = nn.Parameter(
            torch.full((self.continuous_action_count,), -0.5)
        )

    def distributions(self, observations: torch.Tensor) -> tuple[Normal, Bernoulli | None]:
        outputs = self.actor(observations)
        means = outputs[..., : self.continuous_action_count]
        standard_deviations = torch.exp(self.log_standard_deviation).expand_as(means)
        continuous_distribution = Normal(means, standard_deviations)
        if self.binary_action_count:
            logits = outputs[..., self.continuous_action_count :]
            return continuous_distribution, Bernoulli(logits=logits)
        return continuous_distribution, None

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def action_and_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous_distribution, binary_distribution = self.distributions(observations)
        if actions is None:
            continuous_actions = continuous_distribution.sample()
            if binary_distribution is not None:
                binary_actions = binary_distribution.sample()
                actions = torch.cat((continuous_actions, binary_actions), dim=-1)
            else:
                actions = continuous_actions
        continuous_actions = actions[..., : self.continuous_action_count]
        log_probabilities = continuous_distribution.log_prob(continuous_actions).sum(dim=-1)
        entropy = continuous_distribution.entropy().sum(dim=-1)
        if binary_distribution is not None:
            binary_actions = actions[..., self.continuous_action_count :]
            log_probabilities += binary_distribution.log_prob(binary_actions).sum(dim=-1)
            entropy += binary_distribution.entropy().sum(dim=-1)
        return actions, log_probabilities, entropy, self.value(observations)

    @torch.no_grad()
    def predict(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        continuous_distribution, binary_distribution = self.distributions(observations)
        continuous_actions = (
            continuous_distribution.mean if deterministic else continuous_distribution.sample()
        )
        continuous_actions = torch.clamp(continuous_actions, -1.0, 1.0)
        if binary_distribution is None:
            return continuous_actions
        binary_actions = (
            (binary_distribution.probs >= 0.5).float()
            if deterministic
            else binary_distribution.sample()
        )
        return torch.cat((continuous_actions, binary_actions), dim=-1)


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Fixed-size on-policy storage with Generalized Advantage Estimation."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        observation_size: int,
        action_size: int,
        device: torch.device,
    ) -> None:
        shape = (n_steps, n_envs)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.observations = torch.zeros(
            (*shape, observation_size), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros((*shape, action_size), dtype=torch.float32, device=device)
        self.log_probabilities = torch.zeros(shape, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=device)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(shape, dtype=torch.float32, device=device)
        self.returns = torch.zeros(shape, dtype=torch.float32, device=device)
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probabilities: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if self.position >= self.n_steps:
            raise RuntimeError("Rollout buffer is full")
        self.observations[self.position].copy_(observations)
        self.actions[self.position].copy_(actions)
        self.log_probabilities[self.position].copy_(log_probabilities)
        self.rewards[self.position].copy_(rewards)
        self.dones[self.position].copy_(dones)
        self.values[self.position].copy_(values)
        self.position += 1

    def compute_returns_and_advantages(
        self,
        next_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.n_steps:
            raise RuntimeError("Rollout buffer must be full before computing advantages")
        last_advantage = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.n_steps)):
            subsequent_values = next_values if step == self.n_steps - 1 else self.values[step + 1]
            nonterminal = 1.0 - self.dones[step]
            temporal_difference = (
                self.rewards[step]
                + gamma * subsequent_values * nonterminal
                - self.values[step]
            )
            last_advantage = (
                temporal_difference + gamma * gae_lambda * nonterminal * last_advantage
            )
            self.advantages[step] = last_advantage
        self.returns.copy_(self.advantages + self.values)

    def batches(self, batch_size: int) -> Iterator[RolloutBatch]:
        total_size = self.n_steps * self.n_envs
        indices = torch.randperm(total_size, device=self.device)
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        log_probabilities = self.log_probabilities.flatten()
        values = self.values.flatten()
        advantages = self.advantages.flatten()
        returns = self.returns.flatten()
        for start in range(0, total_size, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield RolloutBatch(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_log_probabilities=log_probabilities[batch_indices],
                old_values=values[batch_indices],
                advantages=advantages[batch_indices],
                returns=returns[batch_indices],
            )


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters shared by tube PPO training and saved checkpoints."""

    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (256, 256)
    min_log_standard_deviation: float = -2.0
    max_log_standard_deviation: float = -0.5


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: Adam,
    config: PPOConfig,
    global_step: int,
    stage: Any,
    observation_size: int,
    action_size: int,
) -> None:
    """Save a tube policy checkpoint using tensors and primitive metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "global_step": global_step,
            "stage": asdict(stage),
            "observation_size": observation_size,
            "action_size": action_size,
            "binary_action_count": model.binary_action_count,
        },
        path,
    )


def load_model(
    path: Path,
    device: torch.device,
    observation_size_override: int | None = None,
) -> tuple[ActorCritic, dict[str, Any]]:
    """Load a policy checkpoint without allowing arbitrary pickled objects."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    hidden_sizes = tuple(checkpoint["config"]["hidden_sizes"])
    binary_action_count = int(checkpoint.get("binary_action_count", 2))
    observation_size = observation_size_override or int(checkpoint["observation_size"])
    model = ActorCritic(
        observation_size,
        checkpoint["action_size"],
        hidden_sizes,
        binary_action_count=binary_action_count,
    ).to(device)
    state_dict = dict(checkpoint["model_state_dict"])
    saved_log_standard_deviation = state_dict["log_standard_deviation"]
    optimizer_compatible = (
        saved_log_standard_deviation.numel() == model.continuous_action_count
    )
    if not optimizer_compatible:
        state_dict["log_standard_deviation"] = saved_log_standard_deviation[
            : model.continuous_action_count
        ]
    for first_layer_key in ("actor.0.weight", "critic.0.weight"):
        saved_weight = state_dict[first_layer_key]
        expected_width = model.state_dict()[first_layer_key].shape[1]
        if saved_weight.shape[1] < expected_width:
            padding = torch.zeros(
                (saved_weight.shape[0], expected_width - saved_weight.shape[1]),
                dtype=saved_weight.dtype,
                device=saved_weight.device,
            )
            state_dict[first_layer_key] = torch.cat((saved_weight, padding), dim=1)
            optimizer_compatible = False
        elif saved_weight.shape[1] > expected_width:
            raise ValueError(
                f"Checkpoint observation size {saved_weight.shape[1]} exceeds "
                f"requested size {expected_width}"
            )
    model.load_state_dict(state_dict)
    checkpoint["optimizer_compatible"] = optimizer_compatible
    return model, checkpoint


def update_policy(
    model: ActorCritic,
    optimizer: Adam,
    buffer: RolloutBuffer,
    config: PPOConfig,
) -> dict[str, float]:
    """Apply clipped PPO updates to one completed rollout buffer."""

    normalized_advantages = (buffer.advantages - buffer.advantages.mean()) / (
        buffer.advantages.std() + 1e-8
    )
    buffer.advantages.copy_(normalized_advantages)
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
    }
    updates = 0
    actor_parameters = [*model.actor.parameters(), model.log_standard_deviation]
    critic_parameters = list(model.critic.parameters())

    for _ in range(config.n_epochs):
        for batch in buffer.batches(config.batch_size):
            _, new_log_probabilities, entropy, new_values = model.action_and_value(
                batch.observations, batch.actions
            )
            log_ratio = new_log_probabilities - batch.old_log_probabilities
            ratio = torch.exp(log_ratio)
            unclipped_loss = -batch.advantages * ratio
            clipped_loss = -batch.advantages * torch.clamp(
                ratio, 1.0 - config.clip_range, 1.0 + config.clip_range
            )
            policy_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
            value_loss = F.smooth_l1_loss(new_values, batch.returns)
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor_parameters, config.max_gradient_norm)
            nn.utils.clip_grad_norm_(critic_parameters, config.max_gradient_norm)
            optimizer.step()
            with torch.no_grad():
                model.log_standard_deviation.clamp_(
                    config.min_log_standard_deviation,
                    config.max_log_standard_deviation,
                )
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_range).float().mean()
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy_mean.item())
            metrics["kl"] += float(approximate_kl.item())
            metrics["clip_fraction"] += float(clip_fraction.item())
            updates += 1

    return {name: value / updates for name, value in metrics.items()}
