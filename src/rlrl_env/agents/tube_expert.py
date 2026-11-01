"""Geometric teacher policy for airborne tube navigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlrl_env.envs.tube_car import TubeCarEnv


@dataclass(frozen=True)
class TubeExpertConfig:
    """Parameters for a conservative thrust-vectoring tube controller."""

    target_speed: float = 12.0
    position_gain: float = 1.2
    velocity_time_constant: float = 0.5
    tangent_lookahead: float = 5.0
    rotation_gain: float = 1.0
    pulse_boost_threshold: float = 8.0
    full_boost_threshold: float = 18.0


class TubeExpertController:
    """Produce deterministic, observation-compatible teacher actions.

    Boost alternates when moderate thrust is required.  The previous boost bit
    is part of :class:`TubeCarEnv`'s observation, so the teacher is Markovian
    and can be imitated by a feed-forward policy.
    """

    def __init__(self, config: TubeExpertConfig | None = None) -> None:
        self.config = config or TubeExpertConfig()

    def action(self, env: TubeCarEnv) -> np.ndarray:
        config = self.config
        center = env._centerline(env.progress)
        tangent = env._tangent(
            min(env.progress + config.tangent_lookahead, env.tube_length)
        )
        position_error = center - env.position
        desired_velocity = (
            config.target_speed * tangent + config.position_gain * position_error
        )
        gravity_compensation = np.array([0.0, 0.0, 9.81], dtype=np.float32)
        desired_thrust = (
            (desired_velocity - env.velocity) / config.velocity_time_constant
            + gravity_compensation
        )
        thrust_magnitude = float(np.linalg.norm(desired_thrust))
        desired_forward = desired_thrust / max(thrust_magnitude, 1e-6)
        local_forward = env.orientation.T @ desired_forward

        yaw_error = np.arctan2(local_forward[1], max(float(local_forward[0]), 1e-3))
        horizontal = max(float(np.hypot(local_forward[0], local_forward[1])), 1e-3)
        pitch_error = -np.arctan2(local_forward[2], horizontal)
        pitch = float(np.clip(config.rotation_gain * pitch_error, -1.0, 1.0))
        yaw = float(np.clip(config.rotation_gain * yaw_error, -1.0, 1.0))

        previous_boost = bool(env._last_action[6] > 0.5)
        if thrust_magnitude >= config.full_boost_threshold:
            boost = 1.0
        elif thrust_magnitude >= config.pulse_boost_threshold:
            boost = float(not previous_boost)
        else:
            boost = 0.0
        return np.array([pitch, yaw, 0.0, 0.0, boost], dtype=np.float32)
