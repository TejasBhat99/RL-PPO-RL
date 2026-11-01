"""Registration entry point for the RLRL environments."""

from gymnasium.envs.registration import register

from rlrl_env.envs.tube_car import TubeCarEnv

register(
    id="TubeCar-v0",
    entry_point="rlrl_env.envs:TubeCarEnv",
)

__all__ = ["TubeCarEnv"]
