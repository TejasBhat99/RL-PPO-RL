"""Run a short random tube rollout and report environment diagnostics."""

import gymnasium as gym

import rlrl_env  # noqa: F401 - registers TubeCar-v0


def main() -> None:
    env = gym.make("TubeCar-v0", render_mode="rgb_array", procedural_course=True)
    observation, info = env.reset(seed=7)
    total_reward = 0.0

    for _ in range(300):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(
        f"steps={info['episode_steps']} progress={info['progress_fraction']:.1%} "
        f"crashed={info['crashed']} reward={total_reward:.2f} "
        f"observation_shape={observation.shape}"
    )
    env.close()


if __name__ == "__main__":
    main()
