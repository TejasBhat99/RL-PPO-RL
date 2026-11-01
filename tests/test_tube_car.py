import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rlrl_env  # noqa: F401 - registers TubeCar-v0
from examples.play_tube_keyboard import (
    combine_actions,
    controller_action,
    raw_playstation_action,
    tube_keyboard_action,
)
from rlrl_env import TubeCarEnv
from rlrl_env.agents.tube_expert import TubeExpertController
from rlrl_env.rendering.tube_renderer import (
    EnhancedTubeRenderer,
    playback_button_rects,
)


def test_tube_environment_passes_gymnasium_checker() -> None:
    env = TubeCarEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_tube_observation_and_registration() -> None:
    env = gym.make("TubeCar-v0", curvature=0.0)
    observation, info = env.reset(seed=1)
    assert observation.shape == (47,)
    assert env.observation_space.contains(observation)
    assert info["progress"] == 0.0
    env.close()


def test_seeded_procedural_tube_is_deterministic() -> None:
    env = TubeCarEnv(procedural_course=True)
    first_observation, _ = env.reset(seed=42)
    first_points = np.stack([env._centerline(s) for s in (0.0, 25.0, 75.0)])
    second_observation, _ = env.reset(seed=42)
    second_points = np.stack([env._centerline(s) for s in (0.0, 25.0, 75.0)])
    np.testing.assert_allclose(first_observation, second_observation)
    np.testing.assert_allclose(first_points, second_points)
    env.close()


def test_reaching_tube_exit_gives_large_positive_reward() -> None:
    env = TubeCarEnv(curvature=0.0, frame_skip=1)
    env.reset(seed=0)
    env.position = env._centerline(env.tube_length - 0.01)
    env.velocity = 10.0 * env._tangent(env.tube_length)

    _, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated and not truncated
    assert info["is_success"] is True
    assert reward > 99.0
    env.close()


def test_crossing_tube_exit_outside_wall_is_a_crash() -> None:
    env = TubeCarEnv(curvature=0.0, frame_skip=1)
    env.reset(seed=0)
    center, tangent, right, _ = env._tube_frame(env.tube_length)
    env.position = center - 0.01 * tangent + (
        env.tube_radius - env.car_radius + 0.01
    ) * right
    env.velocity = 10.0 * tangent
    env.progress = env.tube_length - 0.01

    _, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated and not truncated
    assert info["is_success"] is False
    assert info["crashed"] is True
    assert reward < -99.0
    env.close()


def test_touching_tube_wall_gives_large_negative_reward() -> None:
    env = TubeCarEnv(curvature=0.0, frame_skip=1)
    env.reset(seed=0)
    center, _, right, _ = env._tube_frame(0.0)
    env.position = center + (env.tube_radius - env.car_radius + 0.01) * right
    env.velocity.fill(0.0)

    _, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
    assert terminated and not truncated
    assert info["crashed"] is True
    assert reward < -99.0
    env.close()


def test_tube_reward_components_sum_to_reward() -> None:
    env = TubeCarEnv(curvature=0.0, frame_skip=1)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    assert reward == sum(info["reward_terms"].values())
    assert info["reward_terms"]["step"] == -0.01
    env.close()


def test_tube_uses_unlimited_boost() -> None:
    env = TubeCarEnv(curvature=0.0, frame_skip=1)
    env.reset(seed=0)
    action = np.zeros(5, dtype=np.float32)
    action[4] = 1.0
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(action)
        assert not (terminated or truncated)
    assert env.boost == env.max_boost
    env.close()


def test_tube_observation_tracks_previous_boost_action() -> None:
    env = TubeCarEnv(curvature=0.0, frame_skip=1)
    observation, _ = env.reset(seed=0)
    assert observation[-1] == 0.0
    action = np.zeros(5, dtype=np.float32)
    action[4] = 1.0
    observation, _, _, _, _ = env.step(action)
    assert observation[-1] == 1.0
    env.close()


def test_tube_expert_produces_valid_deterministic_action() -> None:
    env = TubeCarEnv(procedural_course=True)
    env.reset(seed=42)
    controller = TubeExpertController()
    first = controller.action(env)
    second = controller.action(env)
    np.testing.assert_array_equal(first, second)
    assert env.action_space.contains(first)
    assert first[3] == 0.0
    env.close()


@pytest.mark.parametrize("render_style", ["legacy", "enhanced"])
def test_tube_rgb_render_shape(render_style: str) -> None:
    env = TubeCarEnv(render_mode="rgb_array", render_style=render_style)
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (540, 960, 3)
    assert frame.dtype == np.uint8
    env.close()


def test_tube_rejects_unknown_render_style() -> None:
    with pytest.raises(ValueError, match="render_style"):
        TubeCarEnv(render_style="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize("render_style", ["legacy", "enhanced"])
def test_playback_ui_pauses_plays_and_requests_restart(render_style: str) -> None:
    import pygame

    env = TubeCarEnv(render_mode="rgb_array", render_style=render_style)
    env.reset(seed=19)
    play_rect, pause_rect, restart_rect = playback_button_rects(pygame, (960, 540))

    playing_frame = env.render()
    initial_state = (
        env.position.copy(),
        env.velocity.copy(),
        env.orientation.copy(),
        env.episode_steps,
    )
    env._handle_render_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=pause_rect.center
        ),
        pygame,
    )
    assert env.paused
    paused_frame = env.render()
    np.testing.assert_array_equal(env.render(), paused_frame)
    assert not np.array_equal(playing_frame, paused_frame)
    np.testing.assert_array_equal(env.position, initial_state[0])
    np.testing.assert_array_equal(env.velocity, initial_state[1])
    np.testing.assert_array_equal(env.orientation, initial_state[2])
    assert env.episode_steps == initial_state[3]

    env._handle_render_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=play_rect.center),
        pygame,
    )
    assert not env.paused
    np.testing.assert_array_equal(env.render(), playing_frame)

    env._handle_render_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=restart_rect.center
        ),
        pygame,
    )
    assert env.consume_restart_request()
    assert not env.consume_restart_request()
    assert not env.paused
    env.close()


def test_reset_clears_playback_state_and_reproduces_seeded_course() -> None:
    import pygame

    env = TubeCarEnv(procedural_course=True, frame_skip=1)
    initial_observation, initial_info = env.reset(seed=2718)
    initial_course = np.stack(
        [env._centerline(progress) for progress in (0.0, 37.0, 82.0)]
    )
    for _ in range(4):
        env.step(np.array([0.1, -0.05, 0.08, 0.0, 0.0], dtype=np.float32))

    _, pause_rect, restart_rect = playback_button_rects(pygame, (960, 540))
    for rect in (restart_rect, pause_rect):
        env._handle_render_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                button=1,
                pos=rect.center,
            ),
            pygame,
        )
    assert env.paused

    reset_observation, reset_info = env.reset(seed=2718)
    reset_course = np.stack(
        [env._centerline(progress) for progress in (0.0, 37.0, 82.0)]
    )
    np.testing.assert_array_equal(reset_observation, initial_observation)
    np.testing.assert_array_equal(reset_course, initial_course)
    assert reset_info == initial_info
    assert not env.paused
    assert not env.consume_restart_request()
    assert env.episode_steps == 0
    assert env.path_length == 0.0
    assert env.reward_terms == {}
    env.close()


def test_enhanced_renderer_has_no_target_crosshair() -> None:
    assert not hasattr(EnhancedTubeRenderer, "_draw_target_reticle")


def test_tube_camera_axes_are_invariant_to_car_pose_inside_tube() -> None:
    env = TubeCarEnv(render_style="enhanced", curvature=0.5)
    env.reset(seed=11)
    env.progress = 43.0
    center, _, right, up = env._tube_frame(env.progress)
    env.position = center.copy()
    env.orientation = np.eye(3, dtype=np.float32)
    centered_camera = env._renderer()._camera()

    env.position = center + 2.1 * right - 1.3 * up
    env.orientation = env._rotation_increment(
        np.array([1.1, -0.8, 2.4], dtype=np.float32)
    )
    offset_camera = env._renderer()._camera()

    expected_follow = 0.60 * (2.1 * right - 1.3 * up)
    np.testing.assert_allclose(
        offset_camera[0] - centered_camera[0], expected_follow, atol=1e-6
    )
    for centered_component, offset_component in zip(
        centered_camera[1:], offset_camera[1:]
    ):
        np.testing.assert_allclose(centered_component, offset_component, atol=1e-6)
    env.close()


@pytest.mark.parametrize("render_style", ["legacy", "enhanced"])
def test_render_style_is_deterministic_and_trajectory_neutral(
    render_style: str,
) -> None:
    plain = TubeCarEnv(procedural_course=True, frame_skip=2)
    rendered = TubeCarEnv(
        render_mode="rgb_array",
        render_style=render_style,  # type: ignore[arg-type]
        procedural_course=True,
        frame_skip=2,
    )
    plain_observation, plain_info = plain.reset(seed=90210)
    rendered_observation, rendered_info = rendered.reset(seed=90210)
    np.testing.assert_array_equal(plain_observation, rendered_observation)
    assert plain_info == rendered_info

    actions = np.random.default_rng(17).uniform(-0.4, 0.4, size=(12, 5)).astype(np.float32)
    actions[:, 3] = 0.0
    actions[:, 4] = 0.0
    for action in actions:
        first_frame = rendered.render()
        second_frame = rendered.render()
        np.testing.assert_array_equal(first_frame, second_frame)

        plain_step = plain.step(action)
        rendered_step = rendered.step(action)
        np.testing.assert_array_equal(plain_step[0], rendered_step[0])
        assert plain_step[1:] == rendered_step[1:]
        if plain_step[2] or plain_step[3]:
            break

    np.testing.assert_array_equal(plain.position, rendered.position)
    np.testing.assert_array_equal(plain.velocity, rendered.velocity)
    np.testing.assert_array_equal(plain.orientation, rendered.orientation)
    plain.close()
    rendered.close()


class _FakeKeys:
    def __init__(self, *pressed: int) -> None:
        self.pressed = set(pressed)

    def __getitem__(self, key: int) -> bool:
        return key in self.pressed


class _FakeController:
    def __init__(
        self,
        axes: dict[int, int] | None = None,
        buttons: set[int] | None = None,
    ) -> None:
        self.axes = axes or {}
        self.buttons = buttons or set()

    def get_axis(self, axis: int) -> int:
        return self.axes.get(axis, 0)

    def get_button(self, button: int) -> bool:
        return button in self.buttons

    def get_numaxes(self) -> int:
        return 6

    def get_numbuttons(self) -> int:
        return 14


def test_tube_keyboard_control_directions() -> None:
    import pygame

    action = tube_keyboard_action(
        _FakeKeys(pygame.K_UP, pygame.K_LEFT, pygame.K_q, pygame.K_SPACE, pygame.K_LSHIFT)
    )
    np.testing.assert_array_equal(action, [1.0, -1.0, 1.0, 1.0, 1.0])

    opposite = tube_keyboard_action(_FakeKeys(pygame.K_DOWN, pygame.K_RIGHT, pygame.K_e))
    np.testing.assert_array_equal(opposite, [-1.0, 1.0, -1.0, 0.0, 0.0])


def test_playstation_controller_mapping() -> None:
    gamepad = _FakeController(
        axes={0: -32768, 1: -32768, 5: 32767},
        buttons={0, 9},
    )
    action = controller_action(gamepad)  # type: ignore[arg-type]
    np.testing.assert_allclose(action, [1.0, -1.0, 1.0, 1.0, 1.0])

    right_roll = controller_action(_FakeController(buttons={10}))  # type: ignore[arg-type]
    assert right_roll[2] == -1.0


def test_raw_playstation_fallback_mapping() -> None:
    class RawController(_FakeController):
        def get_axis(self, axis: int) -> float:
            return {0: -1.0, 1: -1.0, 5: 1.0}.get(axis, 0.0)

    action = raw_playstation_action(RawController(buttons={1, 4}))  # type: ignore[arg-type]
    np.testing.assert_allclose(action, [1.0, -1.0, 1.0, 1.0, 1.0])


def test_keyboard_and_controller_inputs_can_be_combined() -> None:
    keyboard = np.array([0.2, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    gamepad = np.array([-0.8, 0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    combined = combine_actions(keyboard, gamepad)
    np.testing.assert_allclose(combined, [-0.8, 0.5, 1.0, 1.0, 1.0])
