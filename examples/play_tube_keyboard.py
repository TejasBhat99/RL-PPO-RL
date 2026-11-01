"""Fly through the tube with a keyboard or SDL-compatible controller."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pygame
from pygame._sdl2 import controller
from typing import Any

import rlrl_env  # noqa: F401 - registers TubeCar-v0


def tube_keyboard_action(keys: pygame.key.ScancodeWrapper) -> np.ndarray:
    """Map the requested airborne-only keyboard controls."""

    pitch = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
    yaw = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
    roll = float(keys[pygame.K_q]) - float(keys[pygame.K_e])
    jump = float(keys[pygame.K_SPACE])
    boost = float(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
    return np.array([pitch, yaw, roll, jump, boost], dtype=np.float32)


def _stick_value(raw_value: int, deadzone: float = 0.08) -> float:
    value = float(np.clip(raw_value / 32767.0, -1.0, 1.0))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return float(np.sign(value) * (magnitude - deadzone) / (1.0 - deadzone))


def controller_action(gamepad: controller.Controller) -> np.ndarray:
    """Map the standardized SDL game-controller layout.

    SDL maps PlayStation Cross to A (button 0), L1/R1 to shoulder buttons
    9/10, and R2 to the right-trigger axis (axis 5).
    """

    left_x = _stick_value(gamepad.get_axis(0))
    left_y = _stick_value(gamepad.get_axis(1))
    yaw = left_x
    pitch = -left_y
    roll = float(gamepad.get_button(9)) - float(gamepad.get_button(10))
    jump = float(gamepad.get_button(0))
    boost = float(np.clip(gamepad.get_axis(5) / 32767.0, 0.0, 1.0))
    return np.array([pitch, yaw, roll, jump, boost], dtype=np.float32)


def raw_playstation_action(gamepad: pygame.joystick.JoystickType) -> np.ndarray:
    """Fallback for Linux PlayStation devices without an SDL mapping.

    Raw DualShock/DualSense layouts commonly expose Cross as button 1,
    L1/R1 as buttons 4/5, and R2 as axis 5.
    """

    def axis(index: int) -> float:
        return float(gamepad.get_axis(index)) if gamepad.get_numaxes() > index else 0.0

    def button(index: int) -> float:
        return float(gamepad.get_button(index)) if gamepad.get_numbuttons() > index else 0.0

    yaw = _stick_value(int(axis(0) * 32767.0))
    pitch = -_stick_value(int(axis(1) * 32767.0))
    roll = button(4) - button(5)
    jump = button(1)
    raw_trigger = axis(5)
    boost = float(np.clip((raw_trigger + 1.0) * 0.5, 0.0, 1.0))
    return np.array([pitch, yaw, roll, jump, boost], dtype=np.float32)


def combine_actions(keyboard: np.ndarray, gamepad: np.ndarray | None) -> np.ndarray:
    """Use the strongest input per axis and combine binary buttons."""

    if gamepad is None:
        return keyboard
    combined = keyboard.copy()
    use_gamepad = np.abs(gamepad[:3]) > np.abs(combined[:3])
    combined[:3][use_gamepad] = gamepad[:3][use_gamepad]
    combined[3:] = np.maximum(combined[3:], gamepad[3:])
    return combined


def find_input_device() -> tuple[str, Any] | None:
    controller.init()
    pygame.joystick.init()
    for index in range(controller.get_count()):
        if controller.is_controller(index):
            gamepad = controller.Controller(index)
            print(f"Controller connected: {gamepad.name}")
            return "sdl", gamepad
    for index in range(pygame.joystick.get_count()):
        gamepad = pygame.joystick.Joystick(index)
        gamepad.init()
        print(f"Raw PlayStation joystick connected: {gamepad.get_name()}")
        return "raw", gamepad
    print("No controller device detected; keyboard remains active.")
    return None


def main() -> None:
    env = gym.make(
        "TubeCar-v0",
        render_mode="human",
        tube_length=100.0,
        tube_radius=4.0,
        curvature=0.35,
        frame_skip=1,
        max_episode_steps=20_000,
        render_style="enhanced",
    )
    env.reset(seed=7)
    input_device = find_input_device()
    total_reward = 0.0
    print("Keyboard: Up nose down  Down nose up  Left/Right yaw")
    print("Keyboard: Q left roll  E right roll  Space jump  Shift boost")
    print("Controller: left stick yaw/pitch  Cross jump  L1/R1 roll  R2 boost")
    print("R reset  Esc quit")

    running = True
    while running and not env.unwrapped.quit_requested:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break
        if keys[pygame.K_r] or env.unwrapped.consume_restart_request():
            env.reset()
            total_reward = 0.0
            continue
        if env.unwrapped.paused:
            env.render()
            continue

        controller.update()
        gamepad_input = None
        if input_device is not None:
            input_kind, gamepad = input_device
            if input_kind == "sdl" and gamepad.attached():
                gamepad_input = controller_action(gamepad)
            elif input_kind == "raw" and gamepad.get_init():
                gamepad_input = raw_playstation_action(gamepad)
        action = combine_actions(tube_keyboard_action(keys), gamepad_input)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            outcome = "goal" if info["is_success"] else "crash/timeout"
            print(f"Episode ended: {outcome}; reward={total_reward:.2f}")
            env.reset()
            total_reward = 0.0
    if input_device is not None:
        _, gamepad = input_device
        gamepad.quit()
    env.close()


if __name__ == "__main__":
    main()
