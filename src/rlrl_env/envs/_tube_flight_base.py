"""Shared physics and viewer support for the tube environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class _TubeFlightBase(gym.Env[np.ndarray, np.ndarray]):
    """Private base containing tube flight physics and rendering plumbing."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None,
        max_episode_steps: int,
        physics_dt: float,
        frame_skip: int,
        unlimited_boost: bool,
        control_smoothing: float,
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if physics_dt <= 0 or frame_skip < 1 or max_episode_steps < 1:
            raise ValueError("Time and episode parameters must be positive")
        if not 0.0 < control_smoothing <= 1.0:
            raise ValueError("control_smoothing must be in (0, 1]")

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.physics_dt = physics_dt
        self.frame_skip = frame_skip
        self.unlimited_boost = unlimited_boost
        self.control_smoothing = control_smoothing

        self.car_radius = 0.35
        self.max_speed = 30.0
        self.max_angular_speed = 5.0
        self.max_boost = 100.0

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.euler = np.zeros(3, dtype=np.float32)
        self.orientation = np.eye(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.boost = self.max_boost
        self.grounded = True
        self.episode_steps = 0
        self.path_length = 0.0
        self.control_change_sum = 0.0
        self.cross_track_sum = 0.0
        self._jump_was_pressed = False
        # The public tube action has five values, but the physics layout keeps
        # two unused ground-control slots followed by pitch/yaw/roll/jump/boost.
        self._last_action = np.zeros(7, dtype=np.float32)
        self._smoothed_action = np.zeros(7, dtype=np.float32)
        self._quit_requested = False
        self._paused = False
        self._restart_requested = False
        self._window: Any | None = None
        self._clock: Any | None = None

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    @property
    def paused(self) -> bool:
        return self._paused

    def consume_restart_request(self) -> bool:
        requested = self._restart_requested
        self._restart_requested = False
        return requested

    def _physics_step(self, action: np.ndarray) -> int:
        throttle, steer, pitch, yaw, roll, jump, boost_action = action
        dt = self.physics_dt

        forward = self._forward_vector()
        forward_flat = np.array([forward[0], forward[1], 0.0], dtype=np.float32)
        forward_flat /= max(float(np.linalg.norm(forward_flat)), 1e-6)
        right_flat = np.array([-forward_flat[1], forward_flat[0], 0.0], dtype=np.float32)

        acceleration = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        if self.grounded:
            acceleration += forward_flat * (16.0 * throttle)
            forward_speed = float(np.dot(self.velocity, forward_flat))
            lateral_speed = float(np.dot(self.velocity, right_flat))
            acceleration -= right_flat * (8.0 * lateral_speed)
            acceleration[:2] -= 0.45 * self.velocity[:2]
            steering_scale = np.clip(abs(forward_speed) / 4.0, 0.2, 1.0)
            self.angular_velocity[2] = 2.2 * steer * steering_scale
            self.angular_velocity[:2] *= max(0.0, 1.0 - 8.0 * dt)

            jump_pressed = bool(jump > 0.5)
            if jump_pressed and not self._jump_was_pressed:
                self.velocity[2] = 7.5
                self.grounded = False
            self._jump_was_pressed = jump_pressed
        else:
            target_rates = np.array(
                [4.5 * roll, 3.5 * pitch, 3.2 * yaw], dtype=np.float32
            )
            self.angular_velocity += 4.0 * (target_rates - self.angular_velocity) * dt
            acceleration += forward * (1.5 * throttle)
            self._jump_was_pressed = bool(jump > 0.5)

        if boost_action > 0.5 and self.boost > 0.0:
            acceleration += forward * 24.0
            if not self.unlimited_boost:
                self.boost = max(0.0, self.boost - 25.0 * dt)
        if self.unlimited_boost:
            self.boost = self.max_boost

        self.angular_velocity = np.clip(
            self.angular_velocity, -self.max_angular_speed, self.max_angular_speed
        )
        if self.grounded:
            self.euler[2] += self.angular_velocity[2] * dt
            self.euler = (self.euler + np.pi) % (2.0 * np.pi) - np.pi
            self._set_orientation_from_euler()
        else:
            self.orientation = self.orientation @ self._rotation_increment(
                self.angular_velocity * dt
            )
            body_forward = self._unit(self.orientation[:, 0])
            body_left = self.orientation[:, 1] - np.dot(
                self.orientation[:, 1], body_forward
            ) * body_forward
            body_left = self._unit(body_left)
            body_up = self._unit(np.cross(body_forward, body_left))
            self.orientation = np.column_stack(
                (body_forward, body_left, body_up)
            ).astype(np.float32)
            self._sync_euler_from_orientation()

        self.velocity += acceleration * dt
        speed = float(np.linalg.norm(self.velocity))
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        self.position += self.velocity * dt

        if self.position[2] <= self.car_radius:
            self.position[2] = self.car_radius
            self.velocity[2] = max(0.0, float(self.velocity[2]))
            self.grounded = True
            self.euler[:2] *= max(0.0, 1.0 - 10.0 * dt)
            self.angular_velocity[:2] *= max(0.0, 1.0 - 10.0 * dt)
            self._set_orientation_from_euler()

        boundary_hit = 0
        for axis in (0, 1):
            lower = -self.arena_half_extents[axis]
            upper = self.arena_half_extents[axis]
            if self.position[axis] < lower:
                self.position[axis] = lower
                self.velocity[axis] = abs(self.velocity[axis]) * 0.3
                boundary_hit = 1
            elif self.position[axis] > upper:
                self.position[axis] = upper
                self.velocity[axis] = -abs(self.velocity[axis]) * 0.3
                boundary_hit = 1
        if self.position[2] > self.arena_half_extents[2]:
            self.position[2] = self.arena_half_extents[2]
            self.velocity[2] = -abs(self.velocity[2]) * 0.3
            boundary_hit = 1
        return boundary_hit

    def _set_orientation_from_euler(self) -> None:
        roll, pitch, yaw = self.euler
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        self.orientation = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float32,
        )

    def _rotation_matrix(self) -> np.ndarray:
        return self.orientation

    @staticmethod
    def _rotation_increment(rotation_vector: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-8:
            return np.eye(3, dtype=np.float32)
        x, y, z = rotation_vector / angle
        skew = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32
        )
        return (
            np.eye(3, dtype=np.float32)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        ).astype(np.float32)

    def _sync_euler_from_orientation(self) -> None:
        matrix = self.orientation
        pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        self.euler[:] = (roll, pitch, yaw)

    def _forward_vector(self) -> np.ndarray:
        return self._rotation_matrix()[:, 0]

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        try:
            import pygame
        except ImportError as error:  # pragma: no cover
            raise gym.error.DependencyNotInstalled("Install pygame to render") from error

        width, height = 960, 540
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((width, height))
            pygame.display.set_caption(self._window_caption())
        if self._clock is None:
            self._clock = pygame.time.Clock()

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit_requested = True
                else:
                    self._handle_render_event(event, pygame)

        canvas = pygame.Surface((width, height))
        self._draw_3d_scene(canvas, pygame)
        self._draw_minimap(canvas, pygame)
        self._draw_hud(canvas, pygame)

        if self.render_mode == "human":
            assert self._window is not None
            self._window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
            return None

        pixels = pygame.surfarray.array3d(canvas)
        return np.transpose(pixels, axes=(1, 0, 2))

    def _window_caption(self) -> str:
        return "RLRL Tube Flight"

    def _handle_render_event(self, event: Any, pygame: Any) -> None:
        del event, pygame

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    @staticmethod
    def _project_3d(
        point: np.ndarray,
        camera: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], float] | None:
        camera_position, camera_forward, camera_right, camera_up = camera
        relative = point - camera_position
        depth = float(np.dot(relative, camera_forward))
        if depth <= 0.2:
            return None
        width, height = size
        focal_length = 0.82 * width
        screen_x = width * 0.5 + focal_length * float(np.dot(relative, camera_right)) / depth
        screen_y = height * 0.52 - focal_length * float(np.dot(relative, camera_up)) / depth
        return (int(screen_x), int(screen_y)), depth

    def _draw_3d_line(
        self,
        canvas: Any,
        pygame: Any,
        camera: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        projected_start = self._project_3d(start, camera, canvas.get_size())
        projected_end = self._project_3d(end, camera, canvas.get_size())
        if projected_start is not None and projected_end is not None:
            pygame.draw.line(
                canvas, color, projected_start[0], projected_end[0], width=width
            )

    def close(self) -> None:
        if self._window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
