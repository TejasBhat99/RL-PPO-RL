"""Rocket League-inspired aerial navigation inside a curved tube."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlrl_env.envs._tube_flight_base import _TubeFlightBase


class TubeCarEnv(_TubeFlightBase):
    """Fly to the end of a tube without touching its wall.

    The tube is an analytic 3D centerline with a circular cross-section.  The
    vehicle uses five airborne controls: body-local pitch, yaw, roll, jump,
    and unlimited boost.
    """

    ACTION_NAMES = ("pitch", "yaw", "roll", "jump", "boost")

    def __init__(
        self,
        render_mode: str | None = None,
        tube_length: float = 100.0,
        tube_radius: float = 4.0,
        curvature: float = 0.35,
        procedural_course: bool = False,
        procedural_scale_range: tuple[float, float] = (0.55, 1.0),
        lateral_wavelength_range: tuple[float, float] = (42.0, 70.0),
        vertical_wavelength_range: tuple[float, float] = (55.0, 85.0),
        max_episode_steps: int = 1_200,
        physics_dt: float = 0.02,
        frame_skip: int = 4,
        unlimited_boost: bool = True,
        control_smoothing: float = 0.45,
        render_style: Literal["legacy", "enhanced"] = "legacy",
    ) -> None:
        if tube_length <= 5.0:
            raise ValueError("tube_length must be greater than 5")
        if tube_radius <= 1.0:
            raise ValueError("tube_radius must be greater than 1")
        if not 0.0 <= curvature <= 1.0:
            raise ValueError("curvature must be in [0, 1]")
        self._validate_range("procedural_scale_range", procedural_scale_range, 0.0)
        self._validate_range("lateral_wavelength_range", lateral_wavelength_range, 1.0)
        self._validate_range("vertical_wavelength_range", vertical_wavelength_range, 1.0)
        if render_style not in ("legacy", "enhanced"):
            raise ValueError("render_style must be 'legacy' or 'enhanced'")
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            physics_dt=physics_dt,
            frame_skip=frame_skip,
            unlimited_boost=unlimited_boost,
            control_smoothing=control_smoothing,
        )
        self.tube_length = float(tube_length)
        self.tube_radius = float(tube_radius)
        self.curvature = float(curvature)
        self.procedural_course = procedural_course
        self.procedural_scale_range = tuple(map(float, procedural_scale_range))
        self.lateral_wavelength_range = tuple(map(float, lateral_wavelength_range))
        self.vertical_wavelength_range = tuple(map(float, vertical_wavelength_range))
        self.render_style = render_style
        self._enhanced_renderer: Any | None = None
        self.start_x = -12.0
        self.base_height = 8.0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(47,), dtype=np.float32)
        x_extent = max(abs(self.start_x), abs(self.start_x + self.tube_length)) + 12.0
        self.arena_half_extents = np.array([x_extent, 25.0, 24.0], dtype=np.float32)

        self._lateral_amplitude = 0.0
        self._vertical_amplitude = 0.0
        self._lateral_wavelength = 55.0
        self._vertical_wavelength = 70.0
        self._lateral_phase = 0.0
        self._vertical_phase = 0.0
        self.progress = 0.0
        self.radial_distance = 0.0
        self.crashed = False
        self.reached_goal = False
        self.reward_terms: dict[str, float] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        del options
        self._configure_curve()

        center, tangent, _, _ = self._tube_frame(0.0)
        self.position = center.copy()
        self.velocity = 4.0 * tangent
        self.euler[:] = (
            0.0,
            -float(np.arcsin(np.clip(tangent[2], -1.0, 1.0))),
            float(np.arctan2(tangent[1], tangent[0])),
        )
        self._set_orientation_from_euler()
        self.angular_velocity.fill(0.0)
        self.boost = self.max_boost
        self.grounded = False
        self.episode_steps = 0
        self.path_length = 0.0
        self.control_change_sum = 0.0
        self.cross_track_sum = 0.0
        self._jump_was_pressed = False
        self._last_action.fill(0.0)
        self._smoothed_action.fill(0.0)
        self._quit_requested = False
        self._paused = False
        self._restart_requested = False
        self.progress = 0.0
        self.radial_distance = 0.0
        self.crashed = False
        self.reached_goal = False
        self.reward_terms = {}

        observation = self._get_observation()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        previous_controls = self._smoothed_action[2:5].copy()
        self._smoothed_action[2:5] += self.control_smoothing * (
            action[:3] - self._smoothed_action[2:5]
        )
        self._smoothed_action[:2] = 0.0
        self._smoothed_action[5:] = action[3:]
        effective_action = self._smoothed_action.copy()
        self._last_action = effective_action
        action_change = float(np.mean(np.square(effective_action[2:5] - previous_controls)))

        old_progress = self.progress
        boundary_hits = 0
        for _ in range(self.frame_skip):
            old_position = self.position.copy()
            boundary_hits += self._physics_step(effective_action)
            self.path_length += float(np.linalg.norm(self.position - old_position))

            raw_progress = float(self.position[0] - self.start_x)
            self.progress = float(np.clip(raw_progress, 0.0, self.tube_length))
            _, _, _, _, radial = self._relative_to_tube(self.progress)
            self.radial_distance = float(np.linalg.norm(radial))
            if (
                self.radial_distance + self.car_radius >= self.tube_radius
                or raw_progress < -1.0
                or boundary_hits
            ):
                self.crashed = True
                break
            if raw_progress >= self.tube_length:
                self.reached_goal = True
                break

        self.episode_steps += 1
        terminated = self.reached_goal or self.crashed
        truncated = self.episode_steps >= self.max_episode_steps and not terminated

        usable_radius = self.tube_radius - self.car_radius
        radial_ratio = float(np.clip(self.radial_distance / usable_radius, 0.0, 2.0))
        progress_reward = 0.2 * (self.progress - old_progress)
        step_penalty = -0.01
        centering_penalty = -0.02 * radial_ratio**2
        wall_danger = max(0.0, (radial_ratio - 0.75) / 0.25)
        wall_penalty = -0.10 * wall_danger**2
        smoothness_penalty = -0.01 * action_change
        terminal_reward = 100.0 if self.reached_goal else -100.0 if self.crashed else 0.0
        timeout_penalty = -20.0 if truncated else 0.0
        reward = (
            progress_reward
            + step_penalty
            + centering_penalty
            + wall_penalty
            + smoothness_penalty
            + terminal_reward
            + timeout_penalty
        )
        self.control_change_sum += float(np.sqrt(action_change))
        self.cross_track_sum += self.radial_distance
        self.reward_terms = {
            "progress": progress_reward,
            "step": step_penalty,
            "centering": centering_penalty,
            "wall_danger": wall_penalty,
            "smoothness": smoothness_penalty,
            "terminal": terminal_reward,
            "timeout": timeout_penalty,
        }

        observation = self._get_observation()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def _configure_curve(self) -> None:
        scale = self.curvature
        if self.procedural_course:
            scale *= float(self.np_random.uniform(*self.procedural_scale_range))
            self._lateral_phase = float(self.np_random.uniform(-np.pi, np.pi))
            self._vertical_phase = float(self.np_random.uniform(-np.pi, np.pi))
            self._lateral_wavelength = float(
                self.np_random.uniform(*self.lateral_wavelength_range)
            )
            self._vertical_wavelength = float(
                self.np_random.uniform(*self.vertical_wavelength_range)
            )
        else:
            self._lateral_phase = 0.0
            self._vertical_phase = 0.65
            self._lateral_wavelength = 58.0
            self._vertical_wavelength = 73.0
        self._lateral_amplitude = 6.0 * scale
        self._vertical_amplitude = 3.0 * scale

    @staticmethod
    def _validate_range(name: str, values: tuple[float, float], lower_bound: float) -> None:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        lower, upper = map(float, values)
        if lower < lower_bound or upper < lower or not np.isfinite([lower, upper]).all():
            raise ValueError(
                f"{name} must be finite and satisfy {lower_bound} <= min <= max"
            )

    def _centerline(self, progress: float) -> np.ndarray:
        progress = float(np.clip(progress, 0.0, self.tube_length))
        lateral_angle = 2.0 * np.pi * progress / self._lateral_wavelength
        vertical_angle = 2.0 * np.pi * progress / self._vertical_wavelength
        lateral = self._lateral_amplitude * (
            np.sin(lateral_angle + self._lateral_phase) - np.sin(self._lateral_phase)
        )
        height = self.base_height + self._vertical_amplitude * (
            np.sin(vertical_angle + self._vertical_phase) - np.sin(self._vertical_phase)
        )
        return np.array([self.start_x + progress, lateral, height], dtype=np.float32)

    def _tangent(self, progress: float) -> np.ndarray:
        lateral_angle = 2.0 * np.pi * progress / self._lateral_wavelength
        vertical_angle = 2.0 * np.pi * progress / self._vertical_wavelength
        derivative = np.array(
            [
                1.0,
                self._lateral_amplitude
                * (2.0 * np.pi / self._lateral_wavelength)
                * np.cos(lateral_angle + self._lateral_phase),
                self._vertical_amplitude
                * (2.0 * np.pi / self._vertical_wavelength)
                * np.cos(vertical_angle + self._vertical_phase),
            ],
            dtype=np.float32,
        )
        return self._unit(derivative)

    def _tube_frame(
        self, progress: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center = self._centerline(progress)
        tangent = self._tangent(progress)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = self._unit(np.cross(tangent, world_up))
        up = self._unit(np.cross(right, tangent))
        return center, tangent, right, up

    def _relative_to_tube(
        self, progress: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center, tangent, right, up = self._tube_frame(progress)
        offset = self.position - center
        radial = offset - np.dot(offset, tangent) * tangent
        return center, tangent, right, up, radial

    def _get_observation(self) -> np.ndarray:
        _, tangent, _, _, radial = self._relative_to_tube(self.progress)
        world_to_body = self.orientation.T
        local_velocity = world_to_body @ self.velocity
        local_tangent = world_to_body @ tangent
        local_radial = world_to_body @ radial
        lookahead_offsets = []
        lookahead_tangents = []
        for distance in (8.0, 18.0, 32.0):
            lookahead_progress = min(self.progress + distance, self.tube_length)
            point = self._centerline(lookahead_progress)
            lookahead_offsets.append(
                np.clip((world_to_body @ (point - self.position)) / 32.0, -1.0, 1.0)
            )
            lookahead_tangents.append(
                np.clip(world_to_body @ self._tangent(lookahead_progress), -1.0, 1.0)
            )
        usable_radius = self.tube_radius - self.car_radius
        radial_ratio = float(np.clip(self.radial_distance / usable_radius, 0.0, 1.0))
        if self.radial_distance > 1e-6:
            outward_speed = float(np.dot(self.velocity, radial / self.radial_distance))
        else:
            outward_speed = 0.0
        tangent_speed = float(np.dot(self.velocity, tangent))
        observation = np.concatenate(
            [
                np.clip(local_velocity / self.max_speed, -1.0, 1.0),
                np.sin(self.euler),
                np.cos(self.euler),
                np.clip(self.angular_velocity / self.max_angular_speed, -1.0, 1.0),
                np.array([self.boost / self.max_boost], dtype=np.float32),
                np.clip(local_radial / usable_radius, -1.0, 1.0),
                np.clip(local_tangent, -1.0, 1.0),
                *lookahead_offsets,
                np.array([radial_ratio], dtype=np.float32),
                np.array([2.0 * self.progress / self.tube_length - 1.0], dtype=np.float32),
                self._smoothed_action[2:5],
                np.array([1.0 if self._jump_was_pressed else 0.0], dtype=np.float32),
                np.array([1.0 if self.grounded else 0.0], dtype=np.float32),
                *lookahead_tangents,
                np.array(
                    [outward_speed / self.max_speed, tangent_speed / self.max_speed],
                    dtype=np.float32,
                ),
                np.array(
                    [1.0 if self._last_action[6] > 0.5 else 0.0], dtype=np.float32
                ),
            ]
        )
        return observation.astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "is_success": self.reached_goal,
            "crashed": self.crashed,
            "progress": self.progress,
            "progress_fraction": self.progress / self.tube_length,
            "radial_distance": self.radial_distance,
            "tube_radius": self.tube_radius,
            "path_length": self.path_length,
            "mean_control_change": self.control_change_sum / max(self.episode_steps, 1),
            "mean_radial_error": self.cross_track_sum / max(self.episode_steps, 1),
            "boost": float(self.boost),
            "unlimited_boost": self.unlimited_boost,
            "episode_steps": self.episode_steps,
            "reward_terms": dict(self.reward_terms),
        }

    def _camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a tube-locked chase camera independent of vehicle rotation.

        The policy makes frequent small attitude corrections.  Using those
        corrections to steer the camera makes an otherwise smooth flight look
        unstable, so both tube renderers follow the analytic centerline only.
        """

        progress = float(np.clip(self.progress, 0.0, self.tube_length))
        back_progress = progress - 9.0
        if back_progress >= 0.0:
            back_center, _, _, back_up = self._tube_frame(back_progress)
        else:
            start_center, start_tangent, _, back_up = self._tube_frame(0.0)
            back_center = start_center + back_progress * start_tangent

        current_center = self._centerline(progress)
        _, _, _, _, radial = self._relative_to_tube(progress)
        # Translate with most of the car's cross-tube motion so the vehicle
        # stays visible, but add the identical offset to camera and target.
        # It therefore cannot rotate the view or introduce steering wobble.
        follow_offset = 0.60 * radial
        look_progress = min(progress + 11.0, self.tube_length)
        ahead_center = self._centerline(look_progress)
        # Aim mostly at the car's centerline station and partly through the
        # upcoming bend.  This keeps the vehicle in the lower third without
        # borrowing any of its oscillating pose for camera steering.
        look_at = 0.65 * current_center + 0.35 * ahead_center + follow_offset
        camera_position = back_center + 1.0 * back_up + follow_offset
        camera_forward = self._unit(look_at - camera_position)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        camera_right = np.cross(world_up, camera_forward)
        if float(np.linalg.norm(camera_right)) < 1e-6:
            camera_right = np.cross(
                np.array([0.0, 1.0, 0.0], dtype=np.float32), camera_forward
            )
        camera_right = self._unit(camera_right)
        camera_up = self._unit(np.cross(camera_forward, camera_right))
        return (
            np.asarray(camera_position, dtype=np.float32),
            np.asarray(camera_forward, dtype=np.float32),
            np.asarray(camera_right, dtype=np.float32),
            np.asarray(camera_up, dtype=np.float32),
        )

    def _renderer(self) -> Any:
        if self._enhanced_renderer is None:
            from rlrl_env.rendering.tube_renderer import EnhancedTubeRenderer

            self._enhanced_renderer = EnhancedTubeRenderer(self)
        return self._enhanced_renderer

    def _window_caption(self) -> str:
        style = "Enhanced" if self.render_style == "enhanced" else "Legacy"
        return f"RLRL Tube Flight - {style}"

    def _handle_render_event(self, event: Any, pygame: Any) -> None:
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return

        from rlrl_env.rendering.tube_renderer import playback_button_rects

        size = self._window.get_size() if self._window is not None else (960, 540)
        play_rect, pause_rect, restart_rect = playback_button_rects(pygame, size)
        if play_rect.collidepoint(event.pos):
            self._paused = False
        elif pause_rect.collidepoint(event.pos):
            self._paused = True
        elif restart_rect.collidepoint(event.pos):
            self._restart_requested = True
            self._paused = False

    def _draw_3d_scene(self, canvas: Any, pygame: Any) -> None:
        if self.render_style == "enhanced":
            self._renderer().draw_scene(canvas, pygame)
            return
        self._draw_legacy_3d_scene(canvas, pygame)

    def _draw_legacy_3d_scene(self, canvas: Any, pygame: Any) -> None:
        width, height = canvas.get_size()
        canvas.fill((13, 20, 32))
        pygame.draw.rect(canvas, (20, 31, 46), (0, int(height * 0.52), width, height))
        camera = self._camera()

        progress_values = np.linspace(0.0, self.tube_length, max(12, int(self.tube_length / 4)))
        cross_sections: list[list[np.ndarray]] = []
        angles = np.linspace(0.0, 2.0 * np.pi, 17)
        for progress in progress_values:
            center, _, right, up = self._tube_frame(float(progress))
            points = [
                center + self.tube_radius * (np.cos(angle) * right + np.sin(angle) * up)
                for angle in angles
            ]
            cross_sections.append(points)
            color = (54, 175, 210) if progress > self.progress else (43, 75, 91)
            for start, end in zip(points, points[1:]):
                self._draw_3d_line(canvas, pygame, camera, start, end, color, 2)
        for angle_index in range(0, 16, 4):
            for first, second in zip(cross_sections, cross_sections[1:]):
                self._draw_3d_line(
                    canvas, pygame, camera, first[angle_index], second[angle_index],
                    (34, 103, 128), 1
                )
        self._renderer().draw_vehicle(canvas, pygame, camera)

    def _draw_minimap(self, canvas: Any, pygame: Any) -> None:
        if self.render_style == "enhanced":
            self._renderer().draw_minimap(canvas, pygame)
            return
        self._draw_legacy_minimap(canvas, pygame)

    def _draw_legacy_minimap(self, canvas: Any, pygame: Any) -> None:
        map_rect = pygame.Rect(canvas.get_width() - 245, 18, 225, 135)
        pygame.draw.rect(canvas, (8, 14, 24), map_rect, border_radius=8)
        pygame.draw.rect(canvas, (70, 103, 121), map_rect, width=2, border_radius=8)
        samples = [self._centerline(s) for s in np.linspace(0.0, self.tube_length, 80)]
        points = []
        for point in samples:
            x = map_rect.left + 10 + (point[0] - self.start_x) / self.tube_length * (map_rect.width - 20)
            y = map_rect.centery - point[1] / 15.0 * (map_rect.height * 0.42)
            points.append((int(x), int(y)))
        pygame.draw.lines(canvas, (57, 157, 189), False, points, width=3)
        car_x = map_rect.left + 10 + self.progress / self.tube_length * (map_rect.width - 20)
        car_y = map_rect.centery - self.position[1] / 15.0 * (map_rect.height * 0.42)
        pygame.draw.circle(canvas, (255, 174, 65), (int(car_x), int(car_y)), 5)

    def _draw_hud(self, canvas: Any, pygame: Any) -> None:
        if self.render_style == "enhanced":
            self._renderer().draw_hud(canvas, pygame)
        else:
            self._draw_legacy_hud(canvas, pygame)
        self._renderer().draw_playback_controls(canvas, pygame)

    def _draw_legacy_hud(self, canvas: Any, pygame: Any) -> None:
        pygame.font.init()
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 21)
        speed = float(np.linalg.norm(self.velocity))
        status = (
            f"TUBE {100.0 * self.progress / self.tube_length:05.1f}%   "
            f"SPEED {speed:04.1f}   WALL {self.tube_radius - self.radial_distance:03.1f}"
        )
        canvas.blit(font.render(status, True, (232, 238, 242)), (22, 18))
        canvas.blit(
            small_font.render("BOOST UNLIMITED", True, (94, 193, 247)),
            (22, 51),
        )
