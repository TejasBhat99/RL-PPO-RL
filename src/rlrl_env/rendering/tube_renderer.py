"""Stylized Pygame renderer for :class:`~rlrl_env.envs.tube_car.TubeCarEnv`.

The renderer is deliberately separate from the environment dynamics.  It only
reads public and private simulation state needed to compose a frame, and it
never consumes the environment's random generator.  Consequently enabling the
renderer cannot change a seeded rollout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


Color = tuple[int, int, int]
Camera = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def playback_button_rects(
    pygame: Any, size: tuple[int, int]
) -> tuple[Any, Any, Any]:
    """Return the shared play, pause, and restart hit rectangles.

    Keeping this geometry outside the renderer lets the environment's mouse
    event handling use the exact same targets that are drawn on screen.
    """

    width, height = size
    button_height = 38
    gap = 12
    play_width, pause_width, restart_width = 88, 94, 116
    total_width = play_width + pause_width + restart_width + 2 * gap
    left = (width - total_width) // 2
    # Keep transport controls away from the chase car at the lower center.
    top = 62
    play = pygame.Rect(left, top, play_width, button_height)
    pause = pygame.Rect(play.right + gap, top, pause_width, button_height)
    restart = pygame.Rect(pause.right + gap, top, restart_width, button_height)
    return play, pause, restart


class EnhancedTubeRenderer:
    """Draw a filled neon tunnel, vehicle, minimap, and flight HUD."""

    WIDTH = 960
    HEIGHT = 540
    _NEAR_PLANE = 0.35

    def __init__(self, env: Any) -> None:
        self.env = env
        self._fonts: dict[tuple[int, bool], Any] = {}

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    @staticmethod
    def _mix(first: Color, second: Color, amount: float) -> Color:
        amount = float(np.clip(amount, 0.0, 1.0))
        return tuple(
            int(np.clip(round(a + (b - a) * amount), 0, 255))
            for a, b in zip(first, second)
        )

    @staticmethod
    def _scale(color: Color, amount: float) -> Color:
        return tuple(int(np.clip(round(channel * amount), 0, 255)) for channel in color)

    def _font(self, pygame: Any, size: int, *, bold: bool = False) -> Any:
        key = (size, bold)
        if key not in self._fonts:
            pygame.font.init()
            font = pygame.font.Font(None, size)
            font.set_bold(bold)
            self._fonts[key] = font
        return self._fonts[key]

    def _camera(self) -> Camera:
        """Use the same centerline-locked camera as the legacy tube view."""

        return self.env._camera()

    def _project(
        self,
        point: np.ndarray,
        camera: Camera,
        size: tuple[int, int],
    ) -> tuple[tuple[int, int], float] | None:
        camera_position, camera_forward, camera_right, camera_up = camera
        relative = np.asarray(point, dtype=np.float32) - camera_position
        depth = float(np.dot(relative, camera_forward))
        if depth <= self._NEAR_PLANE:
            return None
        width, height = size
        focal_length = 0.88 * width
        screen_x = width * 0.5 + focal_length * float(np.dot(relative, camera_right)) / depth
        screen_y = height * 0.53 - focal_length * float(np.dot(relative, camera_up)) / depth
        # Clamping protects SDL's integer rasterizer while retaining polygons that
        # cross the viewport from just outside the near plane.
        screen_x = float(np.clip(screen_x, -4.0 * width, 5.0 * width))
        screen_y = float(np.clip(screen_y, -4.0 * height, 5.0 * height))
        return (int(screen_x), int(screen_y)), depth

    def _draw_background(self, canvas: Any, pygame: Any) -> None:
        width, height = canvas.get_size()
        upper = (5, 10, 22)
        lower = (10, 25, 38)
        band_height = 4
        for y in range(0, height, band_height):
            color = self._mix(upper, lower, y / max(height - 1, 1))
            pygame.draw.rect(canvas, color, (0, y, width, band_height))

        # Fixed arithmetic positions give a little life to gaps in the mesh
        # without touching either NumPy's or the environment's random state.
        for index in range(72):
            x = (53 + index * 137 + index * index * 11) % width
            y = (17 + index * 71 + index * index * 7) % height
            brightness = 48 + (index * 29) % 92
            radius = 1 if index % 7 else 2
            pygame.draw.circle(canvas, (brightness,) * 3, (x, y), radius)

    def _tube_samples(self) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
        start = max(0.0, float(self.env.progress) - 1.5)
        end = min(float(self.env.tube_length), float(self.env.progress) + 82.0)
        count = max(3, int(np.ceil((end - start) / 3.8)) + 1)
        progresses = np.linspace(start, end, count, dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * np.pi, 21, dtype=np.float32)
        cross_sections: list[list[np.ndarray]] = []
        for progress in progresses:
            center, _, right, up = self.env._tube_frame(float(progress))
            cross_sections.append(
                [
                    center
                    + float(self.env.tube_radius)
                    * (float(np.cos(angle)) * right + float(np.sin(angle)) * up)
                    for angle in angles
                ]
            )
        return progresses, angles, cross_sections

    def _draw_goal_backdrop(
        self, canvas: Any, pygame: Any, camera: Camera, distance_to_goal: float
    ) -> None:
        if distance_to_goal > 86.0:
            return
        center, _, right, up = self.env._tube_frame(float(self.env.tube_length))
        points: list[tuple[int, int]] = []
        depths: list[float] = []
        for angle in np.linspace(0.0, 2.0 * np.pi, 41):
            world = center + 0.82 * float(self.env.tube_radius) * (
                float(np.cos(angle)) * right + float(np.sin(angle)) * up
            )
            projected = self._project(world, camera, canvas.get_size())
            if projected is None:
                return
            points.append(projected[0])
            depths.append(projected[1])
        fog = float(np.clip(np.mean(depths) / 90.0, 0.0, 1.0))
        pygame.draw.polygon(canvas, self._mix((10, 62, 58), (5, 13, 24), fog), points)

    def _draw_tube_mesh(
        self,
        canvas: Any,
        pygame: Any,
        camera: Camera,
        progresses: np.ndarray,
        angles: np.ndarray,
        cross_sections: list[list[np.ndarray]],
    ) -> None:
        panels: list[tuple[float, list[tuple[int, int]], Color, int, int]] = []
        viewport = canvas.get_rect().inflate(520, 400)
        # Project every mesh vertex once. Re-projecting the four shared corners
        # of every panel makes a software renderer unnecessarily slow.
        projected_sections = [
            [self._project(point, camera, canvas.get_size()) for point in section]
            for section in cross_sections
        ]
        for axial_index in range(len(cross_sections) - 1):
            for angular_index in range(len(angles) - 1):
                projected = (
                    projected_sections[axial_index][angular_index],
                    projected_sections[axial_index + 1][angular_index],
                    projected_sections[axial_index + 1][angular_index + 1],
                    projected_sections[axial_index][angular_index + 1],
                )
                if any(item is None for item in projected):
                    continue
                projected_points = [item[0] for item in projected if item is not None]
                panel_bounds = pygame.Rect(
                    min(point[0] for point in projected_points),
                    min(point[1] for point in projected_points),
                    max(1, max(point[0] for point in projected_points) - min(point[0] for point in projected_points)),
                    max(1, max(point[1] for point in projected_points) - min(point[1] for point in projected_points)),
                )
                if not viewport.colliderect(panel_bounds):
                    continue
                depth = float(np.mean([item[1] for item in projected if item is not None]))
                middle_angle = 0.5 * float(
                    angles[angular_index] + angles[angular_index + 1]
                )
                light = (
                    0.73
                    + 0.14 * float(np.sin(middle_angle))
                    + 0.07 * float(np.cos(middle_angle - 0.5))
                )
                checker = 1.08 if (axial_index + angular_index) % 2 == 0 else 0.91
                base: Color = (21, 70, 89)
                if np.sin(middle_angle) < -0.72:
                    base = (25, 77, 96)
                color = self._scale(base, light * checker)
                fog = float(np.clip((depth - 18.0) / 80.0, 0.0, 0.78))
                color = self._mix(color, (6, 15, 27), fog)
                panels.append(
                    (depth, projected_points, color, axial_index, angular_index)
                )

        for _, points, color, axial_index, angular_index in sorted(
            panels, key=lambda item: item[0], reverse=True
        ):
            pygame.draw.polygon(canvas, color, points)
            if axial_index % 3 == 0 and angular_index % 3 == 0:
                pygame.draw.aalines(
                    canvas, self._mix(color, (50, 123, 142), 0.22), True, points
                )

    def _draw_tube_structure(
        self,
        canvas: Any,
        pygame: Any,
        camera: Camera,
        progresses: np.ndarray,
        cross_sections: list[list[np.ndarray]],
        glow: Any,
    ) -> None:
        # Longitudinal rails make angular velocity and air roll immediately legible.
        rail_indices = tuple(
            int(index * (len(cross_sections[0]) - 1) / 6) for index in range(6)
        )
        for angular_index in rail_indices:
            points: list[tuple[int, int]] = []
            depths: list[float] = []
            for section in cross_sections:
                projected = self._project(
                    section[angular_index], camera, canvas.get_size()
                )
                if projected is not None:
                    points.append(projected[0])
                    depths.append(projected[1])
            if len(points) < 2:
                continue
            fog = float(np.clip(np.mean(depths) / 95.0, 0.0, 0.75))
            color = self._mix((35, 170, 202), (12, 45, 66), fog)
            pygame.draw.lines(glow, (*color, 42), False, points, width=7)
            pygame.draw.aalines(canvas, color, False, points)

        first_rib = np.ceil(float(progresses[0]) / 11.0) * 11.0
        for progress in np.arange(first_rib, float(progresses[-1]) + 0.1, 11.0):
            center, _, right, up = self.env._tube_frame(float(progress))
            points: list[tuple[int, int]] = []
            depths: list[float] = []
            complete = True
            for angle in np.linspace(0.0, 2.0 * np.pi, 33):
                world = center + float(self.env.tube_radius) * (
                    float(np.cos(angle)) * right + float(np.sin(angle)) * up
                )
                projected = self._project(world, camera, canvas.get_size())
                if projected is None:
                    complete = False
                    break
                points.append(projected[0])
                depths.append(projected[1])
            if not complete:
                continue
            fog = float(np.clip(np.mean(depths) / 90.0, 0.0, 0.78))
            color = self._mix((64, 188, 215), (14, 44, 62), fog)
            pygame.draw.lines(glow, (*color, 50), True, points, width=8)
            pygame.draw.aalines(canvas, color, True, points)

    def _draw_chevrons(
        self, canvas: Any, pygame: Any, camera: Camera, glow: Any
    ) -> None:
        first = np.ceil((float(self.env.progress) + 5.0) / 10.0) * 10.0
        limit = min(float(self.env.tube_length), float(self.env.progress) + 72.0)
        for progress in np.arange(first, limit + 0.1, 10.0):
            center, tangent, right, up = self.env._tube_frame(float(progress))
            floor = center - (float(self.env.tube_radius) - 0.035) * up
            world_points = (
                floor + 1.25 * tangent,
                floor - 0.85 * tangent + 0.72 * right,
                floor - 0.25 * tangent,
                floor - 0.85 * tangent - 0.72 * right,
            )
            projected = [
                self._project(point, camera, canvas.get_size()) for point in world_points
            ]
            if any(item is None for item in projected):
                continue
            points = [item[0] for item in projected if item is not None]
            depth = float(np.mean([item[1] for item in projected if item is not None]))
            fog = float(np.clip(depth / 85.0, 0.0, 0.82))
            color = self._mix((52, 224, 235), (13, 54, 69), fog)
            pygame.draw.polygon(glow, (*color, 48), points)
            pygame.draw.polygon(canvas, color, points)

    def _draw_goal_portal(
        self, canvas: Any, pygame: Any, camera: Camera, glow: Any
    ) -> None:
        distance_to_goal = float(self.env.tube_length - self.env.progress)
        if distance_to_goal > 86.0:
            return
        center, _, right, up = self.env._tube_frame(float(self.env.tube_length))
        for inset, intensity in ((0.0, 1.0), (0.55, 0.75), (1.05, 0.50)):
            portal_center = center - inset * self.env._tangent(float(self.env.tube_length))
            radius = float(self.env.tube_radius) * (0.90 - 0.04 * inset)
            points: list[tuple[int, int]] = []
            for angle in np.linspace(0.0, 2.0 * np.pi, 41):
                world = portal_center + radius * (
                    float(np.cos(angle)) * right + float(np.sin(angle)) * up
                )
                projected = self._project(world, camera, canvas.get_size())
                if projected is None:
                    points = []
                    break
                points.append(projected[0])
            if points:
                color = self._scale((78, 255, 187), intensity)
                pygame.draw.lines(glow, (*color, 75), True, points, width=12)
                pygame.draw.lines(canvas, color, True, points, width=2)

        projected_center = self._project(center, camera, canvas.get_size())
        if projected_center is not None and projected_center[1] < 90.0:
            label = self._font(pygame, 18, bold=True).render(
                "EXIT", True, (183, 255, 220)
            )
            label_rect = label.get_rect(center=(projected_center[0][0], projected_center[0][1]))
            canvas.blit(label, label_rect)

    def _draw_boost(
        self, canvas: Any, pygame: Any, camera: Camera, glow: Any
    ) -> None:
        last_action = np.asarray(getattr(self.env, "_last_action", np.zeros(7)))
        if last_action.size < 7 or float(last_action[6]) <= 0.5:
            return
        forward = np.asarray(self.env.orientation[:, 0], dtype=np.float32)
        tail = np.asarray(self.env.position, dtype=np.float32) - 0.86 * forward
        pulse = 0.16 * np.sin(float(self.env.episode_steps) * 1.71)
        flame_end = tail - (1.75 + pulse) * forward
        projected_tail = self._project(tail, camera, canvas.get_size())
        projected_end = self._project(flame_end, camera, canvas.get_size())
        if projected_tail is None or projected_end is None:
            return
        pygame.draw.line(
            glow, (39, 164, 255, 100), projected_tail[0], projected_end[0], width=24
        )
        pygame.draw.line(
            canvas, (49, 166, 255), projected_tail[0], projected_end[0], width=10
        )
        pygame.draw.line(
            canvas, (207, 248, 255), projected_tail[0], projected_end[0], width=3
        )

    def _mesh_faces(
        self,
        local_vertices: np.ndarray,
        faces: tuple[tuple[tuple[int, ...], Color], ...],
        camera: Camera,
        size: tuple[int, int],
    ) -> list[tuple[float, list[tuple[int, int]], Color]]:
        orientation = np.asarray(self.env.orientation, dtype=np.float32)
        world_vertices = (
            np.asarray(self.env.position, dtype=np.float32)
            + local_vertices @ orientation.T
        )
        projected = [self._project(vertex, camera, size) for vertex in world_vertices]
        output: list[tuple[float, list[tuple[int, int]], Color]] = []
        for indices, color in faces:
            vertices = [projected[index] for index in indices]
            if any(vertex is None for vertex in vertices):
                continue
            output.append(
                (
                    float(np.mean([vertex[1] for vertex in vertices if vertex is not None])),
                    [vertex[0] for vertex in vertices if vertex is not None],
                    color,
                )
            )
        return output

    def _cuboid_faces(
        self,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        color: Color,
        camera: Camera,
        canvas_size: tuple[int, int],
    ) -> list[tuple[float, list[tuple[int, int]], Color]]:
        """Build a shaded body-space cuboid and return projected faces."""

        center_array = np.asarray(center, dtype=np.float32)
        half = 0.5 * np.asarray(size, dtype=np.float32)
        vertices = np.array(
            [
                center_array + half * direction
                for direction in (
                    (-1.0, -1.0, -1.0),
                    (-1.0, 1.0, -1.0),
                    (1.0, 1.0, -1.0),
                    (1.0, -1.0, -1.0),
                    (-1.0, -1.0, 1.0),
                    (-1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (1.0, -1.0, 1.0),
                )
            ],
            dtype=np.float32,
        )
        faces = (
            ((0, 1, 2, 3), self._scale(color, 0.40)),
            ((0, 4, 5, 1), self._scale(color, 0.72)),
            ((3, 2, 6, 7), self._scale(color, 1.08)),
            ((0, 3, 7, 4), self._scale(color, 0.82)),
            ((1, 5, 6, 2), self._scale(color, 0.94)),
            ((4, 7, 6, 5), self._scale(color, 1.16)),
        )
        return self._mesh_faces(vertices, faces, camera, canvas_size)

    def _wheel_faces(
        self,
        center: tuple[float, float, float],
        camera: Camera,
        canvas_size: tuple[int, int],
    ) -> list[tuple[float, list[tuple[int, int]], Color]]:
        """Return a chunky ten-sided wheel with exposed metallic hubs."""

        segments = 10
        center_x, center_y, center_z = center
        radius = 0.285
        half_width = 0.115
        vertices: list[list[float]] = []
        for side in (-half_width, half_width):
            for angle in np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False):
                vertices.append(
                    [
                        center_x + radius * float(np.cos(angle)),
                        center_y + side,
                        center_z + radius * float(np.sin(angle)),
                    ]
                )
        # Slightly offset hub discs avoid z-fighting with the tire sidewalls.
        for side in (-half_width - 0.006, half_width + 0.006):
            for angle in np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False):
                vertices.append(
                    [
                        center_x + 0.47 * radius * float(np.cos(angle)),
                        center_y + side,
                        center_z + 0.47 * radius * float(np.sin(angle)),
                    ]
                )

        faces: list[tuple[tuple[int, ...], Color]] = []
        for index in range(segments):
            following = (index + 1) % segments
            tread = (16, 20, 23) if index % 2 else (23, 28, 31)
            faces.append(((index, following, segments + following, segments + index), tread))
        faces.extend(
            (
                (tuple(range(segments - 1, -1, -1)), (12, 15, 18)),
                (tuple(range(segments, 2 * segments)), (15, 18, 21)),
                (tuple(range(2 * segments, 3 * segments)), (112, 127, 132)),
                (tuple(range(3 * segments, 4 * segments)), (129, 145, 150)),
            )
        )
        return self._mesh_faces(
            np.asarray(vertices, dtype=np.float32), tuple(faces), camera, canvas_size
        )

    def _draw_rear_lights(self, canvas: Any, pygame: Any, camera: Camera) -> None:
        orientation = np.asarray(self.env.orientation, dtype=np.float32)
        position = np.asarray(self.env.position, dtype=np.float32)
        light_glow = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        crisp_lights: list[list[tuple[int, int]]] = []
        for y_min, y_max in ((-0.36, -0.13), (0.13, 0.36)):
            local_points = np.array(
                [
                    [-0.976, y_min, -0.01],
                    [-0.976, y_max, -0.01],
                    [-0.976, y_max, 0.12],
                    [-0.976, y_min, 0.12],
                ],
                dtype=np.float32,
            )
            projected = [
                self._project(position + point @ orientation.T, camera, canvas.get_size())
                for point in local_points
            ]
            if any(item is None for item in projected):
                continue
            points = [item[0] for item in projected if item is not None]
            pygame.draw.polygon(light_glow, (255, 38, 22, 110), points)
            crisp_lights.append(points)
        canvas.blit(light_glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        for points in crisp_lights:
            pygame.draw.polygon(canvas, (255, 53, 31), points)
            pygame.draw.aalines(canvas, (255, 171, 92), True, points)

    def _draw_car(self, canvas: Any, pygame: Any, camera: Camera) -> None:
        """Draw a compact, high-cab, low-poly battle car."""

        canvas_size = canvas.get_size()
        orange: Color = (226, 77, 30)
        bright_orange: Color = (255, 116, 39)
        dark_trim: Color = (39, 43, 45)
        glass: Color = (18, 48, 61)
        faces: list[tuple[float, list[tuple[int, int]], Color]] = []

        # Four genuinely three-dimensional wheels sit outside the bodywork.
        for wheel_center in (
            (-0.61, -0.55, -0.18),
            (-0.61, 0.55, -0.18),
            (0.62, -0.55, -0.17),
            (0.62, 0.55, -0.17),
        ):
            faces.extend(self._wheel_faces(wheel_center, camera, canvas_size))

        # Low armored chassis, dark side steps, and distinct front/rear bumpers.
        faces.extend(
            self._cuboid_faces((0.0, 0.0, -0.04), (1.86, 0.82, 0.32), orange, camera, canvas_size)
        )
        for side in (-1.0, 1.0):
            faces.extend(
                self._cuboid_faces(
                    (0.0, 0.47 * side, -0.10),
                    (1.25, 0.12, 0.14),
                    dark_trim,
                    camera,
                    canvas_size,
                )
            )
        faces.extend(
            self._cuboid_faces((0.98, 0.0, -0.08), (0.18, 0.88, 0.16), dark_trim, camera, canvas_size)
        )
        faces.extend(
            self._cuboid_faces((-0.98, 0.0, -0.07), (0.18, 0.91, 0.17), dark_trim, camera, canvas_size)
        )

        # Blocky fender shoulders deliberately leave the wheel faces exposed.
        for x_position in (-0.61, 0.62):
            for side in (-1.0, 1.0):
                faces.extend(
                    self._cuboid_faces(
                        (x_position, 0.46 * side, 0.07),
                        (0.55, 0.22, 0.23),
                        bright_orange,
                        camera,
                        canvas_size,
                    )
                )

        # Sloped hood is separate from the cabin, giving a recognizable short,
        # square-nosed battle-car profile.
        hood_vertices = np.array(
            [
                [0.10, -0.42, -0.01],
                [0.10, 0.42, -0.01],
                [1.02, 0.37, -0.03],
                [1.02, -0.37, -0.03],
                [0.10, -0.40, 0.27],
                [0.10, 0.40, 0.27],
                [0.94, 0.34, 0.13],
                [0.94, -0.34, 0.13],
            ],
            dtype=np.float32,
        )
        hood_faces = (
            ((0, 1, 2, 3), self._scale(orange, 0.46)),
            ((0, 4, 5, 1), self._scale(orange, 0.78)),
            ((3, 2, 6, 7), self._scale(bright_orange, 1.06)),
            ((0, 3, 7, 4), self._scale(orange, 0.88)),
            ((1, 5, 6, 2), orange),
            ((4, 7, 6, 5), bright_orange),
        )
        faces.extend(self._mesh_faces(hood_vertices, hood_faces, camera, canvas_size))

        # Tall compact cabin with a narrow roof and dark wraparound glazing.
        cabin_vertices = np.array(
            [
                [-0.55, -0.38, 0.12],
                [-0.55, 0.38, 0.12],
                [0.34, 0.34, 0.18],
                [0.34, -0.34, 0.18],
                [-0.34, -0.29, 0.66],
                [-0.34, 0.29, 0.66],
                [0.16, 0.27, 0.58],
                [0.16, -0.27, 0.58],
            ],
            dtype=np.float32,
        )
        cabin_faces = (
            ((0, 1, 2, 3), self._scale(orange, 0.55)),
            ((0, 4, 5, 1), self._scale(glass, 0.72)),
            ((3, 2, 6, 7), self._scale(glass, 1.45)),
            ((0, 3, 7, 4), self._scale(glass, 1.02)),
            ((1, 5, 6, 2), self._scale(glass, 1.22)),
            ((4, 7, 6, 5), bright_orange),
        )
        faces.extend(self._mesh_faces(cabin_vertices, cabin_faces, camera, canvas_size))

        # Two posts and a broad rear wing complete the high, compact silhouette.
        for side in (-1.0, 1.0):
            faces.extend(
                self._cuboid_faces(
                    (-0.74, 0.27 * side, 0.39),
                    (0.11, 0.09, 0.43),
                    dark_trim,
                    camera,
                    canvas_size,
                )
            )
        faces.extend(
            self._cuboid_faces(
                (-0.79, 0.0, 0.61),
                (0.28, 1.27, 0.12),
                bright_orange,
                camera,
                canvas_size,
            )
        )

        for _, points, color in sorted(faces, key=lambda item: item[0], reverse=True):
            pygame.draw.polygon(canvas, color, points)
            pygame.draw.aalines(
                canvas, self._mix(color, (255, 215, 164), 0.24), True, points
            )
        self._draw_rear_lights(canvas, pygame, camera)

    def draw_vehicle(self, canvas: Any, pygame: Any, camera: Camera) -> None:
        """Draw the shared battle-car model and its current boost plume.

        The method is public so the legacy wireframe scene can use the same
        vehicle without adopting the enhanced tunnel or camera.
        """

        boost_glow = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        self._draw_boost(canvas, pygame, camera, boost_glow)
        canvas.blit(boost_glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        self._draw_car(canvas, pygame, camera)

    def _draw_vignette(self, canvas: Any, pygame: Any) -> None:
        width, height = canvas.get_size()
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        for inset in range(0, 42, 6):
            alpha = max(5, 25 - inset // 3)
            pygame.draw.rect(
                overlay,
                (0, 3, 10, alpha),
                (inset, inset, width - 2 * inset, height - 2 * inset),
                width=8,
            )
        canvas.blit(overlay, (0, 0))

    def draw_scene(self, canvas: Any, pygame: Any) -> None:
        """Draw the enhanced three-dimensional view onto ``canvas``."""

        self._draw_background(canvas, pygame)
        camera = self._camera()
        progresses, angles, cross_sections = self._tube_samples()
        distance_to_goal = float(self.env.tube_length - self.env.progress)
        self._draw_goal_backdrop(canvas, pygame, camera, distance_to_goal)
        self._draw_tube_mesh(
            canvas, pygame, camera, progresses, angles, cross_sections
        )

        tunnel_glow = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        self._draw_tube_structure(
            canvas, pygame, camera, progresses, cross_sections, tunnel_glow
        )
        self._draw_chevrons(canvas, pygame, camera, tunnel_glow)
        self._draw_goal_portal(canvas, pygame, camera, tunnel_glow)
        canvas.blit(tunnel_glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        self.draw_vehicle(canvas, pygame, camera)
        self._draw_vignette(canvas, pygame)

    def _glass_panel(
        self,
        pygame: Any,
        size: tuple[int, int],
        *,
        radius: int = 12,
    ) -> Any:
        panel = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (4, 13, 25, 202), panel.get_rect(), border_radius=radius)
        pygame.draw.rect(
            panel,
            (83, 180, 203, 112),
            panel.get_rect(),
            width=1,
            border_radius=radius,
        )
        pygame.draw.line(panel, (112, 220, 233, 42), (12, 2), (size[0] - 12, 2))
        return panel

    def draw_minimap(self, canvas: Any, pygame: Any) -> None:
        """Draw a compact plan-view map without changing simulation state."""

        panel_width, panel_height = 258, 142
        left = canvas.get_width() - panel_width - 18
        top = 18
        panel = self._glass_panel(pygame, (panel_width, panel_height))
        panel.blit(
            self._font(pygame, 16, bold=True).render(
                "COURSE VECTOR", True, (156, 218, 229)
            ),
            (12, 9),
        )

        samples = np.stack(
            [
                self.env._centerline(float(progress))
                for progress in np.linspace(0.0, float(self.env.tube_length), 100)
            ]
        )
        y_min = float(np.min(samples[:, 1]))
        y_max = float(np.max(samples[:, 1]))
        y_pad = max(1.0, 0.15 * (y_max - y_min))
        y_min -= y_pad
        y_max += y_pad
        map_left, map_right = 12, panel_width - 12
        map_top, map_bottom = 34, panel_height - 24

        def map_point(point: np.ndarray) -> tuple[int, int]:
            x_fraction = float(
                np.clip(
                    (point[0] - float(self.env.start_x)) / float(self.env.tube_length),
                    0.0,
                    1.0,
                )
            )
            y_fraction = (float(point[1]) - y_min) / max(y_max - y_min, 1e-6)
            return (
                int(map_left + x_fraction * (map_right - map_left)),
                int(map_bottom - y_fraction * (map_bottom - map_top)),
            )

        course_points = [map_point(point) for point in samples]
        pygame.draw.lines(panel, (28, 116, 141, 75), False, course_points, width=7)
        pygame.draw.aalines(panel, (78, 207, 222, 230), False, course_points)

        progress_index = int(
            np.clip(
                round((len(course_points) - 1) * self.env.progress / self.env.tube_length),
                0,
                len(course_points) - 1,
            )
        )
        if progress_index > 0:
            pygame.draw.lines(
                panel,
                (85, 255, 190, 220),
                False,
                course_points[: progress_index + 1],
                width=2,
            )
        car = map_point(np.asarray(self.env.position))
        pygame.draw.circle(panel, (255, 118, 66, 70), car, 8)
        pygame.draw.circle(panel, (255, 172, 83, 255), car, 4)

        progress_text = self._font(pygame, 15, bold=True).render(
            f"{100.0 * self.env.progress / self.env.tube_length:05.1f}%",
            True,
            (202, 245, 244),
        )
        panel.blit(progress_text, (panel_width - progress_text.get_width() - 12, 8))
        canvas.blit(panel, (left, top))

    def draw_playback_controls(self, canvas: Any, pygame: Any) -> None:
        """Draw clickable playback controls in the free top-center area."""

        play_rect, pause_rect, restart_rect = playback_button_rects(
            pygame, canvas.get_size()
        )
        paused = bool(getattr(self.env, "paused", False))
        controls = (
            ("PLAY", play_rect, not paused, (66, 231, 183)),
            ("PAUSE", pause_rect, paused, (75, 203, 246)),
            ("RESTART", restart_rect, False, (255, 155, 72)),
        )
        label_font = self._font(pygame, 16, bold=True)
        for label, rect, active, accent in controls:
            shadow = rect.move(0, 3)
            pygame.draw.rect(canvas, (2, 7, 14), shadow, border_radius=9)
            background = self._mix((7, 20, 32), accent, 0.31 if active else 0.06)
            border = accent if active else self._mix((58, 111, 126), accent, 0.28)
            pygame.draw.rect(canvas, background, rect, border_radius=9)
            pygame.draw.rect(canvas, border, rect, width=2 if active else 1, border_radius=9)

            # A small state lamp remains legible when the labels are viewed at
            # reduced playback resolution.
            lamp_center = (rect.left + 14, rect.centery)
            pygame.draw.circle(canvas, self._scale(accent, 0.35), lamp_center, 5)
            pygame.draw.circle(canvas, accent, lamp_center, 3 if active else 2)
            rendered = label_font.render(
                label,
                True,
                (236, 251, 249) if active else (163, 196, 204),
            )
            canvas.blit(
                rendered,
                (
                    rect.centerx - rendered.get_width() // 2 + 6,
                    rect.centery - rendered.get_height() // 2,
                ),
            )

    def draw_hud(self, canvas: Any, pygame: Any) -> None:
        """Draw speed, progress, clearance, and boost instrumentation."""

        width, height = canvas.get_size()
        speed = float(np.linalg.norm(self.env.velocity))
        progress_fraction = float(
            np.clip(self.env.progress / self.env.tube_length, 0.0, 1.0)
        )
        usable_radius = max(
            float(self.env.tube_radius - self.env.car_radius), 1e-6
        )
        clearance = max(0.0, usable_radius - float(self.env.radial_distance))
        clearance_fraction = float(np.clip(clearance / usable_radius, 0.0, 1.0))
        danger = 1.0 - float(np.clip(clearance / 1.5, 0.0, 1.0))
        clearance_color = self._mix((76, 237, 184), (255, 82, 69), danger)

        speed_panel = self._glass_panel(pygame, (192, 82))
        speed_panel.blit(
            self._font(pygame, 15, bold=True).render(
                "AIRSPEED", True, (129, 193, 208)
            ),
            (13, 9),
        )
        speed_panel.blit(
            self._font(pygame, 43, bold=True).render(
                f"{speed:04.1f}", True, (235, 250, 249)
            ),
            (11, 25),
        )
        speed_panel.blit(
            self._font(pygame, 15).render("m/s", True, (108, 179, 196)),
            (145, 49),
        )
        canvas.blit(speed_panel, (18, 18))

        progress_left, progress_top, progress_width = 238, 24, 390
        pygame.draw.rect(
            canvas,
            (7, 20, 32),
            (progress_left, progress_top, progress_width, 13),
            border_radius=6,
        )
        fill_width = int(progress_width * progress_fraction)
        if fill_width > 0:
            pygame.draw.rect(
                canvas,
                (75, 225, 202),
                (progress_left, progress_top, fill_width, 13),
                border_radius=6,
            )
        pygame.draw.rect(
            canvas,
            (108, 196, 207),
            (progress_left, progress_top, progress_width, 13),
            width=1,
            border_radius=6,
        )
        canvas.blit(
            self._font(pygame, 15, bold=True).render(
                "TUBE PROGRESS", True, (150, 211, 219)
            ),
            (progress_left, progress_top + 18),
        )
        progress_label = self._font(pygame, 15, bold=True).render(
            f"{100.0 * progress_fraction:05.1f}%", True, (209, 244, 242)
        )
        canvas.blit(
            progress_label,
            (progress_left + progress_width - progress_label.get_width(), progress_top + 18),
        )

        clearance_panel = self._glass_panel(pygame, (242, 62))
        clearance_panel.blit(
            self._font(pygame, 15, bold=True).render(
                "WALL CLEARANCE", True, (138, 198, 210)
            ),
            (12, 8),
        )
        clearance_panel.blit(
            self._font(pygame, 20, bold=True).render(
                f"{clearance:03.1f} m", True, clearance_color
            ),
            (164, 7),
        )
        bar_rect = pygame.Rect(12, 36, 218, 10)
        pygame.draw.rect(clearance_panel, (26, 42, 50, 230), bar_rect, border_radius=5)
        if clearance_fraction > 0.0:
            pygame.draw.rect(
                clearance_panel,
                (*clearance_color, 255),
                (bar_rect.left, bar_rect.top, int(bar_rect.width * clearance_fraction), bar_rect.height),
                border_radius=5,
            )
        canvas.blit(clearance_panel, (18, height - 80))

        boost_panel = self._glass_panel(pygame, (154, 55))
        boost_panel.blit(
            self._font(pygame, 15, bold=True).render(
                "BOOST", True, (126, 195, 213)
            ),
            (12, 8),
        )
        boost_text = "INF" if self.env.unlimited_boost else f"{self.env.boost:03.0f}"
        rendered_boost = self._font(pygame, 25, bold=True).render(
            boost_text, True, (77, 203, 255)
        )
        boost_panel.blit(rendered_boost, (154 - rendered_boost.get_width() - 12, 21))
        canvas.blit(boost_panel, (width - 172, height - 73))

        if bool(getattr(self.env, "reached_goal", False)) or bool(
            getattr(self.env, "crashed", False)
        ):
            goal = bool(getattr(self.env, "reached_goal", False))
            message = "TUBE COMPLETE" if goal else "WALL IMPACT"
            color = (108, 255, 187) if goal else (255, 92, 75)
            text = self._font(pygame, 42, bold=True).render(message, True, color)
            backing = pygame.Surface((text.get_width() + 38, text.get_height() + 20), pygame.SRCALPHA)
            pygame.draw.rect(backing, (3, 10, 20, 210), backing.get_rect(), border_radius=12)
            pygame.draw.rect(backing, (*color, 150), backing.get_rect(), width=2, border_radius=12)
            backing.blit(text, (19, 10))
            canvas.blit(
                backing,
                ((width - backing.get_width()) // 2, (height - backing.get_height()) // 2),
            )
