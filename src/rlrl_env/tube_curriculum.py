"""Curriculum stages for airborne tube navigation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class TubeStage:
    name: str
    tube_length: float
    tube_radius: float
    curvature: float
    max_episode_steps: int
    procedural_course: bool = False
    procedural_scale_range: tuple[float, float] = (0.55, 1.0)
    lateral_wavelength_range: tuple[float, float] = (42.0, 70.0)
    vertical_wavelength_range: tuple[float, float] = (55.0, 85.0)


TUBE_STAGES = {
    "short_straight": TubeStage("short_straight", 30.0, 6.0, 0.0, 300),
    "long_straight": TubeStage("long_straight", 60.0, 5.0, 0.0, 500),
    "gentle_curve": TubeStage("gentle_curve", 80.0, 5.0, 0.20, 750),
    "curved_tube": TubeStage("curved_tube", 100.0, 4.5, 0.35, 1_000),
    "procedural_gentle": TubeStage(
        "procedural_gentle", 80.0, 5.0, 0.20, 800, procedural_course=True
    ),
    "procedural_curved": TubeStage(
        "procedural_curved", 100.0, 4.5, 0.35, 1_000, procedural_course=True
    ),
    "procedural_hard": TubeStage(
        "procedural_hard", 100.0, 4.25, 0.42, 1_100, procedural_course=True
    ),
    "procedural_tube": TubeStage(
        "procedural_tube", 100.0, 4.0, 0.50, 1_200, procedural_course=True
    ),
    "procedural_challenging": TubeStage(
        "procedural_challenging",
        100.0,
        4.0,
        0.50,
        1_200,
        procedural_course=True,
        procedural_scale_range=(0.75, 1.0),
        lateral_wavelength_range=(46.0, 65.0),
        vertical_wavelength_range=(58.0, 78.0),
    ),
    "procedural_robust": TubeStage(
        "procedural_robust",
        100.0,
        4.0,
        0.50,
        1_200,
        procedural_course=True,
        procedural_scale_range=(0.65, 1.0),
        lateral_wavelength_range=(44.0, 68.0),
        vertical_wavelength_range=(57.0, 82.0),
    ),
    "procedural_hard_tail": TubeStage(
        "procedural_hard_tail",
        100.0,
        4.0,
        0.50,
        1_200,
        procedural_course=True,
        procedural_scale_range=(0.85, 1.0),
        lateral_wavelength_range=(42.0, 58.0),
        vertical_wavelength_range=(55.0, 72.0),
    ),
}


def tube_env_kwargs(stage: TubeStage) -> dict[str, Any]:
    return {
        "tube_length": stage.tube_length,
        "tube_radius": stage.tube_radius,
        "curvature": stage.curvature,
        "max_episode_steps": stage.max_episode_steps,
        "procedural_course": stage.procedural_course,
        "procedural_scale_range": stage.procedural_scale_range,
        "lateral_wavelength_range": stage.lateral_wavelength_range,
        "vertical_wavelength_range": stage.vertical_wavelength_range,
    }


def tube_stage_with_overrides(
    stage: TubeStage,
    *,
    tube_length: float | None = None,
    tube_radius: float | None = None,
    curvature: float | None = None,
) -> TubeStage:
    """Return a stage with optional evaluation-only geometry overrides."""

    overrides: dict[str, float] = {}
    if tube_length is not None:
        if not isfinite(tube_length) or tube_length <= 5.0:
            raise ValueError("tube-length must be finite and greater than 5")
        overrides["tube_length"] = float(tube_length)
    if tube_radius is not None:
        if not isfinite(tube_radius) or tube_radius <= 1.0:
            raise ValueError("tube-radius must be finite and greater than 1")
        overrides["tube_radius"] = float(tube_radius)
    if curvature is not None:
        if not isfinite(curvature) or not 0.0 <= curvature <= 1.0:
            raise ValueError("curvature must be finite and in [0, 1]")
        overrides["curvature"] = float(curvature)
    return replace(stage, **overrides) if overrides else stage
