from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rlrl_env.agents.ppo import ActorCritic, RolloutBuffer, load_model
from rlrl_env.tube_curriculum import TUBE_STAGES, tube_stage_with_overrides


def test_actor_critic_shapes_and_finite_outputs() -> None:
    model = ActorCritic(observation_size=47, action_size=5, hidden_sizes=(32, 32))
    observations = torch.zeros((4, 47))
    actions, log_probabilities, entropy, values = model.action_and_value(observations)

    assert actions.shape == (4, 5)
    assert log_probabilities.shape == (4,)
    assert entropy.shape == (4,)
    assert values.shape == (4,)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probabilities).all()
    assert torch.isfinite(values).all()
    assert torch.logical_or(actions[:, 3:] == 0.0, actions[:, 3:] == 1.0).all()


def test_deterministic_binary_actions_are_explicit_switches() -> None:
    model = ActorCritic(observation_size=47, action_size=5, hidden_sizes=(32, 32))
    actions = model.predict(torch.zeros((3, 47)), deterministic=True)
    assert torch.logical_or(actions[:, 3:] == 0.0, actions[:, 3:] == 1.0).all()


def test_rollout_buffer_computes_expected_returns_without_bootstrap() -> None:
    buffer = RolloutBuffer(
        n_steps=2,
        n_envs=1,
        observation_size=1,
        action_size=1,
        device=torch.device("cpu"),
    )
    for reward, done in ((1.0, 0.0), (1.0, 1.0)):
        buffer.add(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.zeros(1),
            torch.tensor([reward]),
            torch.tensor([done]),
            torch.zeros(1),
        )
    buffer.compute_returns_and_advantages(
        next_values=torch.tensor([100.0]), gamma=1.0, gae_lambda=1.0
    )

    torch.testing.assert_close(buffer.returns[:, 0], torch.tensor([2.0, 1.0]))


def test_tube_curriculum_progresses_to_procedural_courses() -> None:
    assert TUBE_STAGES["short_straight"].curvature == 0.0
    assert (
        TUBE_STAGES["long_straight"].tube_length
        > TUBE_STAGES["short_straight"].tube_length
    )
    assert TUBE_STAGES["curved_tube"].curvature > 0.0
    assert TUBE_STAGES["procedural_tube"].procedural_course
    assert TUBE_STAGES["procedural_hard_tail"].procedural_course


def test_tube_stage_geometry_overrides_preserve_other_settings() -> None:
    original = TUBE_STAGES["procedural_tube"]
    assert tube_stage_with_overrides(original) is original

    customized = tube_stage_with_overrides(
        original,
        tube_length=150.0,
        tube_radius=4.5,
        curvature=0.6,
    )

    assert customized.tube_length == 150.0
    assert customized.tube_radius == 4.5
    assert customized.curvature == 0.6
    assert customized.max_episode_steps == original.max_episode_steps
    assert customized.procedural_course == original.procedural_course
    assert customized.procedural_scale_range == original.procedural_scale_range
    assert customized.lateral_wavelength_range == original.lateral_wavelength_range
    assert customized.vertical_wavelength_range == original.vertical_wavelength_range


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tube_length": 5.0}, "tube-length"),
        ({"tube_length": float("nan")}, "tube-length"),
        ({"tube_radius": 1.0}, "tube-radius"),
        ({"tube_radius": float("inf")}, "tube-radius"),
        ({"curvature": -0.1}, "curvature"),
        ({"curvature": 1.1}, "curvature"),
    ],
)
def test_tube_stage_rejects_invalid_geometry_overrides(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tube_stage_with_overrides(TUBE_STAGES["procedural_tube"], **overrides)


def test_bundled_tube_checkpoint_loads_with_current_observation_size() -> None:
    checkpoint_path = (
        Path(__file__).parents[1]
        / "artifacts"
        / "torch_ppo"
        / "tube_v4_dagger_pass2"
        / "best_model.pt"
    )
    model, checkpoint = load_model(
        checkpoint_path, torch.device("cpu"), observation_size_override=47
    )
    actions = model.predict(torch.zeros((2, 47)), deterministic=True)
    assert actions.shape == (2, 5)
    assert checkpoint["action_size"] == 5
