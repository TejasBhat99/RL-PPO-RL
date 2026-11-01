# RLRL

RLRL is a lightweight Gymnasium and PyTorch project for training a
Rocket League-inspired aerial car to navigate procedural 3D tubes.

The project includes:

- `TubeCar-v0`, an airborne tube-navigation environment;
- a native PyTorch actor–critic and PPO implementation;
- a geometric expert, behavior cloning, and DAgger training;
- enhanced and legacy Pygame renderers;
- a trained tube policy and its evaluation results.

This is an educational simulator with simplified physics. It is not a Rocket
League clone and is not affiliated with Psyonix or Epic Games.

## Results

The included policy was evaluated on 500 held-out procedural tubes using seeds
`800000` through `800499`.

| Policy | Success | Mean progress | Radial error | Control change |
| --- | ---: | ---: | ---: | ---: |
| PPO + expert + DAgger | **93.4%** | **98.8%** | 1.901 m | **0.062** |
| Original PPO baseline | 83.2% | 92.1% | **1.469 m** | 0.080 |

Additional evaluations:

| Course distribution | Episodes | Success |
| --- | ---: | ---: |
| Fixed curved tube | 200 | 100.0% |
| Robust procedural tubes | 300 | 94.0% |
| Extreme hard-tail tubes | 300 | 92.3% |

The selected policy is stored at:

```text
artifacts/torch_ppo/tube_v4_dagger_pass2/best_model.pt
```

Its SHA-256 checksum is:

```text
ca67c07103b3c2dc297fe62b4592f6ad8a60e8aba7efdf75621abe04e46763c6
```

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate rlrl
python -m pytest
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,train]"
python -m pytest
```

## Watch the trained policy

```bash
python scripts/evaluate_tube_ppo.py \
  artifacts/torch_ppo/tube_v4_dagger_pass2/best_model.pt \
  --stage procedural_tube \
  --episodes 1 \
  --seed 800000 \
  --render \
  --playback-fps 3
```

The enhanced renderer is used by default. Use `--graphics legacy` for the
original wireframe view.

Override the selected stage's geometry to try a longer, wider, or more curved
tube:

```bash
python scripts/evaluate_tube_ppo.py \
  artifacts/torch_ppo/tube_v4_dagger_pass2/best_model.pt \
  --stage procedural_tube \
  --seed 800123 \
  --tube-length 150 \
  --tube-radius 4.5 \
  --curvature 0.6 \
  --episodes 1 \
  --render \
  --playback-fps 5
```

Tube length must be greater than `5`, radius greater than `1`, and curvature
between `0` and `1`. The selected stage's episode limit is unchanged, so
extreme tube lengths may time out. Change `--seed` to generate another
procedural tube.

The render window includes clickable **PLAY**, **PAUSE**, and **RESTART**
controls. Rendering is separate from the environment state and does not affect
training, rewards, physics, or policy observations.

## Manual controls

```bash
python examples/play_tube_keyboard.py
```

| Input | Action |
| --- | --- |
| Up / Down | Nose down / nose up |
| Left / Right | Yaw left / yaw right |
| Q / E | Air roll left / air roll right |
| Space | Jump |
| Shift | Boost |
| R | Restart |
| Esc | Quit |

Controller mappings:

| Input | Action |
| --- | --- |
| Left stick | Yaw and pitch |
| PlayStation Cross | Jump |
| L1 / R1 | Air roll left / right |
| R2 | Boost |

## Environment usage

```python
import gymnasium as gym
import rlrl_env  # registers TubeCar-v0

env = gym.make(
    "TubeCar-v0",
    tube_length=100.0,
    tube_radius=4.0,
    curvature=0.5,
    procedural_course=True,
)

observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
```

### Tube action space

```text
[pitch, yaw, roll, jump, boost]
```

All actions are in `[-1, 1]`. Jump and boost activate above `0.5`. The tube
task is airborne-only and currently uses unlimited boost.

### Tube observation

The 47-value normalized observation contains:

- body-local velocity;
- orientation and angular velocity;
- radial offset and local tube tangent;
- three future centerline offsets and tangents;
- outward and tangential speed;
- tube progress, smoothed controls, jump latch, and previous boost state.

### Tube reward

Let `ρ` be radial error divided by usable tube radius:

```text
+0.20 × forward progress
-0.01  per step
-0.02 × ρ²
-0.10 × max(0, (ρ - 0.75) / 0.25)²
-0.01 × control-change²
+100   goal
-100   wall collision
-20    timeout
```

The tube is defined analytically by a 3D centerline and circular cross-section,
keeping collision checks fast and procedural courses deterministic.

## Neural network architecture

The policy uses a native PyTorch actor–critic with separate actor and critic
MLPs. The bundled checkpoint uses this architecture:

```text
47-value observation
├── Actor:  Linear(47, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 5)
│   ├── 3 Normal means: pitch, yaw, roll
│   └── 2 Bernoulli logits: jump, boost
└── Critic: Linear(47, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 1)
    └── scalar state-value estimate V(s)
```

The continuous controls use three learned, state-independent log standard
deviations, one for each control. During deterministic evaluation, the Normal
means are clipped to `[-1, 1]`, while jump and boost activate when their
Bernoulli probabilities reach `0.5`. The actor and critic do not share
parameters. Linear layers use orthogonal initialization, and the complete
model has 157,705 trainable parameters.

## Training

The final policy was produced in four stages:

1. curriculum PPO on increasingly difficult tubes;
2. successful trajectories from a geometric expert;
3. behavior cloning from expert actions;
4. DAgger training on learner and recovery states.

Train the native PyTorch PPO curriculum:

```bash
python scripts/train_tube_ppo.py \
  --timesteps 1500000 \
  --run-dir artifacts/torch_ppo/tube-ppo
```

Pretrain from the geometric expert:

```bash
python scripts/pretrain_tube_expert.py \
  --resume <ppo-checkpoint> \
  --output artifacts/torch_ppo/tube-expert/model.pt
```

Run DAgger:

```bash
python scripts/dagger_tube_expert.py \
  --resume artifacts/torch_ppo/tube-expert/model.pt \
  --run-dir artifacts/torch_ppo/tube-dagger
```

TensorBoard logs are written inside the selected run directory.

## Testing and reproducibility

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -m pytest
sha256sum --check artifacts/SHA256SUMS
python -m pip check
```

- Procedural courses are seeded.
- Evaluation is deterministic unless `--stochastic` is passed.
- The bundled checkpoint is loaded with `weights_only=True`.
- The tests cover environment behavior, rewards, controls, rendering
  determinism, camera stability, playback controls, and PPO components.
- GitHub Actions runs tests, builds the package, verifies the model checksum,
  and smoke-tests the bundled policy.

## Repository layout

```text
src/rlrl_env/
├── envs/             # Tube physics and TubeCarEnv
├── agents/           # Actor-critic, rollout buffer, and tube expert
├── rendering/        # Enhanced tube renderer
└── tube_curriculum.py

scripts/              # Training, DAgger, and evaluation commands
examples/             # Manual-control and random-agent demos
tests/                # Environment, renderer, and PPO tests
artifacts/            # Selected policy and evaluation metrics
```

## Current limitations

- Physics are intentionally simplified.
- The agent observes structured state rather than pixels.
- Tube generation uses a bounded sinusoidal course family.
- Tube boost is unlimited.
- There is no ball, opponent, or ground-driving task.
- The Pygame renderer is designed for visualization, not photorealism.
