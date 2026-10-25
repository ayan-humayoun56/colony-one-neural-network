# Colony One: Neural Network Experiments

A compact CNN-based policy-value model for the Colony: One strategy game, explored through supervised learning, self-play experiments, Monte Carlo Tree Search (MCTS), and ONNX deployment.

This project documents an experimental machine-learning pipeline built around the Colony: One tournament environment. The goal was to explore how a small neural network can provide policy logits and position-value estimates to a search-based game agent.

> **Research status:** Experimental portfolio project. The trained model met the ONNX submission interface requirements, but the experiments did not demonstrate an improvement over the original model against the GreedyAgent baseline.

## Overview

The project explores four stages:

1. **Model design** — a compact convolutional neural network with policy and value heads.
2. **Data generation** — self-play experiments and positions generated from GreedyAgent games.
3. **Training** — supervised policy learning and value regression using PyTorch.
4. **Deployment and evaluation** — ONNX export and evaluation with MCTS against GreedyAgent.

## Model architecture

The `TinyColonyNet` model accepts a batch of encoded board observations and returns action scores and a value estimate.

| Component | Design |
|---|---|
| Input | Float32 tensor shaped `(batch, 16, 8, 8)` |
| Feature extractor | Two 3×3 convolution layers with 16 channels and ReLU activations |
| Policy head | 1×1 convolution, flattening, and a linear layer |
| Policy output | 4096 action logits per position |
| Value head | Global average pooling followed by fully connected layers |
| Value output | Scalar per position, constrained to `[-1, 1]` using `tanh` |

The policy head produces logits, not probabilities. During training, the logits are passed through `log_softmax` to calculate the policy loss. The value head estimates the position value from the current player's perspective.

## Learning experiments

### 1. Random-output self-play data

An initial dataset was generated using a random evaluator. This was useful for exercising the data pipeline, but it should not be treated as strategically informative training data.

### 2. GreedyAgent imitation and value learning

A dataset was generated from GreedyAgent-versus-GreedyAgent games. Each position records the observation, a one-hot policy target for the action selected by the agent, and a terminal outcome value.

The training script combines:

- Policy cross-entropy against the target action distribution.
- Mean squared error (MSE) between predicted and target values.
- Adam optimization.
- A fixed random seed for reproducibility.
- A held-out validation split and early stopping.

### 3. Guided self-play

A small guided self-play experiment was also conducted. Its dataset contained only 76 positions, so it is preliminary evidence that the pipeline ran—not evidence of strong self-play learning.

## Evaluation

The model was evaluated against GreedyAgent using MCTS with 60 simulations per move and a fixed seed configuration.

| Model | Wins in 20 games |
|---|---:|
| Original model | 7/20 (35%) |
| Greedy-data-trained model | 6/20 (30%) |

The observed difference is small and does not establish that either model is stronger. Twenty games are insufficient for a confident comparison, and results may depend on seeds, sides, game dynamics, and training data.

The original ONNX model also passed the provided submission validator. Validation confirms interface compatibility and execution under the validator; it does not establish strategic strength.

## Repository structure

```text
.
├── colony_one/              # Colony: One environment and search components
├── docs/                    # Environment documentation
├── play/                    # Model play utilities
├── tools/                   # Validation utilities
├── tiny_colony_net.py       # CNN policy-value architecture
├── train_model.py           # Initial self-play training
├── train_greedy_data.py     # GreedyAgent dataset generation
├── train_greedy_model.py    # Training on GreedyAgent data
├── train_guided_selfplay.py # Guided self-play experiment
├── evaluate_greedy.py       # MCTS evaluation against GreedyAgent
├── export_onnx.py           # Export original model to ONNX
├── export_greedy_onnx.py    # Export trained model to ONNX
├── model.onnx               # Original exported model
├── model_greedy.onnx        # Greedy-data-trained model
└── greedy_data.npz          # GreedyAgent-generated dataset
```

The environment, rules, and supporting files are included from the underlying Colony: One project. Consult the included `LICENSE` before using or redistributing the code. The experimental scripts and model architecture are the focus of this portfolio; the entire upstream framework should not be assumed to be original work.

## Reproducibility

The project uses Python and PyTorch for training, NumPy for dataset handling, and ONNX Runtime for inference through the environment's evaluation utilities.

Use a compatible Python environment and install the dependencies specified by the project. The commands below assume the environment's dependencies have been installed and the corresponding dataset/checkpoint files are available.

Run a model shape check:

```bash
python tiny_colony_net.py
```

Train on GreedyAgent-generated data:

```bash
python train_greedy_model.py
```

Export the trained checkpoint:

```bash
python export_greedy_onnx.py
```

Evaluate an ONNX model against GreedyAgent:

```bash
python evaluate_greedy.py --model model_greedy.onnx --games 20 --simulations 60 --seed 42
```

Validate an ONNX export:

```bash
python tools/validate_submission.py model_greedy.onnx
```

These commands depend on the underlying environment and the expected files being present. Training and evaluation results may vary with software versions, seeds, and runtime conditions.

## Limitations

- The evaluation sample is small and does not establish statistically reliable strength differences.
- GreedyAgent-generated data teaches the model to imitate the baseline's actions; it does not provide an independent source of stronger strategy.
- The initial random-evaluator dataset is not strategically informative.
- The guided self-play dataset is too small to support conclusions about learning quality.
- Passing the ONNX validator demonstrates interface compliance, not playing strength.
- The project does not establish superiority over GreedyAgent or demonstrate state-of-the-art performance.

## Future work

- Run larger evaluations across multiple seeds and both player sides.
- Record complete training curves and dataset statistics.
- Compare against random and greedy baselines under a consistent evaluation protocol.
- Investigate stronger target policies and more informative self-play.
- Add automated tests for tensor shapes, export compatibility, and evaluation reproducibility.

## Documentation

- [Technical White Paper](research/technical-white-paper.md) — Detailed
  documentation of the model architecture, training experiments, evaluation,
  results, limitations, and lessons learned.

## Attribution and license

This project builds on the Colony: One tournament environment. The included `LICENSE` identifies the original copyright holder and specifies conditions on non-commercial distribution, commercial use, event use, and derivative works.

Preserve the original license and attribution. Do not assume that publishing this repository grants permission for uses prohibited by the license. Seek written clarification from the copyright holder if needed.

---

**Project focus:** neural network engineering, supervised learning, model export, reproducible evaluation, and documenting negative results honestly.
