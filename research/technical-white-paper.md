# Engineering a Compact Policy–Value Network for Colony: One: An Experimental Case Study

**Author:** [Author name — to be completed] · GitHub: `ayan-humayoun56`
**Document status:** Draft technical white paper, 25 October 2026. The structure and style are IEEE-inspired only. This document is not peer reviewed, has not been submitted to or published by IEEE, and makes no claim of acceptance.
**Code and artifacts:** https://github.com/ayan-humayoun56/colony-one-neural-network

---

## Abstract

This report documents a short engineering experiment on Colony: One, a two-team strategy game whose tournament environment evaluates a submitted neural network inside a fixed Monte Carlo tree search. A compact convolutional policy–value network was implemented in PyTorch, trained on three small datasets, exported to the Open Neural Network Exchange format with a dynamic batch axis, and compared against the environment's greedy baseline agent. The datasets contained 87 positions from random-evaluator self-play, 2,710 positions recorded from greedy-agent games with one-hot action targets and terminal outcome values, and 76 positions from a two-game guided self-play run. The original exported model was accepted by the environment's submission validator, which establishes interface and resource compatibility but not playing strength. In a preliminary comparison of 20 games against the greedy baseline with 60 search simulations per move, the original model won 7 games and the model trained on greedy-agent data won 6. This difference is too small to support any ranking, and the experiment did not demonstrate that training on greedy-agent data improved play. Methodological limitations include random position-level validation splits, weak teachers, unverified environment seeding, and unrecorded software versions. The main contribution is a documented end-to-end workflow with a neutral-to-negative result, together with concrete evaluation and reproducibility improvements for future work.

**Index Terms**—Monte Carlo tree search, policy–value network, imitation learning, Open Neural Network Exchange (ONNX), reproducibility.

---

## I. Introduction

### A. Context

Colony: One is a two-team strategy game played on an 8×8 board. It is distributed as a tournament environment associated with the 2026 Cloudflight Coding Contest (CCC) [1], [2]. A participant does not write a move-selection program. Instead, the participant submits one neural-network file, and a referee-controlled Monte Carlo tree search (MCTS) uses the network to choose every action [2]. The network is therefore a position evaluator. It returns policy logits, which act as a prior over 4,096 action indices, and a scalar value estimate for the side to move [2].

MCTS grows a search tree from repeated simulated continuations [3]–[5]. Pairing such a search with learned policy and value networks became prominent through AlphaGo and its successors [6]–[8]. In the Colony: One tournament, the model file is exchanged in the Open Neural Network Exchange (ONNX) format [9] and executed with ONNX Runtime [10].

This report documents a short development sprint around that environment. The contest submission period closed on Friday, 23 October 2026. The author completed the first two contest levels but did not reach a competitive Level 3 solution. After the deadline, the objective changed from contest-specific optimization to learning, reproducibility, documentation, and a public portfolio project. The report is therefore an engineering case study and reproducibility audit. It does not claim a novel algorithm or a strong agent.

### B. Research Questions

The central question is exploratory:

> Can a compact convolutional policy–value network, trained on small self-play or GreedyAgent-generated datasets and used inside the environment's MCTS workflow, produce a functional ONNX-compatible agent, and what can a small baseline evaluation say about its playing strength?

GreedyAgent is the heuristic baseline agent shipped with the environment [2]. Four secondary questions refine the central one:

- **RQ1.** Can a small convolutional neural network (CNN) implement the required policy/value tensor contract and export to ONNX with a dynamic batch axis?
- **RQ2.** Can a dataset of GreedyAgent-selected actions and terminal outcomes train both network heads end to end?
- **RQ3.** Does the GreedyAgent-trained checkpoint improve on the original checkpoint in a small evaluation against GreedyAgent?
- **RQ4.** What does the sprint reveal about data quality, baseline selection, evaluation sample size, and reproducibility?

All four are treated as exploratory. None of the experiments can establish that the model is stronger than GreedyAgent, that one checkpoint is better than the other, or that the approach is generally effective.

### C. Contributions and Non-Contributions

The contributions are deliberately narrow:

1. A compact CNN policy–value model, implemented in PyTorch [11], that satisfies the environment's tensor contract.
2. Three small data-generation and training pipelines: random-evaluator self-play, GreedyAgent imitation with outcome-value regression, and model-guided self-play generation.
3. ONNX export scripts and a recorded validator result for the original model.
4. A 20-game comparison against GreedyAgent, reported with raw counts and a neutral-to-negative interpretation.
5. An audit of methodological and reproducibility gaps, and the lessons drawn from them.

The report does **not** contribute a new search algorithm, learning algorithm, or architecture. The game engine, rules, MCTS implementation, baseline agent, ONNX runner, and submission validator come from the upstream environment [2] and are not the author's work.

### D. Organization

Sections II–III give background and the interface contract. Sections IV–VII describe the model, data, export, and evaluation protocol. Sections VIII–IX report and discuss results. Sections X–XIII cover lessons, limitations, future work, and conclusions. Appendix A lists commands and artifacts.

---

## II. Background and Related Work

### A. Monte Carlo Tree Search

Coulom introduced a framework that combines tree search with Monte-Carlo evaluation without separating a minimax phase from a Monte-Carlo phase, which gives fine-grained control over how the tree grows [3]. Kocsis and Szepesvári proposed UCT, which applies bandit ideas to guide Monte-Carlo planning [4]. Browne et al. surveyed the first five years of MCTS research [5]. The Colony: One referee runs a published MCTS implementation; this report uses it unchanged [2].

### B. Policy–Value Networks in Game Search

AlphaGo used value networks to evaluate positions and policy networks to select moves, trained by supervised learning from human expert games and by reinforcement learning from self-play, and combined them with a search that blends Monte Carlo simulation with the networks [6]. AlphaGo Zero replaced the separate networks with a single network, trained to predict the program's own move selections and the winners of its games, and used a simpler search without rollouts [7]. AlphaZero applied the same general approach to chess, shogi, and Go [8]. The present project uses the same interface pattern (a network supplying a policy prior and a value to a search) at a very small scale. It reproduces none of those systems.

### C. Imitation Learning and Outcome-Value Regression

A common approach to imitation learning trains a classifier or regressor to predict an expert's actions from observations. Because the learner's own actions change the states it later encounters, sequential imitation violates the independent-and-identically-distributed assumption of standard supervised learning, which can degrade performance in practice [12]. In this project the "expert" is GreedyAgent, a heuristic baseline rather than an optimal player, so the policy labels inherit its limitations. Value targets assign every position the final result of its game; the same supervised outcome-prediction formulation appears in self-play pipelines [7].

### D. ONNX Model Exchange

ONNX is an open format that defines a common operator set and file format for machine-learning models [9]. ONNX Runtime executes such models and checks that they conform to the ONNX specification, but its documentation places responsibility for validating accuracy, performance, and suitability on the user [10]. This distinction between conformance and quality recurs in Sections VI and IX.

---

## III. Environment and Interface Contract

### A. Authorship Boundary

The environment is the work of its upstream author and copyright holder [2]. The table separates upstream components from the author's experimental code.

| Upstream [2] | Author's experimental contribution |
|---|---|
| `colony_one/` package (rules, board, encoding, environment, MCTS, ONNX runner, baselines including GreedyAgent, submission checks) | `tiny_colony_net.py` (model) |
| `docs/`, `play/`, `tools/validate_submission.py` | `train_selfplay.py`, `train_model.py` |
| `LICENSE` and the original README | `train_greedy_data.py`, `train_greedy_model.py`, `train_guided_selfplay.py` |
| | `export_onnx.py`, `export_greedy_onnx.py`, `evaluate_greedy.py` |
| | Datasets, exported models, analysis, and this report |

### B. Tensor Contract

**TABLE I. Model tensor contract [2].**

| Tensor | Direction | Shape | Type | Meaning |
|---|---|---|---|---|
| `observation` | input | (batch, 16, 8, 8) | float32 | Canonical encoded board state |
| `policy` | output | (batch, 4096) | float32 | Raw action logits |
| `value` | output | (batch,) | float32 | Side-to-move value in [−1, 1] |

The batch axis must be dynamic on all three tensors [2].

### C. Observation, Actions, and Turn Semantics

The observation is always oriented to the team to move; for Magenta the board is rotated by 180 degrees. Planes 0–5 hold the mover's pieces, planes 6–11 the opponent's, plane 12 the firewall-locked squares, plane 13 the value `actions_left / 2`, plane 14 a firewall-urgency value, and plane 15 an indicator that Magenta is to move [2]. An action index is `from_square × 64 + to_square` with `square = row × 8 + col` in canonical coordinates [2]. Typically only about 12–30 actions are legal at a position [2]. The game has no draws, and the value target is +1 for a win and −1 for a loss for the side to move [2].

`step()` applies a single action, not a full turn: each team takes two consecutive actions before play passes to the other team [2].

### D. Referee Search Settings

The referee wraps the submitted model in MCTS with 60 simulations per action in Level 3 and 100 in the Level 4 tournament; the local default is 300 [2]. Other published settings are `c_puct` = 1.5, search batch size 8, virtual loss 1.0, root Dirichlet noise with α = 0.3 and ε = 0.15, and root action sampling at temperature 1.0 for actions 1–4 and 0.5 for actions 5–8, then greedy selection [2]. The referee masks illegal actions and takes the softmax over legal logits only, and the upstream documentation advises masking during training as well [2].

### E. Submission Validator

A submission must be a single self-contained file of at most 5 MiB, use the `ai.onnx` opset 13–18 with an allow-listed operator set, and have dynamic batch axes [2]. The local validator reports `ACCEPTED` when the file passes structural and isolated CPU-inference checks and one Level 3 match on the local machine fits the published match limits [2]. The local guard values are 1.8 s CPU time for the per-upload check and 13.5 s for a match, with resident-memory guards of 320 MiB and 425 MiB respectively [2].

---

## IV. Model Architecture

`TinyColonyNet` (`tiny_colony_net.py`) uses a shared two-layer convolutional stem and two heads. Let s ∈ ℝ^(16×8×8) be an observation, φ the rectified linear unit (ReLU), and Conv_k a convolution with kernel size k:

$$h = \varphi\big(\mathrm{Conv}_3(\varphi(\mathrm{Conv}_3(s)))\big) \tag{1}$$

$$z = W_p\,\mathrm{vec}\big(\varphi(\mathrm{Conv}_1(h))\big) + b_p,\quad W_p \in \mathbb{R}^{4096\times 64} \tag{2}$$

$$\hat v = \tanh\!\big(w_2^{\top}\,\varphi(W_1\,\mathrm{GAP}(h) + b_1) + b_2\big),\quad W_1 \in \mathbb{R}^{16\times 16} \tag{3}$$

Here z are the policy logits, GAP is global average pooling over the 8×8 board for each channel, and v̂ is the value estimate. The model returns logits; the softmax is applied only in the training loss and by the referee.

**TABLE II. Layers and shapes.** Parameter counts are derived by hand from the layer definitions in the source; they were not read from a program run.

| Block | Layer | Output shape | Parameters (derived) |
|---|---|---|---|
| Stem | Conv2d(16→16, 3×3, pad 1) + ReLU | (B, 16, 8, 8) | 2,320 |
| Stem | Conv2d(16→16, 3×3, pad 1) + ReLU | (B, 16, 8, 8) | 2,320 |
| Policy head | Conv2d(16→1, 1×1) + ReLU | (B, 1, 8, 8) | 17 |
| Policy head | Flatten | (B, 64) | — |
| Policy head | Linear(64→4096) | (B, 4096) | 266,240 |
| Value head | AdaptiveAvgPool2d(1) + Flatten | (B, 16) | — |
| Value head | Linear(16→16) + ReLU | (B, 16) | 272 |
| Value head | Linear(16→1) + Tanh, `squeeze(−1)` | (B,) | 17 |
| **Total** | | | **271,186** |

The derived count implies about 1.03 MiB of float32 weights, below the 5 MiB limit [2]; the actual file size was not recorded. The script's own shape check prints the parameter count and can confirm the derived figure.

**Design observations (analysis, not tested by ablation).**
- Two stacked 3×3 convolutions give each stem feature a 5×5 receptive field on an 8×8 board.
- About 98.2% of the parameters sit in the final policy layer, which maps only 64 scalar features to 4,096 logits. The policy therefore passes through a single-channel bottleneck.
- The value head discards spatial position through global average pooling.
- These properties may limit capacity, but this report did not test that.

---

## V. Data Generation and Training Methodology

### A. Dataset Overview

**TABLE III. Dataset inventory.**

| File | Positions | Target source | Intended role | Caveat |
|---|---|---|---|---|
| `selfplay_data.npz` | 87 | MCTS root-derived policy targets under a random evaluator; ±1 terminal outcome values | Pipeline test | Not strategically informative; excluded from Git |
| `greedy_data.npz` | 2,710 | One-hot GreedyAgent action; ±1 terminal outcome | Imitation plus outcome-value regression | Heuristic teacher; number of source games unverified |
| `guided_selfplay_data.npz` | 76 (two games) | MCTS guided by the model; normalized policy; 36 values of −1 and 40 of +1 | Generation experiment | Far too small; no training script found |

Position counts and the value balance are taken from the author's records; the arrays were not re-counted for this report.

### B. Random-Evaluator Self-Play Data

`train_selfplay.py` runs MCTS with a `RandomEvaluator` that returns standard-normal logits and zero value estimates, with 20 simulations, search batch size 1, and root noise enabled. The policy target is the search's root-derived distribution, and each saved position receives a terminal value of +1 if its team won and −1 otherwise. The 87 positions are consistent with the script's default of two games, but the invocation was not recorded. The random logits use the global NumPy generator, which the script does not seed, so this dataset cannot be regenerated bit for bit. Because the search priors are random and the value estimates are zero, the data carry almost no strategic signal. Low training loss on it would measure fitting, not learning to play.

### C. GreedyAgent Imitation Data

`train_greedy_data.py` plays GreedyAgent against GreedyAgent, with defaults of 100 games and seed 42. Each team's agent is seeded from a seed-42 generator. At every action the script stores the observation and a one-hot policy target at `action_index(action, team)`. After a game ends, every stored position receives +1 if its team won and −1 otherwise. Games with no winner are skipped. Policy targets are recorded for both teams regardless of outcome, so the moves of losing sides are imitated too. The file contains 2,710 positions. Whether they came from exactly 100 completed games is not established by the available records. This is heuristic imitation plus outcome regression, not expert or optimal play.

### D. Guided Self-Play Data

`train_guided_selfplay.py` is a **data generator**; it contains no training step. It runs MCTS with an ONNX evaluator (default `model.onnx`), defaults of 5 games, 20 simulations, search batch size 1 and root noise enabled, and samples actions at temperature 1.0. The recorded dataset has 76 positions from two games, but the invocation that produced it was not recorded. No `model_guided.onnx` was observed, and no script that trains on this dataset was found. The dataset shows that the generation path ran; it provides no evidence about learning.

### E. Training Objective

Let B be the batch size, z_i ∈ ℝ^4096 the logits for sample i, p_i(a) the target distribution over actions, v̂_i the predicted value, and v_i ∈ {−1, +1} the target value. Define softmax(z)_a = exp(z_a) / Σ_b exp(z_b). The policy loss is cross-entropy against the target distribution:

$$\mathcal{L}_{\text{policy}} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{a=1}^{4096} p_i(a)\,\log \operatorname{softmax}(z_i)_a \tag{4}$$

The value loss is the mean squared error (MSE):

$$\mathcal{L}_{\text{value}} = \frac{1}{B}\sum_{i=1}^{B}\left(\hat v_i - v_i\right)^2 \tag{5}$$

The total loss is their unweighted sum:

$$\mathcal{L} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} \tag{6}$$

For GreedyAgent imitation the target is one-hot at the teacher's action a_i*, so Eq. (4) reduces to

$$\mathcal{L}_{\text{policy}} = -\frac{1}{B}\sum_{i=1}^{B}\log \operatorname{softmax}(z_i)_{a_i^{*}} \tag{7}$$

Reported epoch losses are sample-weighted averages of the batch losses. The softmax in Eq. (4) runs over all 4,096 logits. The referee instead normalizes over legal actions only, and the upstream documentation advises masking illegal actions in training [2]. The objective used here therefore differs from inference-time use.

### F. Training Configuration

Both training scripts use the central processing unit (CPU), set the PyTorch and NumPy seeds to 42, and split by a seed-42 `random_split` with validation size `max(1, int(0.2·N))`. Optimization uses the Adam optimizer [13].

**TABLE IV. Training configuration (from `train_model.py` and `train_greedy_model.py`).**

| Setting | Initial (`train_model.py`) | GreedyAgent (`train_greedy_model.py`) |
|---|---|---|
| Dataset | `selfplay_data.npz` | `greedy_data.npz` |
| Batch size | 16 | 32 (training shuffled; validation not) |
| Learning rate | 0.001 | 0.0003 |
| Epochs | 40, fixed | Up to 60 |
| Early stopping | None | Patience 8 on validation loss |
| Saved checkpoint | Final-epoch weights | Best-validation weights |
| Output | `tiny_colony_net.pth` | `tiny_colony_net_greedy.pth` |
| Derived split (train/val) | 70 / 17 (N = 87) | 2,168 / 542 (N = 2,710) |

Loss curves, best epochs, and final losses were not recorded and are not reported.

### G. Split Leakage

Splitting individual positions at random can place adjacent, highly correlated positions from the same game into both the training and validation partitions. Validation loss is then optimistic, and early stopping on it is not a reliable proxy for generalization to unseen games. A game-level split is the appropriate design (Section XII).

---

## VI. ONNX Export and Runtime Validation

### A. Export Procedure

Both export scripts load a checkpoint with `weights_only=True`, switch the model to evaluation mode, and trace with a dummy input of shape (2, 16, 8, 8). They name the tensors `observation`, `policy` and `value`, declare axis 0 dynamic on all three, and use opset 17 with `do_constant_folding=True` and `external_data=False`. `export_onnx.py` converts `tiny_colony_net.pth` into `model.onnx`, and `export_greedy_onnx.py` converts `tiny_colony_net_greedy.pth` into `model_greedy.onnx`.

The meaning of `dynamic_axes` and the default exporter path depend on the PyTorch version [12]. The version used was not recorded and is not pinned in `requirements.txt`.

**Provenance of the "original" model.** By the script chain, `model.onnx` comes from the checkpoint written by `train_model.py` on the random-evaluator data. The untracked `backups/` directory suggests earlier snapshots existed, so the weights behind the `model.onnx` used in the comparison are marked for confirmation.

### B. Validator and Runtime Measurements

**TABLE V. Recorded validator and runtime measurements. Original `model.onnx` only; one recorded run; not a benchmark.**

| Measurement | Recorded value |
|---|---|
| Validator verdict | Accepted |
| Dry-run time | ≈ 127.2 ms |
| Peak resident memory | ≈ 82 MiB |
| CPU time | ≈ 0.94 s |
| Level 3 smoke-test match vs. GreedyAgent | 40 actions, ≈ 2.51 s CPU time |

These values sit below the published local guard values [2]. Which recorded value corresponds to which guard was not recorded. Whether `model_greedy.onnx` passed the validator was not recorded and is not claimed.

### C. What Validation Establishes

Validator acceptance shows that the file meets the structural, resource, and match-time requirements of the submission contract [2]. It does not measure move quality, which depends on the data, the model, and the search [10]. A single dry run on one machine is also not a latency benchmark.

---

## VII. Experimental Protocol

### A. Procedure

`evaluate_greedy.py` takes a model path, a game count, a simulation count, and a base seed s (defaults 10 games, 60 simulations, s = 42). For game index i = 0, …, n−1 it:

1. Creates `ColonyOneEnv()` and an `OnnxEvaluator` for the model, and sets `MCTSConfig(simulations=…)`.
2. Seeds the search generator with `default_rng(s + i)` and creates `GreedyAgent(seed = s + i + 1000)`.
3. Assigns the model to Cyan for even i and Magenta for odd i.
4. Uses MCTS for the model's actions and GreedyAgent for the opponent's, until the game ends or no action is available.
5. Tallies model wins, GreedyAgent wins, unfinished games, total actions, the completed-game win rate and average actions per game.

The reported comparison used 20 games, 60 simulations per move (the Level 3 value [2]), and seed settings based on 42. The exact command line was not recorded; Appendix A gives the presumed invocation.

### B. Seeding and Determinism Audit

**TABLE VI. Randomness sources in the evaluation.**

| Source | Controlled? | Evidence |
|---|---|---|
| MCTS root noise and action sampling | Seeded via `default_rng(s + i)`, assuming the search draws all randomness from the supplied generator | `evaluate_greedy.py`; `mcts.py` not inspected |
| GreedyAgent tie-breaking | Seeded with `s + i + 1000` | `evaluate_greedy.py` |
| Side assignment | Deterministic alternation | `evaluate_greedy.py` |
| Environment | `ColonyOneEnv()` called without a seed. The constructor accepts one [2], but its default behavior is unverified | `env.py` not inspected |
| ONNX Runtime numerics | Not verified | — |
| Software versions | PyTorch not recorded; `requirements.txt` pins numpy, onnx and onnxruntime, but the installed versions were not verified | `requirements.txt` |

The matches therefore cannot be described as fully deterministic. When both models are run with the same seed arguments, per-game search and opponent seeds coincide, but play diverges once the models choose different actions. This is not a controlled paired design.

### C. Context

Development was reported on an Ubuntu laptop (Dell Latitude E7470, 6th-generation Intel Core i5, 8 GB random-access memory, 128 GB solid-state drive). A cached bytecode path suggests Python 3.12, which falls within the upstream-supported range [2], but the interpreter and package versions were not recorded.

---

## VIII. Results

### A. Match Results

**TABLE VII. Twenty-game comparison against GreedyAgent (60 simulations per move). Counts are the primary result.**

| Model | Artifact | Games | Model wins | Reported fraction | GreedyAgent wins / unfinished |
|---|---|---|---|---|---|
| Original | `model.onnx` | 20 | 7 | 35% (7/20) | not recorded |
| GreedyAgent-trained | `model_greedy.onnx` | 20 | 6 | 30% (6/20) | not recorded |
| Difference | — | — | −1 | −5 percentage points (descriptive) | — |

The GreedyAgent-trained model recorded one fewer win. The upstream submission checklist includes beating GreedyAgent more often than not [2], and both recorded results are below 10 wins in 20 games.

**Scale of the noise (illustration only).** For a win proportion near 0.35 from 20 independent games, the standard error is √(0.35 × 0.65 / 20) ≈ 0.11, so a one-game difference (0.05) is about half of that. This is an order-of-magnitude illustration, not a statistical test. No confidence interval, p-value, or significance claim is made, and the two runs are not guaranteed to be independent.

### B. Training Results

No training loss, validation loss, epochs reached, or best-epoch values were recorded, and none are reported.

---

## IX. Discussion

**No demonstrated improvement.** The experiment did not show that training on GreedyAgent data improved the model over the original, and it did not show the reverse. Twenty games cannot separate a one-game difference from chance.

**Contract versus strength.** Matching the output contract (Table I), exporting with dynamic axes, and passing the validator are interface milestones [2], [10]. They are independent of whether the policy and value are useful. RQ1 is therefore answered by engineering evidence, and RQ3 is not answered in favor of either model.

**Data quality.** Random-evaluator labels carry almost no strategic information (Section V-B). GreedyAgent labels can teach imitation but are bounded by the heuristic, and a deterministic-looking teacher produces a narrow state distribution, so a model-guided search may visit states unlike those it saw in training [12]. Outcome labels give every position in a game the same final result, which is a noisy credit assignment. The guided self-play dataset (76 positions from two games) shows that the path executed and supports no learning conclusion.

**Hypotheses for the neutral result (untested).**
1. 2,710 positions are small for a 4,096-way output.
2. The policy-head bottleneck (Section IV) limits what the network can express.
3. The unmasked softmax spends probability mass on illegal actions that the referee never considers [2].
4. Imitating both the winning and losing side of heuristic games dilutes the signal.
5. With 60 simulations, search may dominate the evaluator's influence, so weak evaluators can look alike.
6. Evaluation noise hides small differences.
7. If the original model's weights are close to an untrained or random-data network, the comparison is against a very weak baseline evaluator (provenance unconfirmed).

None of these was tested.

---

## X. Engineering Journey and Lessons Learned

### A. Chronology

1. Read the upstream interface documents; established the tensor contract and the one-action-per-step semantics [2].
2. Implemented `TinyColonyNet`, exported `model.onnx` (opset 17, dynamic batch), and obtained validator acceptance and a Level 3 smoke match.
3. Generated 87 random-evaluator positions and ran the first training pipeline (40 epochs) as a pipeline test.
4. Judged random-evaluator data strategically uninformative.
5. Built the GreedyAgent data generator and trained with early stopping, then exported `model_greedy.onnx`.
6. Compared the two models over 20 games (7/20 vs. 6/20).
7. Ran a small guided self-play generation (76 positions, two games) and judged it too small for conclusions.
8. After the contest closed, shifted to documentation, attribution, Git hygiene, and this report.

### B. Lessons

1. Matching the network's output contract is an independent milestone from building a strong policy.
2. Validator acceptance proves interface and runtime compatibility under its checks, not strategic ability [2], [10].
3. Supervised learning is bounded by its labels: random-evaluator labels are poor targets, and GreedyAgent labels inherit the heuristic's limits.
4. Fewer than 100 positions cannot support learning claims.
5. Lower validation loss need not mean better game outcomes, especially with correlated states and limited evaluation.
6. Game-level train/validation separation would be better than splitting adjacent positions.
7. A 20-game, 60-simulation comparison is a smoke test, not a benchmark.
8. Reproducibility needs more than a few seeds: verify environment seeding, record versions and hardware, save logs and configurations, and play more games.
9. The deadline changed the objective from squeezing out a Level 3 result to documenting a useful case study.
10. Good engineering includes keeping large virtual environments and transient artifacts out of Git, maintaining attribution, and checking license constraints early.

### C. Setup Note

At one point `python --version` returned "command not found" and the shell suggested `python3`; later terminal output showed an active `.venv`. The resolution sequence was not recorded. The `.venv` occupied about 5.5 GB against a few megabytes of project artifacts and is excluded by `.gitignore`, as are caches, environment files, logs, `backups/`, `selfplay_data.npz`, `selfplay_data_random_backup.npz` and the two `.pth` checkpoints. A filename scan for obvious `.env`, credential and secret-named files found no matches, which is not a guarantee that no sensitive data exist. The branch was renamed to `main` and the first push succeeded. As observed on 25 October 2026, the public README is still the upstream README, and the rewritten README is a local, uncommitted change.

---

## XI. Limitations and Threats to Validity

**TABLE VIII. Claims, evidence, and limitations.**

| Claim | Evidence | Limitation |
|---|---|---|
| The model implements the tensor contract | Source (`tiny_colony_net.py`); validator accepted `model.onnx` | `model_greedy.onnx` validation not recorded |
| Original `model.onnx` passes the validator | Recorded run | One run, one machine; not a benchmark |
| GreedyAgent training produced a checkpoint | Training script and exported file | No loss values or logs recorded |
| 7/20 vs. 6/20 wins | Author's record | Raw per-game logs, unfinished counts and GreedyAgent wins not recorded |
| No improvement was demonstrated | Table VII | Cannot show the reverse either |
| Results are reproducible | Several seeds fixed | Environment seeding unverified; PyTorch version unrecorded |

Further threats:

- **Small samples.** 20 games, and datasets of 87, 2,710 and 76 positions.
- **Weak teachers.** Random-evaluator search and a heuristic baseline.
- **Split leakage.** Position-level splits (Section V-G).
- **Objective mismatch.** The unmasked softmax differs from the referee's legal-action softmax [2].
- **Unknown weight provenance** of the original `model.onnx`.
- **No formal statistics.** No verified statistical analysis was performed.
- **Single hardware setup.** All measurements come from one low-end laptop.

### Attribution and License Caution

This work uses the upstream Colony: One environment [2]; the upstream components in Section III-A are not the author's creation, and citing the source does not by itself address licensing. The repository's `LICENSE` file names the upstream copyright holder, permits non-commercial use, copying, modification and distribution under stated conditions, requires its notice to be retained, and states separate conditions for commercial use, use at events, workshops, courses and competitions, and derivative works. This report does not interpret those terms. The license and attribution text must stay unchanged. As a project-management caution, publication and redistribution permission should be clarified with the copyright holder before further public distribution.

---

## XII. Future Work

The following are proposals, not completed work.

1. Use a **game-level** train/validation split and hold out whole games.
2. **Mask illegal actions** in the training loss to match the referee [2].
3. Run a **larger, controlled evaluation**: hundreds of games, side-balanced, with recorded per-game seeds passed to the environment, full raw logs, and a binomial interval.
4. Add baselines: the untrained network, the shipped random model [2], GreedyAgent alone, and direct model-versus-model matches.
5. Use a **paired design** with identical game seeds and report the paired outcomes.
6. Generate **stronger targets**: much more guided self-play with more simulations, and aggregate states visited by the learner's own search [12].
7. Run **architecture ablations**, for example a wider policy head or spatial features in the value head.
8. Record **versions and hardware**, lock dependencies including PyTorch, and save configurations, logs and checksums.
9. Run the **validator on `model_greedy.onnx`**.
10. **Clarify permissions** with the copyright holder (Section XI).

---

## XIII. Conclusion

A compact CNN policy–value model was implemented, trained on three small datasets, exported to ONNX, and compared against the environment's baseline. RQ1 is answered affirmatively for the original model, which the validator accepted. RQ2 is answered at the level of the pipeline: the GreedyAgent data generation, training and export scripts ran, although no loss values were recorded. RQ3 is not answered in favor of either model: the GreedyAgent-trained model won 6 of 20 games against 7 of 20 for the original, and the experiment did not demonstrate improved performance. RQ4 produced concrete lessons about data quality, evaluation size, seeding and versioning. The value of the project lies in a documented end-to-end workflow and an honest account of its limits.

---

## Acknowledgment

None declared. [Author to add any funding or assistance disclosures required by the intended venue.]

---

## References

[1] Cloudflight GmbH, "The Cloudflight Coding Contest." [Online]. Available: https://www.cloudflight.io/en/the-cloudflight-coding-contest/ (accessed Oct. 25, 2026).

[2] P. Puntschart, "Colony: One tournament environment," README, `docs/observation.md`, `docs/submission.md` and `LICENSE`, as included in the repository snapshot. [Online]. Available: https://github.com/ayan-humayoun56/colony-one-neural-network (accessed Oct. 25, 2026).

[3] R. Coulom, "Efficient selectivity and backup operators in Monte-Carlo tree search," in *Computers and Games (CG 2006)*, Lecture Notes in Computer Science. Springer, 2007, pp. 72–83. [Online]. Available: https://remi-coulom.fr/CG2006

[4] L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo planning," in *Machine Learning: ECML 2006*, Lecture Notes in Computer Science, vol. 4212. Berlin, Germany: Springer, 2006, pp. 282–293, doi: 10.1007/11871842_29.

[5] C. B. Browne *et al.*, "A survey of Monte Carlo tree search methods," *IEEE Trans. Comput. Intell. AI Games*, vol. 4, no. 1, pp. 1–43, Mar. 2012, doi: 10.1109/TCIAIG.2012.2186810.

[6] D. Silver *et al.*, "Mastering the game of Go with deep neural networks and tree search," *Nature*, vol. 529, no. 7587, pp. 484–489, Jan. 2016, doi: 10.1038/nature16961.

[7] D. Silver *et al.*, "Mastering the game of Go without human knowledge," *Nature*, vol. 550, no. 7676, pp. 354–359, Oct. 2017, doi: 10.1038/nature24270.

[8] D. Silver *et al.*, "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, vol. 362, no. 6419, pp. 1140–1144, Dec. 2018, doi: 10.1126/science.aar6404.

[9] ONNX, "Open Neural Network Exchange." [Online]. Available: https://onnx.ai/ (accessed Oct. 25, 2026).

[10] ONNX Runtime, "Welcome to ONNX Runtime," ONNX Runtime documentation. [Online]. Available: https://onnxruntime.ai/docs/ (accessed Oct. 25, 2026).

[11] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019, arXiv:1912.01703.

[12] S. Ross, G. J. Gordon, and J. A. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," in *Proc. 14th Int. Conf. Artif. Intell. Statist. (AISTATS)*, ser. PMLR, vol. 15, 2011, pp. 627–635. [Online]. Available: https://proceedings.mlr.press/v15/ross11a

[13] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learn. Represent. (ICLR)*, San Diego, CA, USA, 2015, arXiv:1412.6980.

[14] PyTorch contributors, "torch.onnx," PyTorch documentation. [Online]. Available: https://docs.pytorch.org/docs/stable/onnx.html (accessed Oct. 25, 2026).

---

## Appendix A. Reproducibility Commands and Artifact Map

**Figure placeholders** (not yet generated):
- [Figure 1: Pipeline diagram (observation → CNN stem → policy/value heads → ONNX → MCTS → action; datasets → checkpoint → ONNX export) to be generated from the described stages.]
- [Figure 2: Tensor-shape diagram to be generated from Table II.]
- [Figure 3: Per-game outcome chart to be generated only once raw per-game logs exist.]

**TABLE A1. Commands (from the scripts' defaults; invocations not recorded for the starred rows).**

| Step | Command | Output |
|---|---|---|
| Random-evaluator data | `python train_selfplay.py` | `selfplay_data.npz` |
| Initial training | `python train_model.py` | `tiny_colony_net.pth` |
| Export | `python export_onnx.py` | `model.onnx` |
| Validate | `python tools/validate_submission.py model.onnx` | verdict |
| GreedyAgent data | `python train_greedy_data.py --games 100 --seed 42` | `greedy_data.npz` |
| GreedyAgent training | `python train_greedy_model.py` | `tiny_colony_net_greedy.pth` |
| Export | `python export_greedy_onnx.py` | `model_greedy.onnx` |
| Evaluate* | `python evaluate_greedy.py --model model.onnx --games 20 --simulations 60 --seed 42` | counts |
| Guided data* | `python train_guided_selfplay.py --model model.onnx --games N --simulations S --seed 42` | `guided_selfplay_data.npz` |

**TABLE A2. Artifact map (public `main` branch, 25 October 2026).**

| Artifact | Role | In Git? | Origin |
|---|---|---|---|
| `colony_one/`, `docs/`, `play/`, `tools/`, `LICENSE` | Environment, docs, tools | Yes | Upstream [2] |
| `tiny_colony_net.py`, `train_*.py`, `export_*.py`, `evaluate_greedy.py` | Experiment code | Yes | Author |
| `greedy_data.npz`, `guided_selfplay_data.npz` | Datasets | Yes | Author (generated) |
| `model.onnx`, `model_greedy.onnx` | Exported models | Yes | Author (generated) |
| `selfplay_data*.npz`, `*.pth`, `backups/` | Local artifacts | No (ignored) | Author |