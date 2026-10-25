import numpy as np
import torch

from colony_one import mcts
from colony_one.env import ColonyOneEnv
from colony_one.constants import CYAN, MAGENTA


class RandomEvaluator:
    """Temporary evaluator used to test the self-play data pipeline."""

    def __call__(self, planes):
        batch_size = len(planes)

        # Random logits are a temporary stand-in for the neural network.
        logits = np.random.randn(batch_size, 4096).astype(np.float32)

        # Neutral value estimates: the model has not learned yet.
        values = np.zeros(batch_size, dtype=np.float32)

        return logits, values


def generate_games(num_games=2, simulations=20, seed=42):
    rng = np.random.default_rng(seed)
    evaluator = RandomEvaluator()

    config = mcts.MCTSConfig(
        simulations=simulations,
        batch_size=1,
        add_root_noise=True,
    )

    observations = []
    policies = []
    teams = []
    results = []

    for game_num in range(num_games):
        env = ColonyOneEnv()
        env.reset()

        game_positions = []

        while not env.terminated:
            team = env.current_team

            # Save the position before the move.
            observation = env.observation().copy()

            # Run MCTS and retain its visit distribution.
            root = mcts.run(
                env.board,
                team,
                env.actions_left,
                env.round_number,
                evaluator,
                config,
                rng=rng,
            )

            policy = mcts.policy_target(root)

            if policy.sum() == 0:
                # Defensive fallback: choose a legal action uniformly.
                legal_indices = env.legal_action_indices()
                if len(legal_indices) == 0:
                    break

                action_index = int(rng.choice(legal_indices))
            else:
                action_index = mcts.select_action_index(
                    root,
                    temperature=1.0,
                    rng=rng,
                )

            game_positions.append((observation, policy, team))

            env.step(action_index)

        if env.winner is None:
            print(f"Game {game_num + 1}: no winner recorded; skipping")
            continue

        # Assign values from each saved position's team's perspective.
        for observation, policy, team in game_positions:
            value = 1.0 if env.winner == team else -1.0

            observations.append(observation)
            policies.append(policy)
            teams.append(team)
            results.append(value)

        print(
            f"Game {game_num + 1}/{num_games}: "
            f"winner={env.winner}, "
            f"positions={len(game_positions)}"
        )

    if not observations:
        raise RuntimeError("No training positions were generated.")

    output_path = "selfplay_data.npz"

    np.savez_compressed(
        output_path,
        observations=np.asarray(observations, dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        values=np.asarray(results, dtype=np.float32),
    )

    print(f"\nSaved {len(observations)} positions to {output_path}")
    print("Observation array:", np.asarray(observations).shape)
    print("Policy array:", np.asarray(policies).shape)
    print("Value array:", np.asarray(results).shape)


if __name__ == "__main__":
    generate_games(num_games=2, simulations=20)
