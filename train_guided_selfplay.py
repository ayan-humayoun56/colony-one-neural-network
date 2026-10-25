
import argparse
from pathlib import Path

import numpy as np

from colony_one import mcts
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator


def generate_games(
    model_path="model.onnx",
    num_games=5,
    simulations=20,
    seed=42,
    output_path="guided_selfplay_data.npz",
):
    rng = np.random.default_rng(seed)
    evaluator = OnnxEvaluator(Path(model_path))

    config = mcts.MCTSConfig(
        simulations=simulations,
        batch_size=1,
        add_root_noise=True,
    )

    observations = []
    policies = []
    values = []

    for game_num in range(num_games):
        env = ColonyOneEnv()
        env.reset()
        game_positions = []

        while not env.terminated:
            team = env.current_team
            observation = env.observation().copy()

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

            if policy.sum() <= 0:
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
            print(f"Game {game_num + 1}: no winner; skipping its positions")
            continue

        for observation, policy, team in game_positions:
            value = 1.0 if env.winner == team else -1.0
            observations.append(observation)
            policies.append(policy)
            values.append(value)

        print(
            f"Game {game_num + 1}/{num_games}: "
            f"winner={env.winner}, "
            f"positions={len(game_positions)}"
        )

    if not observations:
        raise RuntimeError("No completed games produced training positions.")

    np.savez_compressed(
        output_path,
        observations=np.asarray(observations, dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
    )

    print(f"\nSaved {len(observations)} positions to {output_path}")
    print("Observations:", np.asarray(observations).shape)
    print("Policies:", np.asarray(policies).shape)
    print("Values:", np.asarray(values).shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="guided_selfplay_data.npz",
    )
    args = parser.parse_args()

    generate_games(
        model_path=args.model,
        num_games=args.games,
        simulations=args.simulations,
        seed=args.seed,
        output_path=args.output,
    )

