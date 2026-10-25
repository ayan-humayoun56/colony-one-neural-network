
import argparse
import numpy as np

from colony_one.baselines import GreedyAgent
from colony_one.constants import CYAN, MAGENTA
from colony_one.encoding import action_index
from colony_one.env import ColonyOneEnv


def generate_games(num_games=100, seed=42):
    rng = np.random.default_rng(seed)

    observations = []
    policies = []
    values = []

    for game_num in range(num_games):
        env = ColonyOneEnv()
        env.reset()

        # Separate agents with reproducible random tie-breaking.
        agents = {
            CYAN: GreedyAgent(seed=int(rng.integers(2**31))),
            MAGENTA: GreedyAgent(seed=int(rng.integers(2**31))),
        }

        positions = []

        while not env.terminated:
            team = env.current_team
            observation = env.observation().copy()

            action = agents[team].select(
                env.board,
                team,
                env.actions_left,
                env.round_number,
            )

            if action is None:
                break

            # A one-hot policy target: imitate the teacher's chosen action.
            idx = action_index(action, team)
            policy = np.zeros(4096, dtype=np.float32)
            policy[idx] = 1.0

            positions.append((observation, policy, team))
            env.step(action)

        if env.winner is None:
            print(f"Game {game_num + 1}: no winner; skipping")
            continue

        for observation, policy, team in positions:
            value = 1.0 if env.winner == team else -1.0
            observations.append(observation)
            policies.append(policy)
            values.append(value)

        print(
            f"Game {game_num + 1}/{num_games}: "
            f"winner={env.winner}, positions={len(positions)}"
        )

    if not observations:
        raise RuntimeError("No training positions were generated.")

    np.savez_compressed(
        "greedy_data.npz",
        observations=np.asarray(observations, dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
    )

    print(f"\nSaved {len(observations)} positions to greedy_data.npz")
    print("Observations:", np.asarray(observations).shape)
    print("Policies:", np.asarray(policies).shape)
    print("Values:", np.asarray(values).shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_games(args.games, args.seed)
