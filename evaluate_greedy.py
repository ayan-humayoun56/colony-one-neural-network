
import argparse
from pathlib import Path

import numpy as np

from colony_one import constants, mcts
from colony_one.baselines import GreedyAgent
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator


def play_game(model_path, model_team, seed, simulations):
    env = ColonyOneEnv()
    evaluator = OnnxEvaluator(Path(model_path))
    config = mcts.MCTSConfig(simulations=simulations)
    rng = np.random.default_rng(seed)

    # Give each game a reproducible but different greedy opponent.
    opponent = GreedyAgent(seed=seed + 1000)
    actions = 0

    while not env.terminated:
        if env.current_team == model_team:
            action = mcts.search_action(
                env.board,
                env.current_team,
                env.actions_left,
                env.round_number,
                evaluator,
                config,
                rng=rng,
            )
        else:
            action = opponent.select(
                env.board,
                env.current_team,
                env.actions_left,
                env.round_number,
            )

        if action is None:
            break

        env.step(action)
        actions += 1

    return env.winner, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_wins = 0
    greedy_wins = 0
    unfinished = 0
    total_actions = 0

    print(f"Model: {args.model}")
    print(f"Games: {args.games}, simulations per move: {args.simulations}")

    for i in range(args.games):
        # Alternate sides to reduce first-player/side bias.
        model_team = constants.CYAN if i % 2 == 0 else constants.MAGENTA
        winner, actions = play_game(
            args.model, model_team, args.seed + i, args.simulations
        )
        total_actions += actions

        if winner is None:
            unfinished += 1
            result = "unfinished"
        elif winner == model_team:
            model_wins += 1
            result = "MODEL WIN"
        else:
            greedy_wins += 1
            result = "GREEDY WIN"

        print(
            f"Game {i + 1:02d}: {result} | "
            f"model={'CYAN' if model_team == constants.CYAN else 'MAGENTA'} "
            f"| actions={actions}"
        )

    completed = model_wins + greedy_wins
    print("\n===== RESULTS =====")
    print(f"Model wins:  {model_wins}/{args.games}")
    print(f"Greedy wins: {greedy_wins}/{args.games}")
    print(f"Unfinished:  {unfinished}/{args.games}")
    if completed:
        print(f"Model win rate (completed games): {100 * model_wins / completed:.1f}%")
    print(f"Average actions per game: {total_actions / args.games:.1f}")


if __name__ == "__main__":
    main()
