#!/usr/bin/env python3

import gym

from vpg_Tho.sit_il.utils import random_seed
from vpg_Tho.sit_il.models.rl import QLearning


def main() -> None:
    env_name = "Taxi-v3"
    env = gym.make(env_name)

    max_episode_length = 100
    env._max_episode_steps = max_episode_length  # pylint: disable=protected-access

    model = QLearning(env)

    # set random seed
    random_seed.set_numpy()
    random_seed.set_tensorflow()
    random_seed.set_gym(env)

    print(
        "Environment:\n"
        f"  name = {env_name}\n"
        f"  observation_space = {env.observation_space}\n"
        f"  action_space = {env.action_space}",
    )
    print("====================")
    print()

    model.pipeline(
        n_train_episodes=50000,
        max_episode_length=max_episode_length,
        discount_rate=0.6,
        exploration_rate=1.0,
        decay_rate=0.005,
        learning_rate=0.8,
        n_test_episodes=10,
    )


if __name__ == "__main__":
    main()
