#!/usr/bin/env python3

import gym

from vpg_Tho.sit_il.utils import random_seed
from vpg_Tho.sit_il.models.rl import VPGContinuousAgent


def main() -> None:
    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)

    max_episode_length = 100
    env._max_episode_steps = max_episode_length  # pylint: disable=protected-access

    model = VPGContinuousAgent(env)

    # set random seed
    # changing stochastic results to deterministic results (the results will be the same for everytime we train)
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
        actor_hidden_units=[32, 16, 16],
        critic_hidden_units=[32, 16, 16],
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        n_train_episodes=500,
        max_episode_length=max_episode_length,
        discount_rate=0.99,
        n_test_episodes=10,
        print_summary=True,
        plot_actor_network_to_file=None,
        plot_critic_network_to_file=None,
    )


if __name__ == "__main__":
    main()
