#!/usr/bin/env python3

import time

import gym


def main() -> None:
    env_name = "FetchPickAndPlace-v1"
    env = gym.make(env_name)
    env.reset()

    try:
        for _ in range(1000):
            env.render()
            action = env.action_space.sample()
            env.step(action)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    main()
