from typing import Dict, List, Union

import gym
import numpy as np
import matplotlib.pyplot as plt

from vpg_Tho.sit_il.utils.moving_average import moving_average


class QLearning:
    """A simple Q-Learning agent."""

    def __init__(self, env: gym.Env):
        self.model_name = self.__class__.__name__

        self.env = env
        self.observation_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        self.q_table = np.zeros((self.observation_size, self.action_size))

        self.history: Dict[str, List[Union[int, float]]] = {
            "exploration_rate": [],
            "count_steps": [],
            "total_reward": [],
        }

    def pipeline(
        self,
        n_train_episodes: int,
        max_episode_length: int,
        discount_rate: float,
        exploration_rate: float,
        decay_rate: float,
        learning_rate: float,
        n_test_episodes: int,
    ) -> None:
        """Run the pipeline including training and testing the agent."""
        print(
            f"[1/3] Training model:\n"
            f"  name: {self.model_name}\n"
            f"  n_train_episodes: {n_train_episodes}\n"
            f"  max_episode_length: {max_episode_length}\n"
            f"  discount_rate: {discount_rate}\n"
            f"  exploration_rate: {exploration_rate}\n"
            f"  decay_rate: {decay_rate}\n"
            f"  learning_rate: {learning_rate}\n",
        )
        self.fit(
            n_episodes=n_train_episodes,
            max_episode_length=max_episode_length,
            discount_rate=discount_rate,
            exploration_rate=exploration_rate,
            decay_rate=decay_rate,
            learning_rate=learning_rate,
        )
        print("====================")
        print()

        print("[2/3] Plotting training results...")
        _, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].set_title("exploration_rate")
        axes[0].plot(self.history["exploration_rate"])

        axes[1].set_title("count_steps")
        axes[1].plot(moving_average(self.history["count_steps"], n_train_episodes // 100))
        axes[1].plot(self.history["count_steps"], color="gray", alpha=0.2)

        axes[2].set_title("total_reward")
        axes[2].plot(
            moving_average(self.history["total_reward"], n_train_episodes // 100),
        )
        axes[2].plot(self.history["total_reward"], color="gray", alpha=0.2)

        plt.show()
        print("====================")
        print()

        print("[3/3] Evaluating model...")
        results = self.evaluate(n_episodes=n_test_episodes)
        print(
            "Evaluation results:\n"
            "  count_steps: "
            f"{results['count_steps_mean']:.4f} ± {results['count_steps_std']:.4f}\n"
            "  total_reward: "
            f"{results['total_reward_mean']:.4f} ± {results['total_reward_std']:.4f}",
        )
        print("====================")
        print()

    def fit(
        self,
        n_episodes: int,
        max_episode_length: int,
        discount_rate: float,
        exploration_rate: float,
        decay_rate: float,
        learning_rate: float,
    ) -> None:
        """Train the agent."""
        # pylint: disable=too-many-locals
        for episode in range(n_episodes):
            observation = self.env.reset()
            step = 0
            total_reward = 0.0
            done = False

            while not done:
                step += 1
                if step > max_episode_length:
                    break

                # choose an action
                epsilon = np.random.uniform(0, 1)
                if epsilon > exploration_rate:
                    action = np.argmax(self.q_table[observation, :])
                else:
                    action = self.env.action_space.sample()

                # take the action and observe the next observation and reward
                next_observation, reward, done, _ = self.env.step(action)

                # update Q-Table
                self.q_table[observation, action] += learning_rate * (
                    reward
                    + discount_rate * np.max(self.q_table[next_observation, :])
                    - self.q_table[observation, action]
                )

                total_reward += reward
                observation = next_observation

            if episode % 1000 == 0:
                print()
                print(
                    f"Episode {episode}:\n"
                    f"  exploration_rate = {exploration_rate}\n"
                    f"  count_steps = {step}\n"
                    f"  total_reward = {total_reward}",
                )
            self._log_history(exploration_rate, step, total_reward)

            exploration_rate = 0.01 + (1.0 - 0.01) * np.exp(-decay_rate * episode)

    def evaluate(self, n_episodes: int) -> Dict[str, float]:
        """Evalute the agent."""
        count_steps_history = []
        total_reward_history = []

        for episode in range(n_episodes):
            observation = self.env.reset()
            step = 0
            total_reward = 0.0
            done = False

            while not done:
                action = self.act(observation)
                next_observation, reward, done, _ = self.env.step(action)
                self.env.render()

                step += 1
                total_reward += reward
                observation = next_observation

            print(
                f"Episode {episode}:\n"
                f"  count_steps = {step}\n"
                f"  total_reward = {total_reward}",
            )
            print()
            count_steps_history.append(step)
            total_reward_history.append(total_reward)

        return {
            "count_steps_mean": np.mean(count_steps_history),
            "count_steps_std": np.std(count_steps_history),
            "total_reward_mean": np.mean(total_reward_history),
            "total_reward_std": np.std(total_reward_history),
        }

    def act(self, observation: np.ndarray) -> int:
        """Return an action based on input observation."""
        return int(np.argmax(self.q_table[observation, :]))

    def _log_history(
        self,
        exploration_rate: float,
        step: int,
        total_reward: float,
    ) -> None:
        self.history["exploration_rate"].append(exploration_rate)
        self.history["count_steps"].append(step)
        self.history["total_reward"].append(total_reward)
