# pylint: disable=no-name-in-module
from typing import Any, Dict, List, Tuple, Union, Optional

import tempfile
from pathlib import Path
from dataclasses import field, dataclass

import gym
import numpy as np
import pandas as pd

from tensorflow.keras.utils import plot_model

import wandb
from vpg_Tho.sit_il.helpers import compute_discounted_return
from vpg_Tho.sit_il.models.rl.vpg.mlp_critic import MLPCritic
from vpg_Tho.sit_il.models.rl.vpg.mlp_continuous_actor import MLPContinuousActor


@dataclass
class Trajectories:
    """
    A set of trajectories: an array to save data values (ex. obs, action, ...etc)
    """

    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    critic_values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def append(
        self,
        observation: np.ndarray,
        action: np.ndarrary, #action is made up of an array of 28 elements
        critic_value: float,
        reward: float,
    ) -> None:
        """Append a trajectory."""
        self.observations.append(observation)
        self.actions.append(action)
        self.critic_values.append(critic_value)
        self.rewards.append(reward)


class VPGContinuousAgent:
    """Vanilla Policy Gradient agent for environments with continuous actions spaces."""

    # pylint: disable=too-many-instance-attributes)

    actor: MLPContinuousActor
    critic: MLPCritic

    def __init__(
        self,
        env: gym.Env,
    ):
        self.model_name = self.__class__.__name__

        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

        self.is_in_wandb_session = False
        self.config: Dict[str, Union[int, float, str]] = {}

        self.history: Dict[str, List[Union[int, float]]] = {
            "episode": [],
            "count_steps": [],
            "total_reward": [],
            "actor_loss": [],
            "critic_loss": [],
        }

    # wrapper function for VPG
    def pipeline(
        self,
        #TODO: change actor network parameters
        actor_hidden_units: List[int],
        critic_hidden_units: List[int],
        actor_learning_rate: float,
        critic_learning_rate: float,
        n_train_episodes: int,
        max_episode_length: int,
        discount_rate: float,
        n_test_episodes: int,
        print_summary: bool,
        plot_actor_network_to_file: Optional[Path],
        plot_critic_network_to_file: Optional[Path],
    ) -> None:
        """Run the pipeline including building, training, and testing the agent."""
        # pylint: disable=too-many-arguments
        with wandb.init(
            project="sit_vpg",
            entity="edward_lin",
            tags=[self.model_name],
            resume=True,
            config={
                "env_name": self.env.spec.id,
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "actor_hidden_units": actor_hidden_units,
                "critic_hidden_units": critic_hidden_units,
                "actor_learning_rate": actor_learning_rate,
                "critic_learning_rate": critic_learning_rate,
                "n_train_episodes": n_train_episodes,
                "max_episode_length": max_episode_length,
                "discount_rate": discount_rate,
                "n_test_episodes": n_test_episodes,
            },
        ):
            self.is_in_wandb_session = True
            self.config = wandb.config

            # build actor & critic networks
            self.build(
                observation_size=self.config["observation_size"],  # type: ignore
                action_size=self.config["action_size"],  # type: ignore
                actor_hidden_units=self.config["actor_hidden_units"],  # type: ignore
                critic_hidden_units=self.config["critic_hidden_units"],  # type: ignore
                actor_learning_rate=self.config["actor_learning_rate"],  # type: ignore
                critic_learning_rate=self.config["critic_learning_rate"],  # type: ignore
            )

            # visualize model architecture
            if print_summary:
                self.summary()

            actor_plot_file, critic_plot_file = self.plot(
                plot_actor_network_to_file,
                plot_critic_network_to_file,
            )

            wandb.log(
                {
                    "actor_architecture": wandb.Image(str(actor_plot_file)),
                    "critic_architecture": wandb.Image(str(critic_plot_file)),
                },
            )

            # train the agent
            self.fit(
                n_episodes=self.config["n_train_episodes"],  # type: ignore
                max_episode_length=self.config["max_episode_length"],  # type: ignore
                discount_rate=self.config["discount_rate"],  # type: ignore
            )

            # evaluate the agent
            results = self.evaluate(n_episodes=n_test_episodes)
            print(
                "Evaluation results:\n"
                "  count_steps: "
                f"{results['count_steps_mean']:.4f} ± {results['count_steps_std']:.4f}\n"
                "  total_reward: "
                f"{results['total_reward_mean']:.4f} ± {results['total_reward_std']:.4f}",
            )
            # wandb.log(
            #     {
            #         "evaluation_results": wandb.Table(  # type: ignore
            #             dataframe=pd.DataFrame(results, index=[0]),
            #         ),
            #     },
            # )

        self.is_in_wandb_session = False

    def build(
        self,
        observation_size: int,
        action_size: int,
        actor_hidden_units: List[int],
        critic_hidden_units: List[int],
        actor_learning_rate: float,
        critic_learning_rate: float,
    ) -> None:
        """Build actor and critic networks."""
        self.actor = MLPContinuousActor()
        self.actor.build(
            observation_size=observation_size,
            action_size=action_size,
            hidden_units=actor_hidden_units,
            learning_rate=actor_learning_rate,
        )

        self.critic = MLPCritic()
        self.critic.build(
            observation_size=observation_size,
            hidden_units=critic_hidden_units,
            learning_rate=critic_learning_rate,
        )

    def summary(self) -> None:
        """Print the summary of the actor and critic networks."""
        print("Actor network:")
        print(self.actor.model.summary())
        print()
        print("====================")
        print()
        print("Critic network:")
        print(self.critic.model.summary())

    def plot(
        self,
        actor_to_file: Optional[Path] = None,
        critic_to_file: Optional[Path] = None,
    ) -> Tuple[Any, Any]:
        """Visualize actor and critic networks."""
        if actor_to_file is None:
            _, temp_file = tempfile.mkstemp(suffix=".jpg")
            actor_to_file = Path(temp_file)

        if critic_to_file is None:
            _, temp_file = tempfile.mkstemp(suffix=".jpg")
            critic_to_file = Path(temp_file)

        plot_model(
            self.actor.model,
            to_file=actor_to_file,
            show_shapes=True,
            show_dtype=True,
        )
        plot_model(
            self.critic.model,
            to_file=critic_to_file,
            show_shapes=True,
            show_dtype=True,
        )
        return actor_to_file, critic_to_file

    def fit(
        self,
        n_episodes: int,
        max_episode_length: int,
        discount_rate: float,
    ) -> None:
        """Train the agent."""
        # pylint: disable=too-many-locals
        # loop through episodes to collect trajectories (Pseudocode line 2, 3)
        for episode in range(n_episodes):
            observation = self.env.reset()
            step = 0
            total_reward = 0.0
            done = False

            trajectories = Trajectories()

            while not done:
                step += 1
                if step > max_episode_length:
                    break

                # selecting an action
                # observation = [
                #    [ s1, s2 ]
                # ] -> action = [
                #     [ a1 ],    (-1 <= a1 <= 1)
                # ]
                #TODO: Generate an aciton from observation using ACM (START HERE FOR CONTINUOUS)
                action_probs = self.actor.model.predict(np.atleast_2d(observation))
                action = np.random.choice(self.action_size, p=action_probs[0])

                # take the action and observe the next observation and reward
                next_observation, reward, done, _ = self.env.step(action)

                critic_value = self.critic.model.predict(np.atleast_2d(observation))[0, 0]

                total_reward += reward
                trajectories.append(
                    observation=observation,
                    action=action,
                    critic_value=critic_value,
                    reward=reward,
                )
                observation = next_observation

            # Computer rewards-to-go (= discounted_returns) (Pseudocode line 4)
            discounted_returns = compute_discounted_return(
                trajectories.rewards,
                discount_rate,
            )
            # Compute advantage estimate (Pseudocode line 5)
            advantages = np.subtract(discounted_returns, trajectories.critic_values)

            # Calling for the "actual training" (Pseudocode line 6-8)
            actor_loss = self.actor.fit(
                np.atleast_2d(trajectories.observations),
                np.expand_dims(trajectories.actions, axis=1),
                np.expand_dims(advantages, axis=1),
            )
            critic_loss = self.critic.fit(
                np.atleast_2d(trajectories.observations),
                np.expand_dims(discounted_returns, axis=1),
            )

            self._log_history(episode, step, total_reward, actor_loss, critic_loss)

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
                new_observation, reward, done, _ = self.env.step(action)
                self.env.render()

                step += 1
                total_reward += reward
                observation = new_observation

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

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return an action based on input observation."""
        # stochastic, action selected based on probability from the training
        return np.random.choice(
            self.action_size,
            p=self.actor.model.predict(np.atleast_2d(observation))[0],
        )

    def _log_history(
        self,
        episode: int,
        count_steps: int,
        total_reward: float,
        actor_loss: float,
        critic_loss: float,
    ) -> None:
        self.history["episode"].append(episode)
        self.history["count_steps"].append(count_steps)
        self.history["total_reward"].append(total_reward)
        self.history["actor_loss"].append(actor_loss)
        self.history["critic_loss"].append(critic_loss)

        if self.is_in_wandb_session:
            wandb.log(
                {
                    "episode": episode,
                    "count_steps": count_steps,
                    "total_reward": total_reward,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                },
            )
