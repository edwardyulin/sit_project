# imports
from typing import List, Dict, Any, Tuple, Union, Optional

import tempfile
from pathlib import Path
from dataclasses import field, dataclass

import gym
import numpy as np
import pandas as pd

from tensorflow.keras.utils import plot_model

import wandb

from sit_il.helpers import compute_discounted_return
from sit_il.models.rl.npg.vpg.mlp_critic import MLPCritic
from sit_il.models.rl.npg.vpg.mlp_discrete_actor import MLPDiscreteActor

@dataclass # for storing data
class Trajectories:
    """ Storing trajectories"""

    # an obs is defined as [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    observations: List[np.ndarray] = field(default_factory=list)
    # an action is either 0 (left) and 1 (right)
    actions: List[int] = field(default_factory=list)
    # critic value from the output of critic network
    critic_values: List[float] = field(default_factory=list)
    # reward is 1 for each step taken, total reward for an episode = # steps taken in that episode
    rewards: List[float] = field(default_factory=list)

    def append(self,
               obs: np.ndarray,
               action: int,
               critic_value: float,
               reward: float
               ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.critic_values.append(critic_value)
        self.rewards.append(reward)

class VPGDiscreteAgent:
    """Vanilla Policy Gradient agent for environments with discrete actions spaces."""

    actor: MLPDiscreteActor
    critic: MLPCritic

    def __init__(self,
                 env: gym.Env
                 ):
        # define variables for VPGDiscreteAgent
        self.model_name = self.__class__.__name__
        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.is_in_wandb_session = False
        self.config: Dict[str, Union[int, float, str]] = {}
        self.history: Dict[str, List[Union[int, float]]] = {
            "episode": [],
            "count_steps": [],
            "total_reward": [],
            "actor_loss": [],
            "critic_loss": []
        }


    def pipeline(
        self,
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
        plot_critic_network_to_file: Optional[Path]
    ) -> None:
        """Run the pipeline including building, training, and testing the agent."""

        with wandb.init(
            project="sit_vpg_LunarLanderContinuous",
            entity="edward_lin",
            tags=[self.model_name],
            resume=False,
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
            self.is_in_wandb_session = True # enable graph plotting on wandb
            self.config = wandb.config

            # build actor & critic networks
            self.build(
                obs_size=self.config["observation_size"],
                action_size=self.config["action_size"],
                actor_hidden_units=self.config["actor_hidden_units"],
                critic_hidden_units=self.config["critic_hidden_units"],
                actor_learning_rate=self.config["actor_learning_rate"],
                critic_learning_rate=self.config["critic_learning_rate"]
            )

            # visualize model architecture
            if print_summary: # boolean from input from the pipeline function
                self.summary()

            # drawing the structures of actor and critic networks to file
            actor_plot_file, critic_plot_file = self.render_networks(
                plot_actor_network_to_file,
                plot_critic_network_to_file
            )
            # log the images of networks onto wandb
            wandb.log(
                {
                    "actor_architecture": wandb.Image(str(actor_plot_file)),
                    "critic_architecture": wandb.Image(str(critic_plot_file))
                }
            )

            #train the agent
            self.fit(
                n_episodes=self.config["n_train_episodes"],
                max_episode_length=self.config["max_episode_length"],
                discount_rate=self.config["discount_rate"]
            )

            # evaluate the agent and log results onto wandb
            results = self.evaluate(n_episodes=n_test_episodes)
            # An episode succeed if step = 100 and total_reward = 100
            print(
                "Evaluation results:\n"
                "  count_steps: "
                f"{results['count_steps_mean']:.4f} ± {results['count_steps_std']:.4f}\n"
                "  total_reward: "
                f"{results['total_reward_mean']:.4f} ± {results['total_reward_std']:.4f}",
            )
            # Log evaluation results
            wandb.log(
                {
                    "evaluation_results": wandb.Table(
                        dataframe=pd.DataFrame(results, index=[0])
                    )
                }
            )
        self.is_in_wandb_session = False # finished logging on wandb






    def build(self,
              obs_size: int,
              action_size: int,
              actor_hidden_units: List[int],
              critic_hidden_units: List[int],
              actor_learning_rate: float,
              critic_learning_rate: float
              ) -> None:
        """ Construct actor network and critic network """

        # Define the actor
        self.actor = MLPDiscreteActor()
        # Construct the actor network
        self.actor.build(
            obs_size=obs_size,
            action_size=action_size,
            hidden_units=actor_hidden_units,
            learning_rate=actor_learning_rate
        )

        # Define the critic
        self.critic = MLPCritic()
        # Construct the critic network
        self.critic.build(
            obs_size=obs_size,
            hidden_units=critic_hidden_units,
            learning_rate=critic_learning_rate,
        )


    def summary(self) -> None:
        """ Print the summary of the actor and critic networks """

        print("Actor network:")
        print(self.actor.model.summary())
        print()
        print("=======================")
        print("Critic network:")
        print(self.critic.model.summary())


    def render_networks(self,
                        actor_to_file: Optional[Path] = None,
                        critic_to_file: Optional[Path] = None
                        ) -> Tuple[Any, Any]:
        """ Visualize the structure (input, hidden, output) of actor and critic networks"""

        if actor_to_file is None:
            _, temp_file = tempfile.mkstemp(suffix=".jpg")
            actor_to_file = Path(temp_file) # find the path of temp_file

        if critic_to_file is None:
            _, temp_file = tempfile.mkstemp(suffix=".jpg")
            critic_to_file = Path(temp_file)  # find the path of temp_file


        plot_model(
            self.actor.model,
            to_file=actor_to_file,
            show_shapes=True,
            show_dtype=True
        ),
        plot_model(
            self.critic.model,
            to_file=critic_to_file,
            show_shapes=True,
            show_dtype=True
        )
        return actor_to_file, critic_to_file


    def fit(self,
            n_episodes: int,
            max_episode_length: int,
            discount_rate: int
            ) -> None:
        """ Train the agent"""

        # For all episodes... (Pseudocode line 2)
        for episode in range(n_episodes):
            # Initialize variables
            observation = self.env.reset()
            step = 0
            total_reward = 0.0
            done = False

            trajectories = Trajectories()

            # Training (for one episode)
            while not done:
                step += 1
                # Terminate after 100 (or w/e max_episode_length is) timesteps
                if step > max_episode_length:
                    break

                # Pair each available action with a probability given the observation
                action_probs = self.actor.model.predict(np.atleast_2d(observation))
                # Randomly generate an action based on action_probs
                action = np.random.choice(self.action_size, p=action_probs[0])
                # Take the action and observe the next observation and reward
                next_observation, reward, done, _ = self.env.step(action)

                critic_value = self.critic.model.predict(np.atleast_2d(observation))[0, 0]

                total_reward += reward

                # Append trajectories (Pseudocode line 3)
                trajectories.append(
                    obs=observation,
                    action=action,
                    critic_value=critic_value,
                    reward=reward
                )
                observation = next_observation

            # Compute rewards-to-go (= discounted_returns) (Pseudocode line 4)
            # Rewards-to-go: a weighted sum of all the rewards for all steps in the episode
            discounted_returns = compute_discounted_return(
                trajectories.rewards,
                discount_rate
            )

            #Compute advantage estimate (Pseudocode line 5)
            advantages = np.subtract(discounted_returns, trajectories.critic_values)

            # Calculate actor loss (Pseudocode line 6-7)
            actor_loss = self.actor.fit(
                np.atleast_2d(trajectories.observations),
                np.expand_dims(trajectories.actions, axis=1),
                np.expand_dims(advantages, axis=1),
            )
            # Calculate critic loss (Pseudocode line 8)
            critic_loss = self.critic.fit(
                np.atleast_2d(trajectories.observations),
                np.expand_dims(discounted_returns, axis=1),
            )

            # log training results
            self._log_history(episode, step, total_reward, actor_loss, critic_loss)

    def evaluate(self,
                 n_episodes: int
                 ) -> Dict[str, float]:
        """Evaluate the agent"""

        count_steps_history = []
        total_reward_history = []

        # Run the environment n_episodes time to check how many times it succeed
        for episode in range(n_episodes):
            observation = self.env.reset()
            step = 0
            total_reward = 0.0
            done = False

            while not done:
                # Step through the episode
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



    def act(self,
            observation: np.ndarray) -> np.ndarray:
        """Return an action given the input observation"""

        return np.random.choice(
            self.action_size,
            p=self.actor.model.predict(np.atleast_2d(observation))[0],
        )

    def _log_history(self,
                     episode: int,
                     count_steps: int,
                     total_reward: float,
                     actor_loss: float,
                     critic_loss: float
                     ) -> None:
        """ Log training results """

        self.history["episode"].append(episode)
        self.history["count_steps"].append(count_steps)
        self.history["total_reward"].append(total_reward)
        self.history["actor_loss"].append(actor_loss)
        self.history["critic_loss"].append(critic_loss)

        if self.is_in_wandb_session:
            # Log relevant graphs on wandb
            wandb.log(
                {
                    "episode": episode,
                    "count_steps": count_steps,
                    "total_reward": total_reward,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                },
            )
