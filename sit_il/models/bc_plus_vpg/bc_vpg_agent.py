from typing import List, Optional, Dict, Union, Tuple, Any
import gym
import keras
import wandb
from pathlib import Path
import numpy as np
import pandas as pd

import tempfile

from dataclasses import field, dataclass

import tensorflow as tf
from tensorflow.keras.utils import plot_model

from sit_il.models.bc.bc import BCAgent
from sit_il.models.rl.npg.vpg.mlp_critic import MLPCritic
from sit_il.models.rl.npg.vpg.mlp_continuous_actor import MLPContinuousActor

from sit_il.helpers import compute_discounted_return


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@dataclass
class Trajectories:
    """ Storing trajectories """

    # observation is represented as an array of [car_position, car_velocity]
    observations: List[np.ndarray] = field(default_factory=list)
    # action is represented as a float in the range of [-1.0, 1.0]
    actions: List[float] = field(default_factory=list)
    # critic value from the output of the critic network
    critic_values: List[float] = field(default_factory=list)
    # reward of 100 is awarded if the agent reached the flag (position = 0.45) on top of the mountain.
    # reward is decrease based on amount of energy consumed each step.
    rewards: List[float] = field(default_factory=list)

    def append(self,
               observation: np.ndarray,
               action: float,
               critic_value: float,
               reward: float
               ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.critic_values.append(critic_value)
        self.rewards.append(reward)



class BC_VPG_Agent:

    actor: MLPContinuousActor
    critic: MLPCritic # TODO: change this to a new critic network

    def __init__(self,
                 env: gym.Env):
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
            "critic_loss": []
        }

    def pipeline(self,
                 actor_hidden_units: List[int],
                 critic_hidden_units: List[int],
                 actor_learning_rate: float,
                 critic_learning_rate: float,
                 n_train_episodes: int,
                 max_vpg_episode_length: int,
                 discount_rate: float,
                 n_test_episodes: int,
                 print_summary: bool,
                 plot_actor_network_to_file: Optional[Path],
                 plot_critic_network_to_file: Optional[Path],
                 save_actor_network_to_file: Optional[Path],
                 save_critic_network_to_file: Optional[Path],
                 load_actor_network_from_file: Optional[Path],
                 load_critic_network_from_file: Optional[Path],
                 load_bc_network: Optional[Path]):

        # Initialize wandb
        with wandb.init(
            project="sit_bc_vpg_door",
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
                "max_episode_length": max_vpg_episode_length,
                "discount_rate": discount_rate,
                "n_test_episodes": n_test_episodes,
            },
        ):
            self.is_in_wandb_session = True
            self.config = wandb.config

            if load_actor_network_from_file and load_critic_network_from_file:
                self.load(load_actor_network_from_file, load_critic_network_from_file)
            else:
                # Train with BC and return the neural network
                # Construct actor and critic networks with weights from the BC neural network
                self.build(
                    observation_size=self.config["observation_size"],
                    action_size=self.config["action_size"],
                    actor_hidden_units=self.config["actor_hidden_units"],
                    critic_hidden_units=self.config["critic_hidden_units"],
                    actor_learning_rate=self.config["actor_learning_rate"],
                    critic_learning_rate=self.config["critic_learning_rate"],
                    load_bc_network=load_bc_network
                )

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

                print("Actor network:")
                print(self.actor.model.summary())
                print()
                print("=======================")
                print("Critic network:")
                print(self.critic.model.summary())
                # Visual model architecture from wandb

                # Train the agent with VPG (while logging wandb)
                self.fit(
                    n_episodes=self.config["n_train_episodes"],
                    max_episode_length=self.config["max_episode_length"],
                    discount_rate=self.config["discount_rate"],
                    actor_learning_rate=self.config["actor_learning_rate"]
                )

                # save after training
                if save_actor_network_to_file and save_critic_network_to_file:
                    self.save(save_actor_network_to_file, save_critic_network_to_file)

            # Evaluate the model
            # evaluate the agent and log results onto wandb
            results = self.evaluate(n_episodes=n_test_episodes)
            # An episode succeed if "what"
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
        self.is_in_wandb_session = False  # finished logging on wandb










    def build(self,
              observation_size: int,
              action_size: int,
              actor_hidden_units: List[int],
              critic_hidden_units: List[int],
              actor_learning_rate: float,
              critic_learning_rate: float,
              load_bc_network: Optional[Path]
              ) -> None:
        self.actor = MLPContinuousActor()
        self.actor.build(
            observation_size=observation_size, #input of the network
            output_size=action_size, # output of one float value to represent the action (instead of prob. of actions seen in discrete)
            hidden_units=actor_hidden_units,
            learning_rate=actor_learning_rate,
            load_bc_network=load_bc_network
        )

        self.critic = MLPCritic()
        self.critic.build(
            obs_size=observation_size,
            hidden_units=critic_hidden_units,
            learning_rate=critic_learning_rate,
        )


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
            discount_rate: int,
            actor_learning_rate: int
            ) -> None:
        """ Train the agent"""

        for episode in range(n_episodes):
            observation = self.env.reset()
            step = 0
            total_reward = 0.0
            done = False

            print("Initial State: ", observation)

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
                action = self.actor.model.predict(np.atleast_2d(np.squeeze(observation)))
                # only take the first argument, don't need batch_size
                action = action[0]
                #print(action)
                next_observation, reward, done, _ = self.env.step(action)

                critic_value = self.critic.model.predict(np.atleast_2d(np.squeeze(observation)))[0, 0]

                total_reward += reward

                trajectories.append(
                    observation=np.squeeze(observation),
                    action=action,
                    critic_value=critic_value,
                    reward=reward
                )

                observation = next_observation

                # Compute rewards-to-go (= discounted_returns) (Pseudocode line 4)
                # Rewards-to-go: a weighted sum of all the rewards for all steps in the episode
                discounted_returns = compute_discounted_return(
                    rewards=trajectories.rewards,
                    discount_rate=discount_rate
                )

                # Compute advantage estimate (Pseudocode line 5)
                advantages = np.subtract(discounted_returns, trajectories.critic_values)

                # Calculate actor loss (Pseudocode line 6-7)
                actor_loss = self.actor.fit(
                    observations=np.atleast_2d(trajectories.observations),
                    actions=trajectories.actions,
                    advantages=advantages,
                    learning_rate=actor_learning_rate,
                    step=step
                )

                # Calculate critic loss (Pseudocode line 8)
                critic_loss = self.critic.fit(
                    obs=np.atleast_2d(trajectories.observations),
                    discounted_returns=np.expand_dims(discounted_returns, axis=1)
                )

                # log training results
            self._log_history(episode, step, total_reward, actor_loss, critic_loss)

    def evaluate(self,
                 n_episodes: int
                 ) -> Dict[str, float]:
        """Evaluate the agent"""

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

    def act(self,
            observation: np.ndarray
            ) -> np.ndarray:
        """Return an action given the input observation"""

        return self.actor.model.predict(np.atleast_2d(np.squeeze(observation)))[0]


    def _log_history(self,
                     episode: int,
                     count_steps: int,
                     total_reward: float,
                     actor_loss: float,
                     critic_loss: float
                     ) -> None:
        """Log training restuls """

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

    def save(self, actor_to_file: Path, critic_to_file: Path)->None:
        """Save the actor and critic to file."""
        self.actor.save(actor_to_file)
        self.critic.save(critic_to_file)

    def load(self, actor_from_file: Path, critic_from_file: Path) ->None:
        """Load the actor and critic."""
        self.actor = tf.keras.models.load_model(actor_from_file)
        self.critic = tf.keras.models.load_model(critic_from_file)
