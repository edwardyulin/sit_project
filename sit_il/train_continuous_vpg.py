from pathlib import Path

import gym

from sit_il.utils import random_seed
from sit_il.models.rl import VPGContinuousAgent

from mj_envs import hand_manipulation_suite

try:
    from console_thrift import KeyboardInterruptException as KeyboardInterrupt
except ImportError:
    pass

"""
MountainCarContinuous-v0 Source Code
https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

def main() -> None:
    env_name = "door-v0"
    env = gym.make(env_name)

    max_episode_length = 100 # each episode has at most 100 steps
    env._max_episode_steps = max_episode_length

    model = VPGContinuousAgent(env) # call vpg_continuous_agent.py

    # set random seed (make each run of the project deterministic, results always the same)
    random_seed.set_numpy()
    random_seed.set_tensorflow()
    random_seed.set_gym(env)

    print("Environment name: ", env_name)
    print("Observation_space: ", env.observation_space)
    print("Action space: ", env.action_space)
    print("==============================")
    print()

    save_actor_to_file = Path(r"C:\Users\User\PycharmProjects\sit_project\sit_il\saved_model\vpg_actor.h5")
    save_critic_to_file = Path(r"C:\Users\User\PycharmProjects\sit_project\sit_il\saved_model\vpg_critic.h5")


    load_actor_from_file = None #Path(r"C:\Users\User\PycharmProjects\sit_project\sit_il\saved_model\vpg_actor.h5")
    load_critic_from_file = None  #Path(r"C:\Users\User\PycharmProjects\sit_project\sit_il\saved_model\vpg_critic.h5")

    try:
        # call for the pipeline of VPG with the model (the networks)
        model.pipeline(
            # the parameters can be changed
            actor_hidden_units=[32, 16, 16],
            critic_hidden_units=[32, 16, 16],
            actor_learning_rate=0.001,
            critic_learning_rate=0.001,
            n_train_episodes=100,
            max_episode_length=max_episode_length,
            discount_rate=0.99,
            n_test_episodes=10,
            print_summary=True,
            plot_actor_network_to_file=None,
            plot_critic_network_to_file=None,
            save_actor_network_to_file=save_actor_to_file,
            save_critic_network_to_file=save_critic_to_file,
            load_actor_network_from_file=load_actor_from_file,
            load_critic_network_from_file=load_critic_from_file
        )
    except KeyboardInterrupt:
        print("Saving the agent...")
        model.save(save_actor_to_file, save_critic_to_file)
        print("Saved!")



if __name__ == "__main__":
    main()
