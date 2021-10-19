import gym

from sit_il.utils import random_seed
from sit_il.models.rl import VPGDiscreteAgent

def main() -> None:
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    max_episode_length = 100 # each episode has at most 100 steps
    env._max_episode_steps = max_episode_length;

    model = VPGDiscreteAgent(env) # call vpg_discrete_agent.py

    # set random seed (make each run of the project deterministic, results always the same)
    random_seed.set_numpy()
    random_seed.set_tensorflow()
    random_seed.set_gym(env)

    print("Environment name: ", env_name)
    print("Observation_space: ", env.observation_space)
    print("Action space: ", env.action_space)
    print("==============================")
    print()

    # call for the pipeline of VPG with the model (the networks)
    model.pipeline(
        # the parameters can be changed
        actor_hidden_units=[32, 16, 16],
        critic_hidden_units=[32, 16, 16],
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        n_train_episodes=200,
        max_episode_length=max_episode_length,
        discount_rate=0.99,
        n_test_episodes=10,
        print_summary=True,
        plot_actor_network_to_file=None,
        plot_critic_network_to_file=None,
    )

if __name__ == "__main__":
    main()
