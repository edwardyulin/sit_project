from pathlib import Path

import gym
from mj_envs import hand_manipulation_suite


from sit_il.models.bc.bc import BCAgent
from sit_il.utils import random_seed

def main() -> None:
    env_name = "door-v0"
    env = gym.make(env_name)

    save_network_to_file = Path(r"C:\Users\User\PycharmProjects\sit_project\sit_il\saved_model\bc_network")

    model = BCAgent(env)

    random_seed.set_numpy()
    random_seed.set_tensorflow()
    random_seed.set_gym(env)

    model.pipeline(save_network_to_file=save_network_to_file)


if __name__ == "__main__":
    main()
