import gym
from mj_envs import hand_manipulation_suite

from sit_il.models.bc.bc import BCAgent
from sit_il.utils import random_seed

def main() -> None:
    env_name = "door-v0"
    env = gym.make(env_name)

    model = BCAgent(env)

    random_seed.set_numpy()
    random_seed.set_tensorflow()
    random_seed.set_gym(env)

    model.pipeline()


if __name__ == "__main__":
    main()
