# pylint: disable=import-outside-toplevel,import-error

from typing import Any

import os
import random

DEFAULT_SEED = 3407


def set_numpy(seed: int = DEFAULT_SEED) -> None:
    """Seed the random and numpy packages to ensure deterministic behavior.

    Args:
        seed (int): Seed to set.

    Examples:
        >>> import random
        >>> import numpy as np
        >>> set_numpy(1234)
        >>> a = random.random()
        >>> b = np.random.rand()
        >>> set_numpy(1234)
        >>> c = random.random()
        >>> d = np.random.rand()
        >>> np.allclose([a, b], [c, d])
        True
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def set_tensorflow(seed: int = DEFAULT_SEED) -> None:
    """Seed the tensorflow package to ensure deterministic behavior.

    Args:
        seed (int): Seed to set.

    Examples:
        >>> import tensorflow as tf
        >>> set_tensorflow(1234)
        >>> a = tf.random.uniform([1])
        >>> set_tensorflow(1234)
        >>> b = tf.random.uniform([1])
        >>> (a == b).numpy()
        array([ True])
    """
    import tensorflow as tf

    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.random.set_seed(seed)


def set_torch(seed: int = DEFAULT_SEED) -> None:
    """Seed the torch package to ensure deterministic behavior.

    Args:
        seed (int): Seed to set.

    Examples:
        >>> import torch
        >>> set_torch(1234)
        >>> a = torch.rand(1)
        >>> set_torch(1234)
        >>> b = torch.rand(1)
        >>> (a == b).numpy()
        array([ True])
    """
    import torch

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore


def set_gym(env: Any, seed: int = DEFAULT_SEED) -> None:
    """Seed the gym package to ensure deterministic behavior.

    Args:
        env (Any): gym environment.
        seed (int): Seed to set.

    Examples:
        >>> import numpy as np
        >>> import gym
        >>> env = gym.make("CartPole-v0")
        >>> set_gym(env, 1234)
        >>> obs_1 = env.reset()
        >>> action_1 = env.action_space.sample()
        >>> set_gym(env, 1234)
        >>> obs_2 = env.reset()
        >>> action_2 = env.action_space.sample()
        >>> np.allclose(obs_1, obs_2)
        True
        >>> action_1 == action_2
        True
    """
    env.seed(seed)
    env.action_space.np_random.seed(seed)
