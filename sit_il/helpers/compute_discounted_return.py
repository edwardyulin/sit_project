import numpy as np


def compute_discounted_return(
    rewards: np.ndarray,
    discount_rate: float,
) -> np.ndarray:
    r"""Compute discounted returns.

    Args:
        rewards (numpy.ndarray): Reward value at each time step in a 1-D array.
        discount_rate (float): Discount factor.

    Returns:
        numpy.ndarray: Discounted returns at each time step.
            .. math::
                G_t = R_{t+1} + \lambda R_{t+2} + \lambda^2 R_{t+3} + ...

    Examples:
        >>> compute_discounted_return([1, 1, 1], 0.99)
        array([2.9701, 1.99  , 1.    ])
    """
    discounted_rewards = []
    temp = 0.0
    for reward in rewards[::-1]:
        temp = discount_rate * temp + reward
        discounted_rewards.append(temp)
    return np.asarray(discounted_rewards[::-1])
