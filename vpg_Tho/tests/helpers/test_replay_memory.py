from collections import deque

import numpy as np
from numpy.testing import assert_allclose

from vpg_Tho.sit_il.helpers import Transition, ReplayMemory


def test_replay_memory_should_work() -> None:
    memory = ReplayMemory(capacity=10)
    transition = Transition(np.array([1.25, 2.25]), 1, np.array([]), -5.0, False)

    assert len(memory) == 0

    memory.add(transition)
    assert len(memory) == 1

    states, actions, next_states, rewards, dones = memory.sample(1)
    assert_allclose(states[0], transition.state)
    assert_allclose(actions[0], transition.action)
    assert_allclose(next_states[0], transition.next_state)
    assert_allclose(rewards[0], transition.reward)
    assert_allclose(dones[0], float(transition.done))

    memory.load(deque([transition, transition]))
    assert len(memory) == 2
