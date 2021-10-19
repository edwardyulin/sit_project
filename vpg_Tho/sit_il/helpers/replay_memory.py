from typing import Deque, Tuple, NamedTuple

import random
from collections import deque

import numpy as np


class Transition(NamedTuple):
    """A single transition in the environment."""

    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


class ReplayMemory:
    """A cyclic buffer to store the experience data."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return the current size of the memory."""
        return len(self.memory)

    def add(self, transition: Transition) -> None:
        """Save a transition."""
        self.memory.append(transition)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a bach of transitions from memory."""
        transitions = random.sample(self.memory, batch_size)

        states, actions, next_states, rewards, dones = map(np.asarray, zip(*transitions))

        dones = dones.astype(np.float)

        return (states, actions, next_states, rewards, dones)

    def load(self, loaded_deque: Deque[Transition]) -> None:
        """Load the replay memory from a saved file."""
        self.memory = loaded_deque
