# imports
from pathlib import Path
from typing import List, Optional
import gym

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import keras as tfk
import matplotlib.pyplot as plt


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import numpy as np

from sit_il.models.bc.bc import BCAgent


class MLPContinuousActor:
    """Actor network for discrete action spaces."""

    model: Sequential
    loss_func: MeanSquaredError
    optimizer: Adam


    def __init__(self) -> None:

        self.model_name = self.__class__.__name__

    def build(self,
              observation_size: int,
              output_size: int,
              hidden_units: List[int],
              learning_rate: float,
              load_bc_network: Optional[Path]
              ) -> None:
        """ Build the actor network"""

        # set the actor with pre-training using behavioural cloning
        self.load(load_bc_network)



    def fit(self,
            observations: np.ndarray,
            actions: np.ndarray,
            advantages: np.ndarray,
            learning_rate: int,
            step: int
            ) -> np.ndarray:
        """ Train the actor model """

        self.loss_func = MeanSquaredError()
        self.optimizer = Adam(learning_rate=learning_rate)

        with tf.GradientTape() as tape:
            pred_action = self.model(observations, training=True)

            # Estimate policy gradient (Pseudocode line 6)
            # actor_loss is a float
            lamb_0 = 0.1 # from Rajeswaran et al. (2018)
            lamb_1 = 0.95 # from Rajeswaran et al. (2018)
            score_fn = tf.math.square(pred_action[:] - actions) # -> shape of (T,)
            #TODO: run the code without weighting_fn
            #weighting_fn = lamb_0 * lamb_1**step * advantages
            actor_loss = -tf.math.reduce_mean(tf.reduce_sum(score_fn) * advantages)
                         #- tf.math.reduce_mean(tf.reduce_sum(score_fn) * weighting_fn)
            # -> shape of (,); a float

        # Compute policy update (Pseudocode line 7)
        # each node at each layer at each time step has a grad
        # grads and FIM is a list of tensor, and tensors have the shape of: (39, 32), (32,), (32,16), (16,), (16,16)
        grads = tape.gradient(actor_loss, self.model.trainable_variables)
        # this returns an array of 28 elements, representing the action at each time step
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return actor_loss
    """
            transposed_score_fn = tf.transpose(score_fn) # -> shape of (,T)
            log_likelihood = -tf.math.reduce_mean(tf.math.multiply(score_fn, transposed_score_fn))
    """

    """
        FIM = tape.gradient(log_likelihood, self.model.trainable_variables)
        FIM_inv = []
        FIM_and_g = []

        for t in range(5):
            FIM_inv.append(tf.divide(1.0, FIM[t]))
            print(FIM_inv)
            FIM_and_g.append(tf.math.multiply(grads[t], FIM_inv[t]))

    """


    def save(self, to_file: Path)->None:
        """Save the model architecture and weight values to file."""
        self.model.save(to_file)

    def load(self, from_file: Path) ->None:
        """Load the saved model architecture and weight values."""
        self.model = tf.keras.models.load_model(from_file)

