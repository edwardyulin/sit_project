# imports
from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import numpy as np


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
              learning_rate: float
              ) -> None:
        """ Build the actor network"""

        input_layer = Input(shape=(observation_size,))
        # May need to change activation functions for hidden and output layers
        hidden_layers = [
            Dense(hidden_unit, activation="relu") for hidden_unit in hidden_units
        ]
        output_layer = Dense(output_size, activation="linear")
        self.model = Sequential(
            layers=[
                input_layer,
                *hidden_layers,
                output_layer
            ],
            name=self.model_name
        )

        self.loss_func = MeanSquaredError()
        self.optimizer = Adam(learning_rate=learning_rate)


    def fit(self,
            observations: np.ndarray,
            actions: np.ndarray,
            advantages: np.ndarray
            ) -> np.ndarray:
        """ Train the actor model """

        with tf.GradientTape() as tape:
            pred_action = self.model(observations, training=True)

            # Estimate policy gradient (Pseudocode line 6)
            # actor_loss is a float
            score_fn = tf.math.square(pred_action[:] - actions) # -> shape of (T,)
            actor_loss = -tf.math.reduce_mean(tf.reduce_sum(score_fn) * advantages) # -> shape of (,); a float

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
