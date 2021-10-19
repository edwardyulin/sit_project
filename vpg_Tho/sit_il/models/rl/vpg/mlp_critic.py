# pylint: disable=no-name-in-module

from typing import List

import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


class MLPCritic:
    """Critic network."""

    model: Sequential
    loss_func: MeanSquaredError
    optimizer: Adam

    def __init__(self) -> None:
        self.model_name = self.__class__.__name__

    def build(
        self,
        observation_size: int,
        hidden_units: List[int],
        learning_rate: float,
    ) -> None:
        """Build the network."""
        input_layer = Input(shape=(observation_size,))
        hidden_layers = [
            Dense(hidden_unit, activation="relu") for hidden_unit in hidden_units
        ]
        output_layer = Dense(1, activation="linear")

        self.model = Sequential(
            layers=[
                input_layer,
                *hidden_layers,
                output_layer,
            ],
            name=self.model_name,
        )
        self.loss_func = MeanSquaredError()
        self.optimizer = Adam(learning_rate=learning_rate)

    def fit(
        self,
        observations: np.ndarray,
        discounted_returns: np.ndarray,
    ) -> np.ndarray:
        """Train the critic model."""
        with tf.GradientTape() as tape:
            pred_value = self.model(observations, training=True)
            #fit value function by regression on mean-squared error (Pseudocode line 8)
            critic_loss = self.loss_func(pred_value, tf.stop_gradient(discounted_returns))

        grads = tape.gradient(critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return critic_loss
