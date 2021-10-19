# used for both discrete and continuous VPG
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense

from typing import List

import numpy as np


class MLPCritic:
    """Critic Network"""

    # define model, loss_func, optimizer
    model: Sequential
    loss_func: MeanSquaredError
    optimizer: Adam

    def __init__(self) -> None:
        self.model_name = self.__class__.__name__

    def build(self,
              obs_size: int, # critic network has obs as input layer
              hidden_units: List[int],
              learning_rate: float) -> None:

        """Build the critic network"""

        input_layer = Input(shape=(obs_size,))
        # range for relu -> [0, inf)
        hidden_layers = [
            Dense(hidden_unit, activation="relu") for hidden_unit in hidden_units
        ]
        output_layer = Dense(1, activation="linear") # 1 output of estimate of total rewards in the future

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
            obs: np.ndarray,
            discounted_returns: np.ndarray
    ) -> np.ndarray:
        """Train the critic model and return critic_loss"""

        with tf.GradientTape() as tape:
            # predicted value is calcualted by subbing in
            pred_value = self.model(obs, training=True)
            # fit value function by regression on mean-squared error (Pseudocode line 8)
            # minimizing the difference between the predicted value and the actual value
            critic_loss = self.loss_func(pred_value, tf.stop_gradient(discounted_returns))

        gradients = tape.gradient(critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return critic_loss

    def save(self, to_file: Path)->None:
        """Save the model architecture and weight values to file."""
        self.model.save(to_file)

    def load(self, from_file: Path) ->None:
        """Load the saved model architecture and weight values."""
        self.model = tf.keras.models.load_model(from_file)
