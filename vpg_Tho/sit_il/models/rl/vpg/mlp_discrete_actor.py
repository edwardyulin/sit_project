# pylint: disable=no-name-in-module
from typing import List

import numpy as np

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class MLPDiscreteActor:
    """Actor network for discrete action spaces."""

    model: Sequential
    loss_func: SparseCategoricalCrossentropy
    optimizer: Adam

    def __init__(self) -> None:
        self.model_name = self.__class__.__name__

    def build(
        self,
        observation_size: int,
        action_size: int,
        hidden_units: List[int],
        learning_rate: float,
    ) -> None:
        """Build the network."""
        input_layer = Input(shape=(observation_size,))
        hidden_layers = [
            Dense(hidden_unit, activation="relu") for hidden_unit in hidden_units
        ]
        output_layer = Dense(action_size, activation="softmax")

        self.model = Sequential(
            layers=[
                input_layer,
                *hidden_layers,
                output_layer,
            ],
            name=self.model_name,
        )

        self.loss_func = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=learning_rate)

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> np.ndarray:
        """Train the actor model."""
        with tf.GradientTape() as tape:
            probs = self.model(observations, training=True)

            # estimate policy gradient (Pseudocode line 6)
            action_probs = tf.gather_nd(
                probs,
                indices=np.vstack([np.arange(probs.shape[0]), actions[:, 0]]).T,
            )
            action_log_probs = tf.math.log(action_probs)
            # minus sign changes gradient ascent into gradient descent
            actor_loss = -tf.math.reduce_mean(
                action_log_probs
                * tf.cast(tf.stop_gradient(advantages[:, 0]), tf.float32),
            )


        # compute policy update (Pseudocode line 7)
        grads = tape.gradient(actor_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return actor_loss
