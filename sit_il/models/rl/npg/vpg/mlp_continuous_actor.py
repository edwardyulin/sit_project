# imports
from pathlib import Path
from typing import List
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
              learning_rate: float
              ) -> None:
        """ Build the actor network"""

        # set the actor with pre-training using behavioural cloning
        self.model = self.bc()



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
            weighting_fn = lamb_0 * lamb_1**step * advantages
            actor_loss = -tf.math.reduce_mean(tf.reduce_sum(score_fn) * advantages) \
                         - tf.math.reduce_mean(tf.reduce_sum(score_fn) * weighting_fn)
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




    """"""""""""""""""" Behavioural Cloning Begins """""""""""""""""""""""""

    def normalize_data(self,
                       data: np.ndarray):
        min_value = -2.0
        max_value = 2.0
        return (data - min_value) / (max_value - min_value)

    def reform(self,
               npy: np.ndarray):
        new = npy[0]
        for i in range(1, len(npy)):
            new = np.concatenate((new, npy[i]), axis=0)
        return new

    def build_network(self):
        model = tfk.Sequential([
            # each element in the state_data has 39 elements
            # each element in eh action_data has 28 elements
            tfk.layers.Dense(input_shape=(39,), units=32, activation='relu', name='input_layer'),
            #tfk.layers.Dense(units=32, activation='relu', name='hidden_layer'),
            tfk.layers.Dense(units=16, activation='relu', name='hidden_layer1'),
            tfk.layers.Dense(units=16, activation='relu', name='hidden_layer2'),
            tfk.layers.Dense(units=28, activation='linear', name='output_layer')
        ])

        print(model.summary())
        return model

    def train_model(self,
                    model: tfk.Sequential,
                    action_data: np.ndarray,
                    state_data: np.ndarray):
        print("Training starts")
        loss_axis = []
        acc_axis = []
        model.compile(
            # TODO: play around different lr, loss_fn, batch_size, epochs, layer num to find optimal training algo
            # gradient descent
            optimizer=tfk.optimizers.Adam(learning_rate=1e-2),  # change learning rate for different accuracy/loss
            # loss function
            loss=tfk.losses.mean_squared_error,
            # use mean_absolute_error if have outliers
        )
        model.fit(
            # training data
            state_data,
            # label data
            action_data,
            # size of data in one training loop
            batch_size=32,  # usually 32, 64, 128, 256...
            epochs=100,  # num of training iterations
            shuffle=True,
            # increase epochs with the complexity of the problem
        )

        history = model.history.history
        print(type(history["loss"]))
        new_loss_axis = loss_axis + history["loss"]
        loss_axis = new_loss_axis

        print(loss_axis)
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        # loss function measures how far an estimated value is from its true value
        axis[0].plot(loss_axis, label="loss")
        # don't need to show plot for now
        axis[0].legend()
        #plt.show()

    def bc(self):
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        action_npy = np.load(r'C:\Users\User\PycharmProjects\sit_project\sit_il\models\bc_plus_vpg\demo_actions.npy')
        obs_npy = np.load(r'C:\Users\User\PycharmProjects\sit_project\sit_il\models\bc_plus_vpg\demo_obs.npy')
        action_data = self.normalize_data(self.reform(action_npy))
        print("Demo actions shape:", action_data.shape)
        state_data = self.normalize_data(self.reform(obs_npy))
        print("Demo obs shape:", state_data.shape)
        print("")
        env = gym.make('door-v0')
        print(action_data)

        model = self.build_network()

        self.train_model(model=model,
                        action_data=action_data,
                        state_data=state_data)

        return model

    """"""""""""""""""" Behavioural Cloning Ends """""""""""""""""""""""""
