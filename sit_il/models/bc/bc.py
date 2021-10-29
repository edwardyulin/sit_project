# import libraries
import os
from pathlib import Path
from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as tfk

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class BCAgent:
    """Behavioural Cloning"""

    def __init__(self,
                 env: gym.Env) -> None:
        self.model_name = self.__class__.__name__
        self.env = env

    # construct network
    def build_network(self):
        model = tfk.Sequential([
            # each element in the state_data has 39 elements
            # each element in eh action_data has 28 elements
            tfk.layers.Dense(input_shape=(39,), units=256, activation='relu', name='input_layer'),
            tfk.layers.Dense(units=32, activation='relu', name='hidden_layer'),
            tfk.layers.Dense(units=16, activation='relu', name='hidden_layer1'),
            tfk.layers.Dense(units=16, activation='relu', name='hidden_layer2'),
            tfk.layers.Dense(units=28, activation='linear', name='output_layer')
        ])

        print(model.summary())
        return model

    # train model
    def train_model(self,
                    model,
                    action_data,
                    state_data):
        print("model type: ", type(model))
        print("action_data type: ", type(action_data))
        print("state_data type: ", type(state_data))
        print("Training starts")
        loss_axis = []
        acc_axis = []
        model.compile(
        # TODO: play around different lr, loss_fn, batch_size, epochs, layer num to find optimal training algo
        # gradient descent
            optimizer=tfk.optimizers.Adam(learning_rate=1e-2), # change learning rate for different accuracy/loss
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
            batch_size=32, #usually 32, 64, 128, 256...
            epochs=100, # num of training iterations
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
        axis[0].legend()
        plt.show()



    def reform(self,
               npy):
        new = npy[0]
        for i in range(1, len(npy)):
            new = np.concatenate((new, npy[i]), axis=0)
        print(new)
        return new

    def evaluate_model(self,
                       model):
        num_episodes = 10
        accuracy = 0
        env = gym.make("door-v0")

        for episode_idx in range(num_episodes):
            print(f"Evaluating episode {episode_idx}")
            done = False
            prev_obs = env.reset()
            count_step = 0

            while not done:
                action = model.predict(prev_obs.reshape(1, 39))
                #action = self.denormalize_data(action)
                obs, _, done, _ = env.step(action[0])
                count_step += 1
                prev_obs = obs

            print(f"done: {done}, count_step: {count_step}")
            if done and count_step < env._max_episode_steps:
                accuracy += 1

        return accuracy/num_episodes

    def save(self, to_file: Path)->None:
        self.model.save(to_file, save_format="tf")

    def pipeline(self,
                 save_network_to_file: Optional[Path]):
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        action_npy = np.load(r"C:\Users\User\PycharmProjects\sit_project\sit_il\models\bc\demo_actions.npy")
        obs_npy = np.load(r"C:\Users\User\PycharmProjects\sit_project\sit_il\models\bc\demo_obs.npy")
        action_data = self.reform(action_npy)
        print("Demo actions shape:", action_data.shape)
        state_data = self.reform(obs_npy)
        print("Demo obs shape:", state_data.shape)
        print("")
        #rewards_data = np.load('demo_rewards.npy')
        #init_state_data = np.load('demo_init_state.npy')
        env = gym.make('door-v0')
        print(action_data)

        # build neural network
        self.model = self.build_network()

        self.train_model(self.model, action_data, state_data)

        print("==========")
        #print("Accuracy: ", self.evaluate_model(self.model))

        # save BC model
        self.save(save_network_to_file)



        # visualize the trained model
        done = False
        prev_obs = env.reset()
        while not done:
            # Render the environment
            env.render()
            # Get expert action
            action = self.model.predict(prev_obs.reshape(1, 39))
            """
            # shape of the each action is (1, 28), 1 row and 28 columns (1 array)
            print(action.shape)
            print(env.action_space)
            # to match the shape of env.action_space with action, we must only get action[0]
            print(action[0].shape)
            print(env.action_space.shape)
            """

            print(action[0])
            # Performing an action in an environment to change state
            obs, reward, done, info = env.step(action[0])
            prev_obs =obs
            env.mj_render()
