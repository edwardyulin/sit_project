# import libraries

from tensorflow import keras as tfk
import numpy as np
import gym
from spinningup.spinup.algos.pytorch.vpg import vpg as spinup



"""

NPG Summary
    INPUT: observations
    Procedure
    1. Generate an episode to keep track of observations, actions and rewards in the agent's memory.
    2. At the end of each episode, go back to the beginning of the episode to compute the discounted
    future return at each time-step.
    3. Use these returns as weights and the actions the agent took as label to perform back-propagation
    and updates the weights of the deep neural network.
    4. Reset the memory and discard all previous experience (only the new weights are kept) then repeat.
    OUTPUT: actions / optimal policies

"""


def normalize_data(data):
    min_value = -2.0
    max_value = 2.0
    return (data - min_value)/(max_value - min_value)


def denormalize_data(data):
    min_value = -2.0
    max_value = 2.0
    return data * (max_value - min_value) + min_value


def reform(npy):
    new = npy[0]
    for i in range(1, len(npy)):
        new = np.concatenate((new, npy[i]), axis=0)
    print(new)
    return new


def build_network():
    # construct network with 39 input neurons and 28 output neurons

    network = tfk.Sequential([
       tfk.layers.Dense(input_shape=(39,), units=256, activation='relu', name='input_layer'),
       tfk.layers.Dense(units=256, activation='relu', name='hidden_layer'),
       tfk.layers.Dense(units=256, activation='relu', name='hidden_layer1'),
       tfk.layers.Dense(units=128, activation='relu', name='hidden_layer2'),
       # tfk.layers.Dense(units=35, activation='relu', name='hidden_layer3'),
       # tfk.layers.Dense(units=35, activation='relu', name='hidden_layer4'),
       tfk.layers.Dense(units=28, activation='linear', name='output_layer')
    ])

    print(network.summary())
    return network


def npg_training(env, model, obs_data, action_data, reward_data):
    traj = form_trajectories(obs_data, action_data, reward_data)

    #TODO: Add back the for loop for episodes (for now, only 1 demo is considered)
    #for i in range(25): # 25 = number of total sets of demonstrations
        # get advantage estimate
    advantages = generalized_advantage_estimate(reward_data)
    print(advantages)

    # compute vanilla policy gradient
    vpg = compute_vanilla_policy_gradient(env, model, obs_data, action_data, advantages) # unsure about input

    # get Fisher Information Matrix
    # fim = get_FIM() # unsure about input

    # compute NPG update (the actual training)


def form_trajectories(obs, act, rew):
    # form trajectories for all triplets (obs, act, rew) for all time-steps within the demonstrations

    traj = []
    assert len(obs) == len(act) == len(rew), "Length of obs, act, rew lists different"
    for i in range(len(obs)): # will loop 6729 (demo # * timestep #) times
        current = [obs[i], act[i], rew[1]]
        traj.append(current)

    return traj


def generalized_advantage_estimate(rew):
    # generate a list of advantages for all time-steps within the demonstration

    gamma = 1 # exponential mean discount
    lamb = 1 # trajectory discount
    value_fn_list = get_value_functions(rew)
    advantage = np.zeros(len(rew))
    done = np.zeros(len(rew))
    print(rew)

    for t in reversed(range(len(rew) - 1)):
        delta = rew[t] + (gamma * value_fn_list[t+1] * done[t]) - value_fn_list[t]
        print("delta", delta)
        advantage[t] = delta + (gamma * lamb * advantage[t + 1])
        print("adv", advantage[t])

    return advantage

def get_value_functions(rew):
    reward_list = []

    for i in range(len(rew)):
        if len(reward_list) == 0:
            reward_list.append(rew[-1])
        else:
            reward_list.append(reward_list[-1] + rew[len(rew) - i - 1])

    reward_list.reverse()

    return reward_list


def compute_vanilla_policy_gradient(env, model, obs, action, adv):
    # compute vanilla policy gradient

    # call to spinup.vpg_Tho also includes finding advantage using GAE
    vpg = spinup.vpg(env)

    print(vpg)
    return vpg


def get_FIM():
    # pre-conditions the gradient with the (inverse of) Fisher Information Matrix
    return



def main():
    # get observations, actions and rewards from demo_obs_npy, demo_actions.py and demo_rewards.npy
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    action_data = normalize_data(reform(np.load('demo_actions.npy')))
    print("Demo actions shape:", action_data.shape)
    state_data = normalize_data(reform(np.load('demo_obs.npy')))
    print("Demo obs shape:", state_data.shape)
    print("")
    reward_data = normalize_data(reform(np.load('demo_rewards.npy')))
    print("Demo reward shape:", reward_data.shape)
    # init_state_data = np.load('demo_init_state.npy')

    # get gym environment door-v0
    env = gym.make('door-v0')
    # print(action_data)

    network = build_network()

    npg_training(env, network, state_data, action_data, reward_data)

    # visualize the trained model

if __name__ == '__main__':
    main()
