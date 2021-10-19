# import necessary libraries
import pickle
import numpy as np
from mjrl.utils.gym_env import GymEnv

def demo_playback(env, demo):
    e = GymEnv(env)
    e.reset()
    for path in demo:
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        for t in range(actions.shape[0]):
            e.step(actions[t])
            e.env.mj_render()



def save_demo(da, dr, do, dis):
    np.save('demo_actions.npy', da)
    np.save('demo_rewards.npy', dr)
    np.save('demo_obs.npy', do)
    np.save('demo_init_state.npy', dis)



def main():
    # get expert demonstrations
    demo_pickle = open("door-v0_demos.pickle", "rb")
    objects = []
    env_name = "door-v0"
    #demo_playback(env_name, pickle.load(demo_pickle))
    objects.append(pickle.load(demo_pickle))

    # print(objects)

    demo_dict = objects[0]
    demo_actions = []
    demo_rewards = []
    demo_obs = []
    demo_init_state = []

    demo_actions.append(demo_dict[0].get('actions'))
    demo_rewards.append(demo_dict[0].get('rewards'))
    demo_obs.append(demo_dict[0].get('observations'))

    """ THIS IS FOR ALL 25 DEMO, UN-COMMENT LATER
    for i in range(len(demo_dict)):
        demo_actions.append(demo_dict[i].get('actions'))
        demo_rewards.append(demo_dict[i].get('rewards'))
        demo_obs.append(demo_dict[i].get('observations'))
        demo_init_state.append(demo_dict[i].get('init_state_dict'))
        print(len(demo_actions), len(demo_obs))
    """

    print(demo_actions)
    print(len(demo_actions[0]))


    demo_pickle.close()
    save_demo(demo_actions, demo_rewards, demo_obs, demo_init_state)

    """

    # These are only for one demo
    demo_dict = objects[0][0]
    demo_actions = demo_dict.get('actions')
    demo_rewards = demo_dict.get('rewards')
    demo_obs = demo_dict.get('observations')
    demo_init_state = demo_dict.get('init_state_dict')
    print(demo_actions)
    print(len(demo_actions), len(demo_obs))
    demo_pickle.close()
    save_demo(demo_actions, demo_rewards, demo_obs, demo_init_state)
    """



if __name__ == "__main__":
    main()
