import safety_gymnasium
from DDPG.ddpg import DDPG
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
from markov_model import MarkovModel

import pickle

def genTransitions(yaml_file, dict_filename):
    env_id = 'SafetyCarGoal1-v0'
    env=safety_gymnasium.make(env_id, max_episode_steps=250)
    env.task.mechanism_conf.continue_goal=False
    ddpg_controller=DDPG(yaml_file)

    e=0
    E=1000

    if os.path.exists(dict_filename):
        print(dict_filename)
        with open(dict_filename, 'rb') as f:
            markov_model=pickle.load(f)
    else:
        markov_model=MarkovModel()

    state_start_ind=24
    while e<E:
        new_state=env.reset()
        obs=new_state[0][state_start_ind:]
        done=False
        truncated=False
        crashed=False
        t=0
        print("Episode {0} of {1}".format(e, E))

        dist_u=obs_new[16:32].max()
        dist_v=obs_new[32:].max()
        markov_model.initialiseState(dist_u, dist_v)

        while not done and not truncated and not crashed:
            act=ddpg_controller.act(obs)
            new_state=env.step(act)
            obs_new=new_state[0][state_start_ind:]

            reward=new_state[1]
            cost=new_state[2]
            done=new_state[3]
            truncated=new_state[4]

            v_state=0
            dist_u=obs_new[16:32].max()
            dist_v=obs_new[32:].max()

            if cost>0.9:
                v_state=2
            elif done:
                v_state=3
            elif cost>0:
                v_state=1
            markov_model.addStateTransition(v_state, dist_u, dist_v)
            
            obs=obs_new
            t+=1
        e+=1
    with open(dict_filename, 'wb') as f:
        pickle.dump(markov_model, f)

yaml_file=f'yaml_files/ddpg_1_1_1.yaml'
markov_model_filename=f'markov_models/mm_1_1_1.pickle'
genTransitions(yaml_file, markov_model_filename)