import safety_gymnasium
from ddpg import DDPG
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
from markov_model import MarkovModel

import pickle

def genTransitions(yaml_file, dict_filename):
    env_id = 'SafetyCarGoal1-v0'
    env=safety_gymnasium.make(env_id, max_steps=250)
    env.task.mechanism_conf.continue_goal=False
    env.task.num_steps=1000
    ddpg_controller=DDPG(yaml_file)

    e=0
    E=2000
    TT=1000

    if os.path.exists(dict_filename):
        print(dict_filename)
        with open(dict_filename, 'rb') as f:
            markov_model=pickle.load(f)
    else:
        markov_model=MarkovModel()

    while e<E:
        new_state=env.reset()
        obs=new_state[0][24:]
        done=False
        truncated=False
        crashed=False
        t=0
        reward_max=0
        costs_max=0
        print("Episode {0} of {1}".format(e, E))

        markov_model.initialiseState(obs[24:].max())

        while not done and not truncated and not crashed:
            if (t+1)%200==0:
                print('    - '+str(t+1))

            act=a2c_controller.act(obs)
            act=act.clip(-2,2)
            new_state=env.step(act)
            obs=new_state[0][-16*3:]        
            reward=new_state[1]
            cost=new_state[2]
            done=new_state[3]
            truncated=new_state[4]

            if cost>0:
                crashed=True
                markov_model.crashedStateTransition()
                print("    - Crashed at t= "+str(t+1))
            elif done:
                markov_model.goalStateTransition()
                print('    - Success at t='+str(t+1))
            else:
                markov_model.addStateTransition(obs[-16*2:].max())
            t+=1
        e+=1

    for i in markov_model.transition_matrix_dict:
        print(markov_model.transition_matrix_dict[i])
    with open(dict_filename, 'wb') as f:
        pickle.dump(markov_model, f)


yaml_files=os.listdir('yaml_files/')

for yaml_file in yaml_files:
    gain=yaml_file[:-5].split('_')[1:]
    markov_model_filename=f'markov_models/mm_{gain[0]}_{gain[1]}_{gain[2]}.pickle'
    genTransitions(yaml_file, markov_model_filename)

