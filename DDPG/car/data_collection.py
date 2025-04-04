import safety_gymnasium
from A2C import A2C
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
from markov_model import MarkovModel

import pickle

def genTransitions(actor_model, critic_model, dict_filename):
    env_id = 'SafetyCarGoal1-v0'

#    render_choice=input('Render?')

#    while not render_choice.isnumeric() or not (int(render_choice)==0 or int(render_choice)==1):
#        render_choice=input('Render?')
#        if not(render_choice.isnumeric()):
#            print("Not an int")
#        else:
#            print(int(render_choice))
#        if not(int(render_choice)==0) and not(int(render_choice)==1):
#            print("Enter either 0 or 1")
#    if int(render_choice):
#        env=safety_gymnasium.make(env_id, render_mode='human')
#    else:
#        env=safety_gymnasium.make(env_id)#, render_mode='human')

    env=safety_gymnasium.make(env_id)
    env.task.mechanism_conf.continue_goal=False
    env.task.num_steps=1000

    ##path=os.environ["MODEL_PATH"]
    actor_filename=actor_model
    critic_filename=critic_model
#    start_ind=0
#    if os.path.exists(path+actor_filename):
#        start_ind=len(np.loadtxt('rewards.txt'))
#        actor_filename='actor_model_latest.h5'
#        critic_filename='critic_model_latest.h5'
    a2c_controller=A2C(2,16*3, actor_weights=actor_filename, critic_weights=critic_filename) # One output which will then be discretised

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
        obs=new_state[0][-16*3:]
        done=False
        truncated=False
        crashed=False
        t=0
        reward_max=0
        costs_max=0
        print("Episode {0} of {1}".format(e, E))

        markov_model.initialiseState(obs[15:].max())

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



reward_weights=np.array([[5, 3],
                         [1, 5],
                         [3, 6],
                         [6, 6],
                         [5, 2]]
)

for reward in reward_weights:
    actor_filename='actor_model_'+str(reward[0])+'_'+str(reward[1])+'_latest.h5'
    critic_filename='critic_model_'+str(reward[0])+'_'+str(reward[1])+'_latest.h5'
    print(actor_filename, critic_filename)
    markov_model_filename='markov_models/mm_'+str(reward[0])+'_'+str(reward[1])+'.pickle'
    genTransitions(actor_filename, critic_filename, markov_model_filename)

