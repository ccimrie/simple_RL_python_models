import safety_gymnasium
from gymnasium.wrappers import TimeLimit
from DDPG.ddpg import DDPG
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
import yaml
from multiprocessing import Pool
#import tensorflow as tf

def trainAgent(yaml_file, reward_filename, cost_file, r_ind, c1_ind, c2_ind, seed):

    print("AGENT IS INITIALISED WITH FILES: ", yaml_file, reward_filename, cost_file, r_ind, c1_ind, c2_ind, seed)


    env_id = 'SafetyCarGoal1-v0'
    reward_gain=r_ind
    
    cost_hazard_gain=c1_ind
    cost_vase_gain=c2_ind

    env=safety_gymnasium.make(env_id, max_episode_steps=500)
    env.task.mechanism_conf.continue_goal=False
    env.set_seed(seed)
    print(f"AGENT IS USING FILE {yaml_file}")
    ddpg_controller=DDPG(yaml_file)
    start_ind=0
    reward_timecourse=[]

    TT=1000
    tt=0
    learn_step=1
    start_learn=256

    state_start_ind=24

    e=0
    E_max=250

    print(f"AGENT IS USING REWARD FILE {reward_filename} FOR LOADING")
    reward_temp=len(np.loadtxt(reward_filename)) if os.path.exists(reward_filename) else 0
    E=E_max-reward_temp
    
    while e<E:
        print(f"  - Episode {e}/{E}")
        new_state=env.reset()
        obs=new_state[0][state_start_ind:]
        done=False
        truncated=False
        crashed=False
        t=0
        reward_max=0
        costs_max=0
        while not done and not truncated and not crashed:
            cost_hazard=0
            cost_vase=0

            act=ddpg_controller.step(obs)
            new_state=env.step(act)
            obs_new=new_state[0][state_start_ind:]
            #print(obs_new)
            reward=new_state[1]
            cost=new_state[2]
            done=new_state[3]
            truncated=new_state[4]

            ## Check if crashed into obstacle
            if cost>100:
                cost_vase=10
                crashed=True
                if cost-1000>0:
                    cost_hazard=0.1
            elif cost>0:
                cost_hazard=0.1

            reward_total=reward*reward_gain-cost_hazard*cost_hazard_gain-cost_vase*cost_vase_gain
            ddpg_controller.recordStep(obs, act, reward_total, obs_new)
            
            ddpg_controller.learn()

            reward_max+=reward_total
            costs_max+=cost
            obs=obs_new
            t+=1
            tt+=1

            ddpg_controller.saveNets()
        
        print(f"AGENT IS USING REWARD FILE {reward_filename} FOR CHECKING")
        if os.path.exists(reward_filename):
            permission='a'
        else:
            permission='w'
        print(f"AGENT IS USING REWARD FILE {reward_filename} FOR WRITING")
        with open(reward_filename, permission) as f:
            f.write(str(reward_max)+'\n')
            f.close()
        e+=1
    return reward_max

### Make reward lists
## Get N pairs
N=5

setup_filename='setup/setup.npz'


## TODO improve so that it can select which are below max train episodes
if os.path.exists(setup_filename):
    reward_cost_array=np.load(setup_filename)['gains']
    NN=len(reward_cost_array)
    if NN<N:
        new_vals=np.random.randint(1,10,[N-NN, 3])
        reward_cost_array=np.vstack((reward_cost_array, new_vals))
        np.savez('setup/setup', gains=reward_cost_array)
    elif NN>N:
        #ind=np.random.randint(0,N,N)
        reward_cost_array=reward_cost_array[:N]
else:
    reward_cost_array=np.random.randint(1,10,[N,3])
    np.savez('setup/setup', gains=reward_cost_array)

### Make YAML files
stream=open('template_ddpg.yaml', 'r')
rl_setup=yaml.safe_load(stream)

setups=[]

for [r,c1,c2] in reward_cost_array:
    setup=[]
    yaml_filename='yaml_files/ddpg_{0}_{1}_{2}.yaml'.format(r,c1, c2)
    r_filename='rewards/{0}_{1}_{2}.txt'.format(r,c1, c2)
    cost_filename='costs/{0}_{1}_{2}.txt'.format(r,c1, c2)
    
    seed=np.random.randint(0,10e6)

    setups.append((yaml_filename, r_filename, cost_filename, r, c1, c2, seed))
    
    model=rl_setup.copy()
    model['actor']['filename']='actor_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['actor']['target_filename']='actor_target_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['critic']['filename']='critic_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['critic']['target_filename']='critic_target_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['buffer filename']='buffer_{0}_{1}_{2}.npz'.format(r,c1, c2)
    with open(yaml_filename,'w') as f:
        yaml.dump(model, f)

print(setups)
'''
for i in np.arange(len(setups)):
        print(f"Agent {i}/{N}")
        reward=trainAgent(setups[i][0], setups[i][1], setups[i][2], setups[i][3], setups[i][4], setups[i][5], setups[i][6])
'''
with Pool(5) as p:
    rewards=p.starmap(trainAgent, setups)

'''
e=0
E=800
while e<E:
    print("Episode: {0}".format(e))
    with Pool(5) as p:
        rewards=p.starmap(trainAgent, setups)
    print(rewards, reward_cost_array)
    for i in np.arange(len(rewards)):
        if os.path.exists(reward_filename):
            permission='a'
        else:
            permission='w'
        with open(setups[i][1], permission) as f:
            f.write(str(rewards[i])+'\n')    
        temp_setup=list(setups[i])
        temp_setup[5]=np.random.randint(0,10e6)
        setups[i]=tuple(temp_setup)
    print(f"Conducting episode {e}/{E}")
    for i in np.arange(len(setups)):
        print(f"  - Agent {i}/{N}")
        reward=trainAgent(setups[i][0], setups[i][1], setups[i][2], setups[i][3], setups[i][4], setups[i][5], setups[i][6])
        reward_filename=setups[i][1]
        if os.path.exists(reward_filename):
            permission='a'
        else:
            permission='w'
        with open(setups[i][1], permission) as f:
            f.write(str(reward[i])+'\n')    
        temp_setup=list(setups[i])
        temp_setup[5]=np.random.randint(0,10e6)
        setups[i]=tuple(temp_setup)
    e+=1
    '''