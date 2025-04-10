import safety_gymnasium
from gymnasium.wrappers import TimeLimit
from ddpg import DDPG
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
import yaml
from multiprocessing import Pool

def trainAgent(yaml_file, reward_file, cost_file, r_ind, c1_ind, c2_ind, seed):
    env_id = 'SafetyCarGoal1-v0'
    reward_gain=r_ind
    
    cost_hazard_gain=c1_ind
    cost_vase_gain=c2_ind

    env=safety_gymnasium.make(env_id, max_episode_steps=500)
    env.task.mechanism_conf.continue_goal=False
    env.set_seed(seed)
    ddpg_controller=DDPG(yaml_file)
    start_ind=0
    reward_timecourse=[]

    TT=1000
    tt=0
    learn_step=1
    start_learn=256

    state_start_ind=24

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
        if tt%learn_step==0 and tt>start_learn:
            ddpg_controller.learn()

        reward_max+=reward_total
        costs_max+=cost
        obs=obs_new
        t+=1
        tt+=1

        ddpg_controller.saveNets()
    return reward_max

### Make reward lists
## Get N pairs
N=1
reward_cost_array=np.random.randint(0,10,[N,3])

### Make YAML files
stream=open('template_ddpg.yaml', 'r')
rl_setup=yaml.safe_load(stream)

setups=[]

for [r,c1,c2] in reward_cost_array:
    setup=[]
    yaml_filename='yaml_files/ddpg_{0}_{1}_{2}.yaml'.format(r,c1, c2)
    reward_filename='rewards/{0}_{1}_{2}.txt'.format(r,c1, c2)
    cost_filename='costs/{0}_{1}_{2}.txt'.format(r,c1, c2)
    
    seed=np.random.randint(0,10e6)

    setups.append((yaml_filename, reward_filename, cost_filename, r, c1, c2, seed))
    
    model=rl_setup.copy()
    model['actor']['filename']='actor_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['actor']['target_filename']='actor_target_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['critic']['filename']='critic_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['critic']['target_filename']='critic_target_network_{0}_{1}_{2}.keras'.format(r, c1, c2)
    model['buffer filename']='buffer_{0}_{1}_{2}.npz'.format(r,c1, c2)
    with open(yaml_filename,'w') as f:
        yaml.dump(model, f)

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
    e+=1