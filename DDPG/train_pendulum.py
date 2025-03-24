import numpy as np
# import safety_gymnasium
import gym
from ddpg import DDPG
import matplotlib.pyplot as plt 
import time
import os

env_id = 'Pendulum-v1'
env=gym.make(env_id, render_mode='human')
#env.task.mechanism_conf.continue_goal=False

rl_agent=DDPG('test_pendulum_rl.yaml')

reward_timecourse=[]

e=0
E=100
TT=1000

reward_filename='rewards_pendulum.txt'
op='a' if os.path.exists(reward_filename) else 'w+'
rewards_txt=open(reward_filename, op)

while e<E:
    new_state=env.reset()
    state=new_state[0]
    done=False
    truncated=False
    t=0
    reward_max=0
    print(e)
    while not done and not truncated and t<TT:
        if (t+1)%50==0:
            print('    - '+str(t+1))
        act=rl_agent.step(state)
        outcome=env.step(act*2.0)
        next_state=outcome[0]
        reward=outcome[1]
        cost=outcome[2]
        done=outcome[3]
        truncated=outcome[4]
        #print(f"Outcome:  {state}  {act}  {reward}  {next_state}")
        rl_agent.recordStep(state, act, reward, next_state)
        rl_agent.learn()
        reward_max+=reward
        state=next_state
        t+=1

    rewards_txt.write(str(reward_max)+'\n')
    rewards_txt.flush()
    e+=1
    if e%1==0:
        print("Saving networks")
        rl_agent.saveNets()