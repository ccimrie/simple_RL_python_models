import matplotlib.pyplot as plt
import numpy as np
import os

def smooth_average_sma(data, window_size):
    if len(data) < window_size:
        return []  # Not enough data to form a full window
    sma_values = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        sma_values.append(sum(window) / window_size)
    return sma_values

reward_str=[1.0,1.0]

reward_filename='rewards_ddpg.txt'
if os.path.exists(reward_filename):
	reward=np.loadtxt(reward_filename) 
	#cost=np.loadtxt('costs.txt')
	plt.plot(reward)
	plt.plot(smooth_average_sma(reward, 10))

#plt.plot(np.log(reward-reward.min()+0.001), color=[0,0,1])
#plt.plot(np.log(cost-cost.min()+0.001), color=[0.5,0,0])
#plt.plot(np.log((reward-cost-(reward-cost).min())+0.001), color=[0.7, 0, 0.2])
plt.show()
