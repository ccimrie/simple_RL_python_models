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

reward_weights=np.array([[5, 3],
                         [1, 5],
                         [3, 6],
                         [6, 6],
                         [5, 2]]
)


reward_files=os.listdir('rewards/')

colours=[[255/255.0,165/255.0,0],
          [0,255/255.0,127/255.0],
          [0, 191/255.0, 255/255.0],
          [0, 0, 255/255.0],
          [255/255.0, 20/255.0, 147/255.0]]

c=0
for file in reward_files:
    reward=np.loadtxt('rewards/'+file)
    gains=file[:-4].split('_')
    label=f"({gains[0]}, {gains[1]}, {gains[2]})"
    plt.plot(reward, c=colours[c], alpha=0.4)
    plt.plot(smooth_average_sma(reward, 5), c=colours[c], label=label)
    c+=1
plt.legend()
#plt.plot(np.log(reward-reward.min()+0.001), color=[0,0,1])
#plt.plot(np.log(cost-cost.min()+0.001), color=[0.5,0,0])
#plt.plot(np.log((reward-cost-(reward-cost).min())+0.001), color=[0.7, 0, 0.2])
plt.show()
