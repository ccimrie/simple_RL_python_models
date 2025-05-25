import matplotlib.pyplot as plt
import numpy as np
import distinctipy

def smooth_average_sma(data, window_size):
    if len(data) < window_size:
        return []  # Not enough data to form a full window
    sma_values = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        sma_values.append(sum(window) / window_size)
    return sma_values

# colours=distinctipy.get_colours(36)
reward_colour=(0.2,0.2,0.8)
success_colour=(0.2,0.8,0.2)
fail_colour=(0.8,0.2,0.2)

reward=np.loadtxt("rewards/1_1_1.txt")
e_outcome=np.loadtxt("tracked_training_values/e_outcome.txt")
u_time=np.loadtxt("tracked_training_values/u_time.txt")

fig, ax_reward=plt.subplots()
ax_outcome=ax_reward.twinx()

## reward=blue, success=green, crash=red
ee_length=np.arange(len(reward))

sma_reward=smooth_average_sma(reward, 5)
ax_reward.plot(ee_length, sma_reward, c=reward_colour, alpha=1.0, label='episode reward')
ax_reward.plot(ee_length, reward, c=reward_colour, alpha=0.4)

success_count=[np.sum(e_outcome[:i+1,0]>0) for i in np.arange(ee_length)]
fail_count=[np.sum(e_outcome[:i+1,0]<0) for i in np.arange(ee_length)]
ax_outcome.plot(ee_length, success_count, c=success_colour, label='cumulative collisions')
ax_outcome.plot(ee_length, fail_count, c=fail_colour, label='cumulative collisions')

plt.show()