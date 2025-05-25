import matplotlib.pyplot as plt
import numpy as np
import distinctipy

def smooth_average_sma(data, window_size):
    sma_values=np.zeros(len(data))
    for i in range(len(data)):
    	start=max(0, i-window_size+1)
    	sma_values[i]=np.mean(data[start:i+1])
    return sma_values

# colours=distinctipy.get_colours(36)
reward_colour=(0.2,0.2,0.8)
success_colour=(0.2,0.8,0.2)
fail_colour=(0.8,0.2,0.2)

plt.rcParams['font.size']=28

reward=np.loadtxt("rewards/1_1_1.txt")
e_outcome=np.loadtxt("tracked_training_values/e_outcome.txt")
u_time=np.loadtxt("tracked_training_values/u_time.txt")

fig, ax_reward=plt.subplots()
ax_outcome=ax_reward.twinx()

## reward=blue, success=green, crash=red
ee_length=np.arange(len(reward))
lw=4.0

sma_reward=smooth_average_sma(reward, 5)
ax_reward.plot(ee_length, sma_reward, c=reward_colour, linewidth=lw, alpha=1.0, label='episode reward')
ax_reward.plot(ee_length, reward, c=reward_colour, linewidth=lw, alpha=0.4)
ax_reward.plot(0, 0, c=success_colour, linewidth=lw, label='cumulative goal success')
ax_reward.plot(0, 0, c=fail_colour, linewidth=lw, label='cumulative collisions')

success=e_outcome[:,0]>0
collisions=e_outcome[:,0]<0
success_count=[np.sum(success[:i+1]) for i in ee_length]
fail_count=[np.sum(collisions[:i+1]) for i in ee_length]
ax_outcome.plot(ee_length, success_count, c=success_colour, linewidth=lw, label='cumulative goal success')
ax_outcome.plot(ee_length, fail_count, c=fail_colour, linewidth=lw, label='cumulative collisions')
ax_reward.legend(loc='lower right')

ax_reward.set_xlabel("Episode")
ax_reward.set_ylabel("Reward")
ax_outcome.set_ylabel("Count", rotation=270, labelpad=20)

fig_time, ax_time=plt.subplots()

time_succeeded=np.where(e_outcome[:,0]==1)
times=e_outcome[time_succeeded, 1][0]
print(times)
ax_time.scatter(time_succeeded, times, label="Time to reach goal", c=success_colour, s=100)
print(smooth_average_sma(times, 3))
ax_time.plot(time_succeeded[0], smooth_average_sma(times,3), c=success_colour, linewidth=lw, linestyle='--')


ax_time.plot(ee_length, u_time, label="Time inside an unsafe zone", c=fail_colour, linewidth=lw)
ax_time.legend(loc="upper right")
ax_time.set_xlabel("Episode")
ax_time.set_ylabel("Time")

plt.show()