import numpy as np

results=np.load("test_results/results_1_1_1.npz")['results']
completed_mission=np.mean(results[:,0])
energy_remaining=np.mean(results[np.where(completed_mission>0),1])
crashed=np.mean(results[:,2])
danger_time=np.mean(results[:,3])


completed_mission_str_out='  - Successeful mission:  {completed_mission}'
energy_remaining_str_out='   - Average energy when successful:  {energy_remaining}'
crashed_str_out='  - Average collision rate:  {crashed}'
danger_time_str_out='  - Average time in unsafe zones:  {danger_time}'
print(f"Results are:\n{completed_mission_str_out}\n{energy_remaining_str_out}\n{crashed_str_out}\n{danger_time_str_out}")