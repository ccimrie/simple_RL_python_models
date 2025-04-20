import numpy as np
import os

test_results_dir='test_results'
results_files=os.listdir(test_results_dir)

overall_results={}

for file in results_files:
    key_parts=file[:-4].split('_')[1:]
    key='-'.join(key_parts)
    results=np.load(f"{test_results_dir}/{file}")['results']
    print(results)
    sim_number=len(results)
    success=np.mean(results[:,0])
    energy_left=np.mean(results[results[:,0]>0][:,1])
    crashed=np.mean(results[:,2])
    danger_time=np.mean(results[:,3])

    overall_results[key]=f"  -- Success rate:  {success}\n  -- Energy left after completion:  {energy_left}\n  -- Crash rate:  {crashed}\n  -- Time spent in danger:  {danger_time}"

for key in overall_results:
    print(f"Results for {key}:")
    print(overall_results[key]+'\n')

