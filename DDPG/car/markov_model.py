import numpy as np

class MarkovModel():
	## Assume 4 vehicle stages: driving, driving in danger zone, crashed, goal reached
	## Assume 3 distances for danger zone
	## Assyme 3 distances for obstacle
	## 3*3*4=36 states total

	def __init__(self):
		self.vehicle_stages=np.arange(4)
		self.u_dist=np.arange(3)
		self.o_dist=np.arange(3)

		self.combinations=np.array([[[[v, u, o] for v in self.vehicle_stage] for u in self.u_dist] for o in self.o_dist])
		self.combinations=np.reshape(combinations,(36, 3))

		self.transition_matrix_dict={}
		self.initial_state={}

	def initialiseState(self, dist_u, dist_o):
		state_u=self.getDistState(dist_u, self.u_dist)
		state_o=self.getDistState(dist_o, self.o_dist)
		self.state=f'0{state_u}{state_o}'
		if self.state not in self.transition_matrix_dict:
			self.transition_matrix_dict[self.state]={}
		self.initial_state[self.state]+=1
		
	def addStateTransition(self, end, dist_u, dist_o):
		state_v=end
		state_u=self.getDistState(dist_u, self.u_dist)
		state_o=self.getDistState(dist_o, self.o_dist)
		next_state=f'{state_v}{state_u}{state_o}'
		if next_state in self.transition_matrix_dict[self.state]:
			self.transition_matrix_dict[self.state][next_state]+=1
		else:
			self.transition_matrix_dict[self.state][next_state]=1
		if next_state not in self.transition_matrix_dict:
			self.transition_matrix_dict[next_state]={}
		self.state=next_state

	def getDistState(self, dist, bins):
		return int(dist*bins)