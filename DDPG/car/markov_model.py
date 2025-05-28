import numpy as np

class MarkovModel():
	## Assume 4 vehicle stages: driving, driving in danger zone, crashed, goal reached
	## Assume 3 distances for danger zone
	## Assyme 3 distances for obstacle
	## 3*3*4=36 states total

	def __init__(self):

		self.u_dist_bins=3
		self.o_dist_bins=3

		self.vehicle_stages=np.arange(4)
		self.u_dist=np.arange(self.u_dist_bins)
		self.o_dist=np.arange(self.o_dist_bins)

		self.combinations=np.array([[[[v, u, o] for v in self.vehicle_stages] for u in self.u_dist] for o in self.o_dist])
		self.combinations=np.reshape(self.combinations,(36, 3))

		self.transition_matrix_dict={}
		self.initial_state={}

	def initialiseState(self, dist_u, dist_o):
		state_u=self.getDistState(dist_u, self.u_dist_bins)
		state_o=self.getDistState(dist_o, self.o_dist_bins)
		self.state=f'0{state_u}{state_o}'
		if self.state not in self.transition_matrix_dict:
			self.transition_matrix_dict[self.state]={}
		if self.state in self.initial_state:
			self.initial_state[self.state]+=1
		else:
			self.initial_state[self.state]=1
		
	def addStateTransition(self, end, dist_u, dist_o):
		state_v=end
		state_u=self.getDistState(dist_u, self.u_dist_bins)
		state_o=self.getDistState(dist_o, self.o_dist_bins)
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