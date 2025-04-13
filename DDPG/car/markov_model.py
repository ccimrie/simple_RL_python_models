import numpy as np

class MarkovModel():
	## Assume 4 vehicle stages: driving, driving in danger zone, crashed, goal reached
	## Assume 5 distances for danger zone
	## Assyme 5 distances for obstacle
	## 50 states total(?)

	def __init__(self):
		self.danger_distance_bins=5
		self.danger_distance_metric={}

		self.osbtacle_distance_bins=5
		self.obstacle_distance_metric={}

		self.transition_matrix_dict={}
		for i in np.arange(2):
			self.transition_matrix_dict[i]=np.zeros(self.danger_distance_bins+self.osbtacle_distance_bins+2)
		self.current_state=0

	def initialiseState(self, dist, danger_zone=False):
		self.danger_zone=danger_zone
		self.current_state=self.getDistState(dist)
		
	def addStateTransition(self, dist):
		next_state=self.getDistState(dist)
		self.transition_matrix_dict[self.current_state][next_state]+=1
		self.current_state=next_state

	def goalStateTransition(self):
		self.transition_matrix_dict[self.current_state][self.distance_bins+2]+=1

	def crashedStateTransition(self):
		self.transition_matrix_dict[self.current_state][self.distance_bins+1]+=1

	def getDistState(self, dist):
		return int(dist*self.distance_bins)