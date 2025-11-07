import numpy as np
import pickle

dtmc_model=open('dtmc_start.txt', 'r').readlines()

file = open("markov_models/mm_1_1_1.pickle",'rb')
object_file = pickle.load(file)
combinations=object_file.transition_matrix_dict
initialisation_states=object_file.initial_state

total_sum=np.sum([initialisation_states[i_s] for i_s in initialisation_states])
initialisation_line_start='[] t=-1 ->\n'
initialisation_state_template='{0}:(vehicle_stage\'={1})&(u_dist\'={2})&(o_dist\'={3})&(t\'=0)'
for i_s in initialisation_states:
	p=initialisation_states[i_s]/total_sum
	line=initialisation_state_template.format(p, i_s[0], i_s[1], i_s[2])
	initialisation_line_start+=(line+' + \n')

state_transition_lines=initialisation_line_start[:-3]+';\n\n'
for combination in combinations:
	if len(combinations[combination])>0:
		com_transitions=combinations[combination]
		total_sum=np.sum([com_transitions[next_state] for next_state in com_transitions])
		transition_lines=f"[] t>-1 & t<e_lim & vehicle_stage={combination[0]} & u_dist={combination[1]} & o_dist={combination[2]} -> \n"
		transition_template="{0}: (vehicle_stage'={1})&(u_dist'={2})&(o_dist'={3})&(t'=t+1)"
		for next_state in com_transitions:
			p=com_transitions[next_state]/total_sum
			transition_line=transition_template.format(p, next_state[0], next_state[1], next_state[2])
			transition_lines+=(transition_line+' +\n')
		state_transition_lines+=transition_lines[:-3]+';\n\n'

dtmc_model_strip=[line.strip() for line in dtmc_model]
ind=dtmc_model_strip.index('endmodule')
# print(dtmc_model)
dtmc_model.insert(ind, state_transition_lines)
# print(state_transition_lines)
with open('rl_vehicle_dtmc.pm', 'w') as f:
	[f.write(line) for line in dtmc_model]
# vehicle_stage=np.arange(4)
# u_dist=np.arange(3)
# o_dist=np.arange(3)

# combinations=np.array([[[[v, u, o] for v in vehicle_stage] for u in u_dist] for o in o_dist])
# combinations=np.reshape(combinations,(36, 3))

# def genTransitions(v, u, o, V, U, O):
# 	transition_lines=""
# 	gen_temp="p_{0}{1}{2}_{3}{4}{5}: (vehicle_stage'={3})&(u_dist'={4})&(o_dist'={5})&(t'=t+1)"
# 	for vv in V:
# 		for uu in U:
# 			for oo in O[:-1]:
# 				transition_lines+=(gen_temp.format(v, u, o, vv, uu, oo)+" +\n")
# 	transition_lines+=gen_temp.format(v, u, o, V[-1], U[-1], O[-1])+";\n\n"
# 	return transition_lines

# [] t<TT & vehicle_stage=A & u_dist=B & o_dist=C -> p_ABC_DEF: (vehicle_stage'=D)&(u_dist'=E)&(o_dist'=F)&(t'=t+1) +
# all_transitions=""
# state_line_part_temp="[] t<TT & vehicle_stage={0} & u_dist={1} & o_dist={2} -> "

# for v in vehicle_stage:
# 	for u in u_dist:
# 		for o in o_dist: 
# 			state_line_part=state_line_part_temp.format(v,u,o)
# 			transitions=genTransitions(v, u, o, vehicle_stage, u_dist, o_dist)
# 			all_transitions+=(state_line_part+transitions)
# with open('transitions.txt', 'w') as f:
# 	f.write(all_transitions)
# 	f.close()

# combinations=np.array([[[[v, u, o] for v in vehicle_stage] for u in u_dist] for o in o_dist])
# combinations=np.reshape(combinations,(36, 3))
# with open('probabilties.txt', 'w') as f:
# 	for [v, u, o] in combinations:
# 		for [vv, uu, oo] in combinations:
# 			f.write(f'const double p_{v}{u}{o}_{vv}{uu}{oo};\n')
