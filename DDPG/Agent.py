import numpy as np
from ddpg import DDPG
import time
import os

class Agent(object):

    def __init__(self, _arena_limit=3.0, _HER_no_subgoals=4):
        self.rl=DDPG('test_rl.yaml')

     ## Agent params
        self.x=0
        self.y=0
        self.goal_x=0
        self.goal_y=0
        self.vel_max=0.2
        self.arena_limit=_arena_limit
        
        reward_filename='reward.txt'
        op='a' if os.path.exists(reward_filename) else 'w+'
        self.reward_file=open(reward_filename, op)

        final_dist_filename='final_dist.txt'
        op='a' if os.path.exists(final_dist_filename) else 'w+'
        self.final_dist_file=open(final_dist_filename, op)

     ## Variables for HER
        self.state_HER=None
        self.state_HER_subgoals=[]
        self.HER_no_subgoals=_HER_no_subgoals

    
    def checkVelMag(self,act):
        ## Limit velocity components
        vel_mag=act[0]**2+act[1]**2
        if (vel_mag)>self.vel_max**2:
            factor=self.vel_max/np.sqrt(vel_mag)
            act[0]=act[0]*factor
            act[1]=act[1]*factor
        return act

    
    def move(self, act):
        ## move agent
        #print(act)
        #act_vel_checked=self.checkVelMag(act)
        self.x+=act[0]
        self.y+=act[1]

    
    def getX(self):
        return self.x

    
    def getY(self):
        return self.y
 
    
    def bufferCollection(self, TT):
        prev_dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
        thresh=0.05
        for t in np.arange(TT):
            prev_state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])

            ## Take step in the world ##
            act=self.rl.step(prev_state)
            act_vel=act*self.vel_max
            self.move(act_vel)

            ## Get next state
            state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])

            ## Get reward
            dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
            reward_sign=1.0
            reward=0#reward_sign*(1.0/(dist+1e-5))

            reward_const=1.0
            done=False
            reward=self.rewardFunc(prev_dist, dist)

            prev_dist=dist
            self.rl.recordStep(prev_state, act_vel, reward, state)

    
    def rewardFunc(self, prev_dist, dist):
        thresh=0.1
        factor=0.1

        #return (prev_dist[0]-dist[0])+(prev_dist[1]-dist[1])

        if abs(self.getX())>self.arena_limit or abs(self.getY())>self.arena_limit:
            reward=-100
        elif dist<thresh:
            print("REACHED GOAL!")
            reward=100.0/factor
        elif dist<prev_dist:
            reward=1.0/(dist+factor)
        else:
            reward=-0.1#dist
        return reward    


    def getDist(self, p1, p2):
        diff=p1-p2
        dist=np.sqrt(np.dot(diff, diff))
        return dist

    
    def getPos(self):
        return np.array([self.getX(), self.getY()])


    def getGoalPos(self):
        return np.array([self.goal_x, self.goal_y])


    def episode(self, TT):
        thresh=0.05
        e_reward=0
        vel_file=open('vel_timecourse.txt','w+')
        dist_to_goal_file=open('dist_to_goal_timecourse.txt','w+')

        prev_dist_lst=np.array([self.getDist(self.getPos(), self.getGoalPos())])
        prev_dist=self.getDist(self.getPos(), self.getGoalPos())
        closest_dist=prev_dist

        # self.state_HER_subgoals.append([self.getX(), self.getY(), self.getNormalisedState(), act, state])

        for t in np.arange(TT):
            print(f"  - Timestep: {t}")
            ## Get state
            prev_pos=np.array([self.getX(), self.getY()])
            prev_state=self.getNormalisedState()

            ## Get current distance to goal
            prev_dist=self.getDist(self.getPos(), self.getGoalPos())

            ## Take step in the world
            act=self.rl.step(prev_state)
            act_vel=act*self.vel_max
            self.move(act_vel)
            # vel_record=np.sqrt(np.dot(act_vel))
            vel_file.write("{0}  {1}\n".format(act_vel[0], act_vel[1]))
            vel_file.flush()

            ## Get next state
            next_pos=np.array([self.getX(), self.getY()])
            state=self.getNormalisedState()
            dist_lst=np.array([self.getDist(self.getPos(), self.getGoalPos())])#[abs(self.getX()-self.goal_x), abs(self.getY()-self.goal_y)]
            dist=self.getDist(self.getPos(), self.getGoalPos())
            dist_to_goal_file.write("{0}\n".format(dist))
            dist_to_goal_file.flush()
            
            ## Get reward
            reward_sign=1.0
            reward=0
            reward_const=1.0
            done=False
            #reward=self.rewardFunc(prev_dist_lst, dist_lst)
            reward=self.rewardFunc(prev_dist, dist)

            if dist<closest_dist:
                closest_dist=dist
            e_reward+=reward
            self.rl.recordStep(prev_state, act, reward, state)
            if reward==-100:
                print(f"  - Problem of leaving:  {prev_pos}  {act}  {act_vel}  {next_pos}")
                self.rl.learn()
                break

          ## Check HER
            self.state_HER_subgoals.append([prev_pos, prev_state, act, next_pos, state])
            if len(self.state_HER_subgoals)>self.HER_no_subgoals:
                self.recordHER()
            self.rl.learn()


        ## Record episode
        dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
        self.final_dist_file.write(str(dist))
        self.final_dist_file.write('\n')
        self.final_dist_file.flush()
        self.reward_file.write(str(e_reward))
        self.reward_file.write('\n')
        self.reward_file.flush()

        print("Closest agent got to goal: {0}".format(closest_dist))

        self.state_HER_subgoals=[]
        '''
        print("Agent moved {0} times towards goal with a total distance of {1}".format(moved_closer_count, moved_closer_sum))
        print("Agent moved {0} times away from goal with a total distance of {1}/{2}".format(moved_away_count, moved_away_sum, moved_away_vel_sum))
        print("Total distance moved (x,y): ({0}, {1})".format(abs(start_x-self.getX()), abs(start_y-self.getY())))
        '''

    
    def recordHER(self):
     ## 1. Get state to apply HER to 
        prev_pos=self.state_HER_subgoals[0][0]
        prev_state=self.state_HER_subgoals[0][1]
        act=self.state_HER_subgoals[0][2]
        next_pos=self.state_HER_subgoals[0][3]

     ## 2. Get subgoals from next states
        for [_, _, _, goal_pos, next_state] in self.state_HER_subgoals:
         ## 3. Augment states with new subgoal
            aug_goal=goal_pos/self.arena_limit
            prev_state[-2:]=aug_goal
            next_state[-2:]=aug_goal

         ## 4. Calculate rewards w.r.t. augmented goals
            prev_dist=self.getDist(prev_pos, goal_pos)
            dist=self.getDist(next_pos, goal_pos)
            reward=self.rewardFunc(prev_dist, dist)
         
         ## 5. Record new transitions
            self.rl.recordStep(prev_state, act, reward, next_state)

     ## 6. Remove first state-act pair
        self.state_HER_subgoals=self.state_HER_subgoals[1:]

    
    def getNormalisedState(self):
        state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])
        state/=self.arena_limit
        return state

    
    def act(self):
        state=self.getNormalisedState()
        act=self.rl.act(state)
        self.move(act)


    def reset(self):
        ## Reset agent and goal position
        self.x=np.random.uniform(-self.arena_limit,self.arena_limit)
        self.y=np.random.uniform(-self.arena_limit,self.arena_limit)

        self.goal_x=np.random.uniform(-self.arena_limit,self.arena_limit)
        self.goal_y=np.random.uniform(-self.arena_limit,self.arena_limit)