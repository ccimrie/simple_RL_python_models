import numpy as np
from ddpg import DDPG
import time

class Agent(object):

    def __init__(self, _arena_limit=10.0):
        self.rl=DDPG('test_rl.yaml')

        ## Agent params
        self.x=0
        self.y=0
        self.goal_x=0
        self.goal_y=0
        self.vel_max=0.5
        self.arena_limit=_arena_limit
        self.reward_file=open('reward.txt', 'w+')
        self.final_dist_file=open('final_dist.txt', 'w+')

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
        thresh=0.5
        factor=1

        return (prev_dist[0]-dist[0])+(prev_dist[1]-dist[1])

        if dist<thresh:
            print("REACHED GOAL!")
            reward=100.0/factor
        elif dist<prev_dist:
            reward=0.1#1.0/(dist+factor)
        else:
            reward=-0.1#dist
        return reward    

    def episode(self, TT):
        thresh=0.05
        e_reward=0
        vel_file=open('vel_timecourse.txt','w+')
        dist_to_goal_file=open('dist_to_goal_timecourse.txt','w+')


        prev_dist_lst=[abs(self.getX()-self.goal_x), abs(self.getY()-self.goal_y)]
        prev_dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
        closest_dist=prev_dist
        '''
        ## Checking agent
        moved_away_count=0
        moved_away_sum=0
        moved_away_vel_sum=0
        moved_closer_count=0
        moved_closer_sum=0

        start_x=self.getX()
        start_y=self.getY()
        '''

        for t in np.arange(TT):
            prev_state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])
            #prev_state=np.array([self.getX()-self.goal_x, self.getY()-self.goal_y])
            '''
            prev_x=prev_state[0]
            prev_y=prev_state[1]
            '''

            ## Take step in the world ##
            act=self.rl.step(prev_state)
            act_vel=act*self.vel_max
            #print("out act", act,)
            #act_vel=act.copy()*self.vel_max
            #print("Velocity evolution pt 1: {0}  {1}".format(act, act_vel))
            #print(act)
            self.move(act_vel)
            #print("Velocity evolution pt2: {0}  {1}".format(act, act_vel))
            vel_record=np.sqrt(act_vel[0]**2+act_vel[1]**2)
            vel_file.write("{0}\n".format(vel_record))
            vel_file.flush()

            ## Get next state
            state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])
            #state=np.array([self.getX()-self.goal_x, self.getY()-self.goal_y])

            ## Get reward
            dist_lst=[abs(self.getX()-self.goal_x), abs(self.getY()-self.goal_y)]
            dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
            dist_to_goal_file.write("{0}\n".format(dist))
            dist_to_goal_file.flush()

            #if np.sqrt(dist)<thresh:
            #   print("REACHED: ", np.sqrt(dist))
            #else:
            #    reward=1.0/(np.sqrt(dist)+1e-2)

            #reward_sign=1.0 if dist<prev_dist else -1.0
            reward_sign=1.0
            reward=0#reward_sign*(1.0/(dist+1e-5))

            reward_const=1.0
            done=False
            #reward=self.rewardFunc(prev_dist, dist)
            reward=self.rewardFunc(prev_dist_lst, dist_lst)

            prev_dist=dist
            prev_dist_lst=dist_lst
            if prev_dist<closest_dist:
                closest_dist=prev_dist
            #print("DIST/REWARD: ", dist, reward)
            # print("AT END")
            e_reward+=reward
            #print(prev_state, act, reward, state)
            self.rl.recordStep(prev_state, act, reward, state)
            #print(self.rl.buffer_counter)
            self.rl.learn()

        ## Learn
        dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
        self.final_dist_file.write(str(dist))
        self.final_dist_file.write('\n')
        self.final_dist_file.flush()
        ##reward=1.0/(np.sqrt(dist)+1e-2)
        self.reward_file.write(str(e_reward))
        self.reward_file.write('\n')
        self.reward_file.flush()

        print("Closest agent got to goal: {0}".format(closest_dist))

        '''
        print("Agent moved {0} times towards goal with a total distance of {1}".format(moved_closer_count, moved_closer_sum))
        print("Agent moved {0} times away from goal with a total distance of {1}/{2}".format(moved_away_count, moved_away_sum, moved_away_vel_sum))
        print("Total distance moved (x,y): ({0}, {1})".format(abs(start_x-self.getX()), abs(start_y-self.getY())))
        '''

    def act(self):
        state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])
        act=self.rl.act(state)
        self.move(act)

    def reset(self):
        ## Reset agent and goal position
        self.x=np.random.uniform(-self.arena_limit,self.arena_limit)
        self.y=np.random.uniform(-self.arena_limit,self.arena_limit)

        self.goal_x=np.random.uniform(-self.arena_limit,self.arena_limit)
        self.goal_y=np.random.uniform(-self.arena_limit,self.arena_limit)