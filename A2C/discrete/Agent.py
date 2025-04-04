import numpy as np
from a2c_discrete import DiscreteA2C
import time

class Agent(object):

    def __init__(self):
        self.rl=DiscreteA2C('test_rl.yaml')

        ## Agent params
        self.x=0
        self.y=0
        self.goal_x=0
        self.goal_y=0
        self.vel_max=0.05

        self.vel=np.zeros(2)

        self.act_dict={0: [0.01, 0.0], 1: [0.0, 0.01], 2: [0.01, 0.01], 
                       3: [-0.01, 0.0], 4: [0.0, -0.01], 5: [-0.01, -0.01],
                       6: [0.01, -0.01], 7: [-0.01, 0.01], 8: [0,0]}
        # self.act_dict={0: [0.01, 0.0], 1: [-0.01, 0.0], 
        #                2: [0.0, 0.01], 3: [0.0, -0.01], 4: [0,0]}

        self.f=open('test.txt', 'w+')

   
    def checkVelMag(self,act):
        ## Limit velocity components
        vel_mag=act[0]**2+act[1]**2
        if (vel_mag)>self.vel_max**2:
            factor=self.vel_max/np.sqrt(vel_mag)
            act[0]=act[0]*factor
            act[1]=act[1]*factor
        return act

   
    def move(self):
        #print(self.vel)
        self.x=self.x+self.vel[0]
        self.y=self.y+self.vel[1]

   
    def getX(self):
        return self.x

   
    def getY(self):
        return self.y
 
    def updateVel(self, _act):
        act=self.act_dict[_act]      
        #self.vel=np.clip(self.vel+act, -self.vel_max, self.vel_max)
        #print(act, self.act_dict, _act)
        self.vel=act
    
    def getState(self):
        state_position=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])
        state_position=(state_position+3)/6
        #state_vel=np.array([self.vel[0]/self.vel_max, self.vel[1]/self.vel_max])
        # state=np.hstack([state_position, state_vel])
        state=state_position
        return state


    def episode(self, TT):
        thresh=0.05
        done=False
        reward_total=0
        prev_dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
        learning_length=250
        for t in np.arange(TT):
            state=self.getState()#, self.vel[0]/0.5, self.vel[1]/0.5])
            #print(f"Timestep:  {t}\nState:  {state/5.0}\nVelocity:  {self.vel}")

            ## Take step in the world
            act, act_out=self.rl.step(state)
            self.updateVel(act)
            self.move()

            ## Get reward
            dist=np.sqrt((self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2)
            reward=1.0/(dist+1e-2) if prev_dist>dist else -1.5
            #print(f"Prev dist and new dist:  {act}  {prev_dist}  {dist}")
            #reward=5.0 if prev_dist>dist else -0.5
            prev_dist=dist
            if dist<thresh:
                reward=150
                done=True
            elif abs(self.getX())>3 or abs(self.getY())>3:
                reward=-150
                done=True
            reward_total+=reward
            self.rl.recordReward(reward)
            if done: 
                self.learn()
                break
            elif t%learning_length==0:
                self.learn()
        self.f.write(f"{reward_total}\n")
        self.f.flush()


    def learn(self):
        print(f"Learning time")
        self.rl.learnActor()
        self.rl.learnCritic()
        self.rl.clearHistory()


    def act(self):
        state=self.getState()
        ## Take step in the world
        act=self.rl.act(state).copy()
        #print(act)
        self.updateVel(act)
        self.move()

        ## Get reward
        dist=(self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2
        reward=1.0/(np.sqrt(dist)+1e-2)
        print(reward)

    def reset(self):
        ## Reset agent and goal position
        self.x=np.random.uniform(-3,3)
        self.y=np.random.uniform(-3,3)

        self.goal_x=np.random.uniform(-3,3)
        self.goal_y=np.random.uniform(-3,3)
        #self.vel*=0
        #self.rl.clearHistory()

