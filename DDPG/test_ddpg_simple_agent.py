# !/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time

import random

from math import pi

import sys
import os
from Agent import Agent

agent=Agent()
agent.reset()

TT=50

fig, ax=plt.subplots()

agent_pos=[agent.getX(), agent.getY()]
goal_pos=[agent.goal_x, agent.goal_y]

scat=plt.scatter(agent.getX(), agent.getY(), s=100)
circle=plt.Circle((goal_pos[0], goal_pos[1]), 0.1, color=[0,0.7,0.])
ax.add_patch(circle)
ax.set_aspect('equal', 'box')

limit=3

plt.xlim(-limit,limit)
plt.ylim(-limit,limit)

def update_plot(i):
    agent.act()
    #ax.add_patch(circle)
    scat.set_offsets([agent.getX(),agent.getY()])
    dist=np.sqrt((agent.getX()-agent.goal_x)**2+(agent.getY()-agent.goal_y)**2)
    if i%50==0:
        print("DISTANCE TO GOAL: {0}".format(dist))
    return scat

ani=animation.FuncAnimation(fig, update_plot, frames=range(TT), interval=1)

plt.show()