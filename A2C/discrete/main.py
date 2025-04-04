# !/usr/bin/env python3
import numpy as np
import math
import time

import random

from math import pi

import sys
import os
from Agent import Agent

agent=Agent()
agent.reset()

TT=10000
eps_length=1000

for t in np.arange(TT):
    agent.episode(eps_length)
    #agent.learn()
    agent.reset()
    print(f"\n\nEpisode:  {t}")
    if t%1==0:
        agent.rl.saveNets()

## Uncomment to test trained networks:
# agent.reset()
# for t in np.arange(TT):
#     agent.act()