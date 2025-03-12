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

TT=10e6
e_length=200

'''
for t in np.arange(10):
    agent.bufferCollection(50)
    agent.reset()
'''

for t in np.arange(TT):
    agent.episode(e_length)
    agent.reset()
    if t%5==0:
        print(t)
        agent.rl.saveNets()

## Uncomment to test trained networks:
# agent.reset()
# for t in np.arange(TT):
#     agent.act()