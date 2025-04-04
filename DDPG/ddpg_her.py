# import yaml
# import numpy as np
# import tensorflow as tf
# import time
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.utils import plot_model
# import random
# import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
from ddpg import DDPG

class DDPGHER(DDPG):
    def recordHER(self, prev_state, act, reward, state, done=0):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index]=prev_state
        self.action_buffer[index]=act
        self.reward_buffer[index]=reward
        self.next_state_buffer[index]=state
        self.done_buffer[index]=done

        self.saveBuffer()

        self.buffer_counter+=1
        self.batch_size=min(self.batch_max_size, int(np.ceil(0.1*self.buffer_counter)))