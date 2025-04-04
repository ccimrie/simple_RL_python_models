import yaml
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import random
import os
os.environ["KERAS_BACKEND"] = "tensorflow"


"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev=x
        return x

    def reset(self):
        print("Resetting noise")
        if self.x_initial is not None:
            self.x_prev=self.x_initial
        else:
            self.x_prev=np.zeros_like(self.mean)


class DDPG:
##### Setting up RL class #####
    def __init__(self, rl_yaml):
        seed=np.random.randint(0,10e6)
        tf.random.set_seed(seed)
        stream=open(rl_yaml, 'r')
        rl_setup=yaml.safe_load(stream)

        ## RL params
        self.gamma=rl_setup['gamma']
        actor=rl_setup['actor']
        critic=rl_setup['critic']
        self.actor_optimizer=keras.optimizers.Adam(actor['learning rate'], clipnorm=actor['clipnorm'])
        self.critic_optimizer=keras.optimizers.Adam(critic['learning rate'], clipnorm=critic['clipnorm'])
        self.huber_loss = keras.losses.Huber()
        self.num_actions=rl_setup['action size']
        self.state_size=rl_setup['state size']
        self.tau=rl_setup['tau']

        self.model_dir=rl_setup['model dir']
        self.actor_filename=self.model_dir+'/'+actor['filename']
        self.actor_target_filename=self.model_dir+'/'+actor['target filename']
        self.critic_filename=self.model_dir+'/'+critic['filename']
        self.critic_target_filename=self.model_dir+'/'+critic['target filename']

        self.buffer_filename=self.model_dir+'/'+rl_setup['buffer filename']
        print(self.buffer_filename)

        if os.path.exists(self.actor_filename):
            print("Loading models")
            self.actor=keras.models.load_model(self.actor_filename)
            self.actor_target=keras.models.load_model(self.actor_target_filename)
            self.critic=keras.models.load_model(self.critic_filename)
            self.critic_target=keras.models.load_model(self.critic_target_filename)
        else:
            print("Creating models")
            ## Create actor models
            self.actor=self.createActorModel(actor['weights'])
            self.actor_target=self.createActorModel(actor['weights'])
            self.actor_target.set_weights(self.actor.get_weights())
            ## Create critic models
            self.critic=self.createCriticModel(critic['state input weights'], critic['action input weights'], critic['weights'])
            self.critic_target=self.createCriticModel(critic['state input weights'], critic['action input weights'], critic['weights'])
            self.critic_target.set_weights(self.critic.get_weights())
            self.viewModels()
            #self.critic_target=keras.models.clone_model(self.critic, self.critic.get_weights())
        
        ## Set up buffer
        # Number of "experiences" to store at max
        self.buffer_capacity=rl_setup['buffer capacity']
        # Num of tuples to train on.
        self.batch_max_size=rl_setup['batch size']

        # Its tells us num of times record() was called.
        self.buffer_counter=0

        create_buffer=True
        ## To-do: make buffer name part of yaml file
        if os.path.exists(self.buffer_filename):
            create_buffer=False
            print("Loading buffer: {0}".format(self.buffer_filename))    
            try:
                buffer=np.load(self.buffer_filename)
                self.state_buffer=buffer['state_buffer']
                self.action_buffer=buffer['action_buffer']
                self.reward_buffer=buffer['reward_buffer']
                self.next_state_buffer=buffer['next_state_buffer']
                self.done_buffer=buffer['done_buffer']
                self.buffer_counter=buffer['buffer_counter'][0]
            except Exception as e:
                # If loading fails, print an error message and skip
                print("Error in loading buffer {0}".format(self.buffer_filename))
                create_buffer=True 
                #continue

        if create_buffer==True:
            print("Creating buffer")
            # Instead of list of tuples as the exp.replay concept go
            # We use different np.arrays for each tuple element
            self.state_buffer=np.zeros((self.buffer_capacity, self.state_size))
            self.action_buffer=np.zeros((self.buffer_capacity, self.num_actions))
            self.reward_buffer=np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer=np.zeros((self.buffer_capacity, self.state_size))
            self.done_buffer=np.zeros((self.buffer_capacity, 1))
        ## Noise model
        std_dev=0.1
        self.ou_noise=OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev)*np.ones(1))


##### Creating DNN models for DRL #####
    def createActorModel(self, weights):
        ## Create actor network
        actor_layers=[layers.Input(shape=(self.state_size,))]
        for layer in weights:
            actor_layers.append(layers.Dense(layer, activation='relu')(actor_layers[-1]))
        # Initialize weights between -3e-3 and 3-e3
        last_init=keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        outputs=layers.Dense(self.num_actions, activation='tanh', kernel_initializer=last_init)(actor_layers[-1])
        actor_net=keras.Model(inputs=actor_layers[0], outputs=outputs)
        return actor_net

    def createCriticModel(self, state_weights, act_weights, weights):
        ## Create critic model
        ##  - Set up pre-layers for state input
        critic_state_layers=[layers.Input(shape=(self.state_size,))]
        for state_layer in state_weights:
            critic_state_layers.append(layers.Dense(state_layer, activation='relu')(critic_state_layers[-1]))
        
        ##  - Set up pre-layers for action input
        critic_act_layers=[layers.Input(shape=(self.num_actions,))]
        for act_layer in state_weights:
            critic_act_layers.append(layers.Dense(act_layer, activation='relu')(critic_act_layers[-1]))
        
        ##  - Set up for combined layers
        state_act_out=layers.Concatenate()([critic_state_layers[-1], critic_act_layers[-1]])

        critic_layers=[state_act_out]
        for layer in weights:
            critic_layers.append(layers.Dense(layer, activation='relu')(critic_layers[-1]))

        critic=layers.Dense(1)(critic_layers[-1])
        critic_net=keras.Model(inputs=[critic_state_layers[0], critic_act_layers[0]], outputs=critic)
        return critic_net


##### Step in the world #####
    def step(self, _state):
        ##self.memory=_state
        #self.state_history.append(_state)
        state = tf.convert_to_tensor(_state)
        state=tf.expand_dims(state,0)

        sampled_actions = keras.ops.squeeze(self.actor(state))
        noise=self.ou_noise()

        # Adding noise to action
        act=sampled_actions.numpy()+noise
        return np.clip(act,-1,1)

    def recordStep(self, prev_state, act, reward, state, done=0):
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

    def saveBuffer(self):
        np.savez(self.buffer_filename, 
            state_buffer=self.state_buffer, action_buffer=self.action_buffer,
            reward_buffer=self.reward_buffer, next_state_buffer=self.next_state_buffer, 
            done_buffer=self.done_buffer, buffer_counter=np.array([self.buffer_counter]))

    def setModelsDir(self, _model_dir_path):
        self.model_dir=_model_dir_path

##### Learning #####
    def getBatch(self):
      ## Get sampling range
        record_range=min(self.buffer_counter, self.buffer_capacity)
        
      ## Randomly sample indices
        batch_indices=np.random.choice(record_range, self.batch_size)

      ## Convert to tensors
        state_batch=keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch=keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch=keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch=keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch=keras.ops.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch=keras.ops.convert_to_tensor(np.ones(self.batch_size)-self.done_buffer[batch_indices])
        done_batch=keras.ops.cast(done_batch, dtype="float32")
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        #print("UPDATING")
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions=self.actor_target(next_state_batch, training=True)
            done_batch*self.critic_target([next_state_batch, target_actions], training=True)
            y=reward_batch+self.gamma*done_batch*self.critic_target([next_state_batch, target_actions], training=True)
            critic_value=self.critic([state_batch, action_batch], training=True)
            critic_loss=keras.ops.mean(keras.ops.square(y-critic_value))
        critic_grad=tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions=self.actor(state_batch, training=True)
            critic_value=self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss=-keras.ops.mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        return actor_loss, critic_loss
        #actor_target_temp=keras.models.clone_model(self.actor_target, self.actor_target.get_weights())
        #critic_target_temp=keras.models.clone_model(self.critic_target, self.critic_target.get_weights())

    def updateTarget(self, target, original):
        ## Update target parameters
        target_weights=target.get_weights()
        original_weights=original.get_weights()
        
        for i in range(len(target_weights)):
            target_weights[i]=original_weights[i]*self.tau+target_weights[i]*(1-self.tau)
        #target.set_weights(target_weights)
        return(target_weights)

    def learn(self):
        #print("LEARNING")
        #weights_init=self.critic.get_weights()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch=self.getBatch()
        actor_loss, critic_loss=self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        #print("\n\n",critic_loss, y, critic_value)
        #weight_diff=0
        #weights_final=self.critic.get_weights()
        #for i in range(len(weights_final)):
        #    weight_diff+=np.sum(abs(np.array(weights_init[i])-np.array(weights_final[i])))
        #print("WEIGHT DIFFERENCE: ", weight_diff)
        #print(np.shape(weights_final))
        #print(np.sum(weights_init-weights_final))
        #print("UPDATED")
        actor_target_new_weights=self.updateTarget(self.actor_target, self.actor)
        critic_target_new_weights=self.updateTarget(self.critic_target, self.critic)
        self.actor_target.set_weights(actor_target_new_weights)
        self.critic_target.set_weights(critic_target_new_weights)
        return actor_loss, critic_loss


##### Exploit #####
    def act(self, _state):
        state = tf.convert_to_tensor(_state)
        state=tf.expand_dims(state,0)
        
        sampled_actions=keras.ops.squeeze(self.actor_target(state))
        act=sampled_actions.numpy()
        return act

    def saveNets(self, actor_filename=None, actor_target_filename=None, critic_filename=None, critic_target_filename=None):
        self.actor.save(self.actor_filename) if actor_filename==None else self.actor.save(actor_filename)
        self.actor_target.save(self.actor_target_filename) if actor_target_filename==None else self.actor_target.save(actor_target_filename)
        self.critic.save(self.critic_filename) if critic_filename==None else self.critic.save(critic_filename) 
        self.critic_target.save(self.critic_target_filename) if critic_target_filename==None else self.critic_target.save(critic_target_filename)

    def viewModels(self):
        plot_model(self.actor, to_file=self.model_dir+'/'+'actor_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(self.actor_target, to_file=self.model_dir+'/'+'actor_target_plot.png', show_shapes=True, show_layer_names=True)

        plot_model(self.critic, to_file=self.model_dir+'/'+'critic_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(self.critic_target, to_file=self.model_dir+'/'+'critic_target_plot.png', show_shapes=True, show_layer_names=True)