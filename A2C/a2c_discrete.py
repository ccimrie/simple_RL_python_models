import numpy as np
import math
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
# from tensorflow import compat
# import tensorflow_probability as tfp
import random
# from collections import deque
import os
import yaml
os.environ["KERAS_BACKEND"] = "tensorflow"

## TODO: Update so it works with latest version of TF and Keras

class DiscreteA2C:
 ##### Setting up RL class #####
    def __init__(self, rl_yaml):
        seed=np.random.randint(0,5e6)
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
        self.entropy_beta=rl_setup['entropy']
        ##TODO Investigate if target networks are useful for A2C models?
        #self.tau=rl_setup['tau']

        self.model_dir=rl_setup['model dir']
        self.actor_filename=self.model_dir+'/'+actor['filename']
        self.critic_filename=self.model_dir+'/'+critic['filename']
        # self.actor_target_filename=self.model_dir+'/'+actor['target filename']
        # self.critic_target_filename=self.model_dir+'/'+critic['target filename']

     ## Create/Load RL models
        def instantiateModel(filename, weights, output):
            if os.path.exists(filename):
                return keras.models.load_model(filename)
            return self.createModel(weights, output)

        self.actor=instantiateModel(self.actor_filename, actor['weights'], actor['output'])
        self.critic=instantiateModel(self.critic_filename, critic['weights'], critic['output'])
        self.viewModels()

        ## Memory
        self.state_history=[]
        self.action_history=[]
        self.entropy_history=[]
        self.value_pred_history=[]
        self.reward_history=[]
        print("Initialised RL agent")

######## Creating DNN models for DRL #######
    def createModel(self, weights, output):
        def layerBlock(block):
            block_layers=[layers.Input(shape=(int(block['input size']),))]
            for layer in block['weights']:
                block_layers.append(layers.Dense(layer, activation='relu')(block_layers[-1]))
            return block_layers
        network_layers=[]
        network_input_layers=[]
     ## Check if more than one layer block
        if len(weights)>1:
            block_names=weights.keys()
            for block_name in block_names:
                if not block_name=='combined':
                    network_layers.append(layerBlock(weights[block_name]))
            input_layers=[layer[0] for layer in network_layers]
            network_layers=[layers.Concatenate()([layer[-1] for layer in network_layers])]
        else:
            input_size=weights['combined']['input size']
            network_layers=[layers.Input(shape=(input_size,))]
        
        for layer in weights['combined']['weights']:
            network_layers.append(layers.Dense(layer, activation='relu')(network_layers[-1]))

     ## Initialize weights between -3e-3 and 3-e3
     ##  - original set up only applied to actor; okay for critic as well?
     ##  - Shouldn't this be applied to all weights and not just last layer? Is this what is this doing?
        last_init=keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        network_inputs=network_input_layers if len(network_input_layers)>0 else network_layers[0]
        output_layer_size=output['size']
        activation_function=output['activation function'] if 'activation function' in output else None
        network_outputs=layers.Dense(output_layer_size, activation=activation_function, kernel_initializer=last_init)(network_layers[-1])
        model_net=keras.Model(inputs=network_inputs, outputs=network_outputs)
        return model_net

####### Step in the world ######
    ## State needs to be tensor
    def tdTarget(self, reward, next_state):
        value=self.critic(next_state)[0,0]
        y_target=self.reward+self.gamma*value
        return y_target

    ## Assume that the _state is constructed to match what network expects as defined in YAML file
    def step(self, _state):
     ## Convert state to tensor
        if any([isinstance(state_component, list ) for state_component in _state]):
            state=[keras.ops.convert_to_tensor(state_component) for state_component in _state]
        else:
            state=keras.ops.convert_to_tensor(_state)
            state=tf.expand_dims(state,0)
        self.state_history.append(state)
     
     ## Get action probabilities
        act_out=keras.ops.squeeze(self.actor(state))
      ## Ensure sum(probs)==1
        act_probs=act_out/np.sum(act_out)

        act_probs_filter=act_probs[act_probs>0]
        info_gains=[-(1.0/act_prob)*np.log(act_prob) for act_prob in act_probs]
        entropy=np.sum(info_gains)
        self.entropy_history.append(entropy)

        p_sum=0
        p=np.random.rand()
        for ind in np.arange(len(act_probs)):
            if p<(act_probs[ind]+p_sum):
                act=ind
                break
            else:
                p_sum+=act_probs[ind]

        ## Get value from critic network
        value=self.critic(state)[0,0]

        ## Record values
        try:
            self.action_history.append(act)
        except Exception as e:
            print(f"Problem: {_state}  {act_out}  {act_probs}  {np.sum(act_probs)}")
        self.value_pred_history.append(value)

        return act, act_out

    def recordReward(self, reward):
        self.reward_history.append(reward)

###### Learning #######

    def actorLossFunc(self,log_prob, adv):
        loss=keras.ops.convert_to_tensor(-log_prob*adv)
        return loss

    def calcValue(self):
        rets=[]
        discounted_sum=0
        for r in self.reward_history[::-1]:
            discounted_sum=r+self.gamma*discounted_sum
            rets.insert(0, discounted_sum)
        return rets

    ## Need to be done before critic learning due to TD
    #@tf.function
    def learnActor(self):
        print(f"LEARNING ACTOR")
        returns_true=self.calcValue()
        history=zip(self.state_history, self.action_history, self.value_pred_history, returns_true)
        with tf.GradientTape() as tape:
            actor_loss=0
            for state, act, value, ret in history:
                act_probs=keras.ops.squeeze(self.actor(state))
                ## Get action log prob
                p=act_probs[act] if act_probs[act]>0 else act_probs[act]+1e-32
                log_prob=keras.ops.log(act_probs[act])
                adv=ret-value
                actor_loss+=(-log_prob*adv)
                if np.isinf(log_prob):
                    print(f"Probability: {act}  {act_probs}  {act_probs/np.sum(act_probs)}")
                # print(act, adv, log_prob, actor_loss)
                    print(f"Act:  {act}\nAdv:  {adv}\nLog probability:  {log_prob}\nActor loss:  {actor_loss}\n\n")
        print(f"Actor loss:  {actor_loss}")
        grads=tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    ## Needs next state
    #@tf.function
    def learnCritic(self):
        print(f"LEARNING CRITIC")
        returns_true=self.calcValue()
        history=zip(self.state_history, returns_true, self.entropy_history)
        entropy_samples=len(self.entropy_history)
        with tf.GradientTape() as tape:
            critic_loss=0
            for state, ret, entropy in history:
                state=keras.ops.convert_to_tensor(state)
                value=self.critic(state)[0,0]
                critic_loss=critic_loss+self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))-(self.entropy_beta*entropy*1.0/entropy_samples)
        print(f"Critic loss:  {critic_loss}")
        grads=tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def clearHistory(self):
        self.state_history=[]
        self.action_history=[]
        self.value_pred_history=[]
        self.reward_history=[]

##### Exploit ######
    def act(self, _state):
        if any([isinstance(state_component, list ) for state_component in _state]):
            state=[keras.ops.convert_to_tensor(state_component) for state_component in _state]
        else:
            state=keras.ops.convert_to_tensor(_state)
            state=tf.expand_dims(state,0)

        act_out=keras.ops.squeeze(self.actor(state))
        print(act_out)
        act=np.argmax(act_out)
        return act

    def saveNets(self, actor_filename=None, actor_target_filename=None, critic_filename=None, critic_target_filename=None):
        self.actor.save(self.actor_filename) if actor_filename==None else self.actor.save(actor_filename)
        self.critic.save(self.critic_filename) if critic_filename==None else self.critic.save(critic_filename) 

    def viewModels(self):
        plot_model(self.actor, to_file=self.model_dir+'/'+'actor_plot.png', show_shapes=True, show_layer_names=True)
        plot_model(self.critic, to_file=self.model_dir+'/'+'actor_critic_plot.png', show_shapes=True, show_layer_names=True)
