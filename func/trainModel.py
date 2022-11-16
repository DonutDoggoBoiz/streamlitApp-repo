import streamlit as st
import altair as alt
import datetime
import time
from deta import Deta
##### ------------------------------ #####
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
##### ------------------------------ #####


##### ------------ REPLAY BUFFER OBJECT ------------ #####
class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):  # to store experience into memory
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):     # to randomly select experience from memory
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)  # randomly select an array of 'batch_size' indexes from 'max_mem' indexes pool

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

##### ------------ BUILD DQN FUNCTION ------------ #####
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):   # this function will be call when we want to create a neural network
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation='linear')])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

  
##### ------------ BUILD DQN FUNCTION ------------ #####
class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end  ### ***change word to eps_min
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:  # exploration
            action = np.random.choice(self.action_space)
        else:  # exploitation
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self): 
        if self.memory.mem_cntr < self.batch_size:
            return
          
        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)
        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min  # 1.00 ---> eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        
##### ------------------------------ #####
def train_model(ag_train_prices,
               ag_name,
               ag_gamma,
               ag_eps,
               ag_eps_dec,
               ag_eps_min,
               ag_lr,
               ag_ini_bal,
               ag_trade_size,
               ag_com_fee,
               ag_train_episode):
    ### --- environment parameters
    action_space = 2      # consist of 0(Sell) , 1(Buy)
    window_size = 5      # n-days of prices used as observation or state
    n_episodes = ag_train_episode      # 10ep use around 6 mins

    train_prices = ag_train_df

    ### --- trading parameters
    initial_balance = ag_ini_bal
    trading_size_pct = ag_trade_size
    commission_fee_pct = ag_com_fee
    trade_size = (trading_size_pct/100) * initial_balance
    commission_fee = (commission_fee_pct/100) * 1.07

    ### --- episodic History
    all_acc_reward_history = []
    all_balance_history = []
    all_eps_history = []
    
    ### --- History dict
    acc_reward_history_dict = {}
    action_history_dict = {}
    trade_exposure_history_dict = {}
    account_balance_history = {}

    agent = Agent(gamma=ag_gamma, 
        epsilon=ag_eps, 
        epsilon_dec=ag_eps_dec,
        lr=ag_lr, 
        input_dims=window_size,
        n_actions=action_space, 
        mem_size=1000000, 
        batch_size=32,
        epsilon_end=ag_eps_min)

    ## --- loop through episodes
    for i in range(n_episodes):
        ### --- start episode --- ###
        #print ("---------- Episode " + str(i+1) + " / " + str(n_episodes) + ' ----------' )
        st.write("--- Episode " + str(i+1) + " / " + str(n_episodes) + ' ---' )

        # slider window
        start_tick = window_size
        end_tick = len(train_prices) - 2 
        current_tick = start_tick
        done = False

        # bundle train_prices data into state and new_state
        state = train_prices[ (current_tick - window_size) : current_tick ]
        new_state = train_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

        # initiate episodial variables
        acc_reward_history = []
        action_history = []
        trade_exposure_history = []
        account_balance_history = []
        nom_return_history = []
        real_return_history = []

        acc_reward = 0
        account_balance = initial_balance
        trade_exposure = False
        trade_exposure_ledger = []
        last_buy = []
########
        while not done:
            action = agent.choose_action(state)

            if action == 1: # buy
                reward = train_prices[current_tick+1] - train_prices[current_tick]
                acc_reward += reward
                if trade_exposure == False:
                    last_buy.append(train_prices[current_tick])
                    trade_exposure = True 

            elif action == 0: # sell
                reward = train_prices[current_tick] - train_prices[current_tick+1]
                acc_reward += reward
                if trade_exposure == True:
                  return_pct = (train_prices[current_tick] - last_buy[-1]) / last_buy[-1]
                  market_value = (return_pct+1) * trade_size
                  nom_return = return_pct * trade_size
                  real_return = (return_pct * trade_size) - (market_value * commission_fee) - (trade_size * commission_fee)
                  account_balance += real_return
                  nom_return_history.append([int(current_tick),nom_return])
                  real_return_history.append([int(current_tick),real_return])
                  trade_exposure = False

            done = True if current_tick == end_tick else False

            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()

            current_tick += 1
            state = new_state
            new_state = train_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

            # append history lists
            acc_reward_history.append(acc_reward)
            action_history.append(action)
            trade_exposure_history.append(trade_exposure)
            account_balance_history.append(account_balance)

            ### --- end of 1 episode --- ###
            if done: 
                st.write("---Episode {} of {} done...".format(i+1, n_episodes) )
                st.write("---Total Reward: {:.2f} | Account_Balance: {:.2f}".format(acc_reward, account_balance) )

                acc_reward_history_dict['episode_'+str(i+1)] = acc_reward_history
                action_history_dict['episode_'+str(i+1)] = action_history
                trade_exposure_history_dict['episode_'+str(i+1)] = trade_exposure_history
                account_balance_history['episode_'+str(i+1)] = account_balance_history

                all_acc_reward_history.append([(i+1),acc_reward])
                all_balance_history.append([(i+1),account_balance])
                all_eps_history.append([(i+1)agent.epsilon])
                ### --- start next episode --- ###
        
