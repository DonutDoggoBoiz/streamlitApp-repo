# --- IMPORT LIBRARY
import numpy as np
import pandas as pd
import yfinance as yf
  # from dqn_object import Agent
import streamlit as st
# ------------------------

## --- parameters setting ---
def set_parameters():
  ### --- environment parameters
  action_space = 2      # consist of 0(Sell) , 1(Buy)
  window_size = 5      # n-days of prices used as observation or state
  x_episodes = 5      # 10ep use around 6 mins
  
  ### --- trading parameters
  initial_balance = 1000000
  trading_size_pct = 10
  commission_fee_pct = 0.157
  trade_size = (trading_size_pct/100) * initial_balance
  commission_fee = (commission_fee_pct/100) * 1.07

  ### --- episodic History
  eval_total_acc_reward_history = []
  eval_end_balance_history = []
  eval_eps_history = []

  ### --- trading History
  eval_acc_reward_history = []
  eval_action_history = []
  eval_account_balance_history = []
  eval_nom_return_history = []
  eval_real_return_history = []

# --- EVALUATING MODULE ---
## --- load trained model ---
def load_model():
  pass

def eval_model():
## --- loop through x episodes
  for i in range(x_episodes):
      ### --- start episode --- ###
      print ("Episode " + str(i+1) + "/" + str(x_episodes) )

      # slider window
      start_tick = window_size
      end_tick = len(train_prices) - 2 
      current_tick = start_tick
      ##last_trade_tick = current_tick - 1
      done = False

      # bundle train_prices data into state and new_state
      state = train_prices[ (current_tick - window_size) : current_tick ]
      new_state = train_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

      # initiate episodial variables
      acc_reward = 0
      account_balance = initial_balance
      trade_exposure = False
      trade_exposure_ledger = []
      last_buy = []

      while not done:
        pred_action = agent.q_eval.predict(np.array([state]))
        action = np.argmax(pred_action)

        if action == 1: # buy
            reward = train_prices[current_tick+1] - train_prices[current_tick]
            acc_reward += reward
            if trade_exposure == False:
              last_buy.append(train_prices[current_tick])     # memorize bought price
              account_balance -= trade_size * commission_fee  # pay fees on purchase
              trade_exposure = True 

        elif action == 0: # sell
            reward = train_prices[current_tick] - train_prices[current_tick+1]
            acc_reward += reward
            if trade_exposure == True:
              return_pct = (train_prices[current_tick] - last_buy[-1]) / last_buy[-1]   # profit/loss percentage on investment
              market_value = (return_pct+1) * trade_size                                # market value of investment
              nom_return = return_pct * trade_size
              real_return = (return_pct * trade_size) - (market_value * commission_fee)
              account_balance += real_return
              nom_return_history.append([int(current_tick),nom_return])
              real_return_history.append([int(current_tick),real_return])
              trade_exposure = False

        done = True if current_tick == end_tick else False

        # agent.store_transition(state, action, reward, new_state, done)
        # agent.learn()

        current_tick += 1
        state = new_state
        new_state = train_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

        # append history lists
        eval_total_acc_reward_history.append(acc_reward)
        eval_end_balance_history.append(account_balance)
        eval_eps_history.append(agent.epsilon)

        if done: 
          # print ("-----------------------------------------")
          print ("Total Reward: {:.2f} , Account_Balance: {:2f}".format(acc_reward, account_balance) )
          print ("-----------------------------------------")
        ### --- end of 1 episode --- ###

      total_acc_reward_history.append(acc_reward)
      end_balance_history.append(account_balance)
      eps_history.append(agent.epsilon)
      
   # --- reshape history data to array ---
def reshape_history(): 
  record_num = np.array(eval_action_history).shape[0]
  np_eval_acc_reward_history = np.reshape( np.array(eval_acc_reward_history) , ( int(x_episodes) , int(record_num/x_episodes) ) )
  np_eval_action_history = np.reshape( np.array(eval_action_history) , ( int(x_episodes) , int(record_num/x_episodes) ) )
  np_eval_account_balance_history = np.reshape( np.array(eval_account_balance_history) , ( int(x_episodes) , int(record_num/x_episodes) ) )
  # --- print shape of history arrays ---
  print('np_eval_acc_reward_history.shape: {}'.format(np_eval_acc_reward_history.shape))
  print('np_eval_action_history.shape: {}'.format(np_eval_action_history.shape))
  print('np_eval_account_balance_history.shape: {}'.format(np_eval_account_balance_history.shape))
  
def show_eval_metric():
  print("Start price: "+ str(train_prices[0]) )
  print("End price: "+ str(train_prices[-1]) )
  print("Avg price: "+ str(np.mean(train_prices)))
  print("----------------------------------------")
  print("HODL return rate: {:+.4f}".format( ((train_prices[-1]) - (train_prices[0]))/(train_prices[0]) ) )
  print("DCA return rate: {:+.4f}".format( ((train_prices[-1]) - (np.mean(train_prices)))/(np.mean(train_prices)) ) )
  print("----------------------------------------")
  print("AGENT return rate: {:+.4f}".format( (np_eval_account_balance_history[0,-1] - initial_balance)/initial_balance ) )
