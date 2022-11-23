import streamlit as st
import altair as alt
##### ---------------------------------------- #####
from deta import Deta
##### ---------------------------------------- #####
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
##### ---------------------------------------- #####

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
                mem_size=1000000, fname='dqn_model'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end  ### ***change word to eps_min
        self.batch_size = batch_size
        self.model_file_name = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:  # exploration
            action = np.random.choice(self.action_space)
        else:  # exploitation
            state = np.array([observation])
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)
        return action

    def learn(self): 
        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)
        q_eval = self.q_eval.predict(states, verbose=0)
        q_next = self.q_eval.predict(states_, verbose=0)
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)
        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min  # 1.00 ---> eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
########## ---------------DETA_FUNCTION--------------- ##########
def deta_update_train(username, deta_key):
    deta = Deta(deta_key)
    model_db = deta.Base("model_db")
    model_frame = pd.DataFrame(model_db.fetch([{'username':username},{'model_name':agent.model_file_name}]).items)
    key_to_update = model_frame['key'].to_list()[0]
    update_dict = {'episode_trained':eps_trained,
                   'trained_result':result_train_pl}
    model_db.update(updates=update_dict, key=key_to_update)
    
def deta_update_test(username, deta_key):
    deta = Deta(deta_key)
    model_db = deta.Base("model_db")
    model_frame = pd.DataFrame(model_db.fetch({'username':username},{'model_name':agent.model_file_name}).items)
    key_to_update = model_frame['key'].to_list()[0]
    update_dict = {'test_result':result_test_pl}
    model_db.update(updated=update_dict, key=key_to_update)
########## ---------------TRAIN_MODEL--------------- ##########
        
########## ---------------TRAIN_MODEL--------------- ##########
def train_model(ag_df_price_train,
               ag_name,
               ag_gamma,
               ag_eps,
               ag_eps_dec,
               ag_eps_min,
               ag_lr,
               ag_ini_bal,
               ag_trade_size_pct,
               ag_com_fee_pct,
               ag_train_episode):
    global agent, eps_trained, result_train_pl
    ### --- environment parameters
    action_space = 2      # consist of 0(Sell) , 1(Buy)
    window_size = 5      # n-days of prices used as observation or state
    n_episodes = ag_train_episode      # 10ep use around 6 mins

    train_prices = ag_df_price_train['Close'].to_numpy()

    ### --- trading parameters
    initial_balance = ag_ini_bal
    trading_size_pct = ag_trade_size_pct
    commission_fee_pct = ag_com_fee_pct
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
    account_balance_history_dict = {}
    net_pl_history_dict = {}
    net_pl_pct_history_dict = {}

    agent = Agent(gamma=ag_gamma,
                  epsilon=ag_eps,
                  epsilon_dec=ag_eps_dec,
                  lr=ag_lr,
                  input_dims=window_size,
                  n_actions=action_space,
                  mem_size=1000000,
                  batch_size=32,
                  epsilon_end=ag_eps_min,
                  fname=ag_name)
    
    train_log_expander = st.expander('Training Logs',expanded=True)
    ## --- loop through episodes
    for i in range(n_episodes):
        ### --- start episode --- ###
        with train_log_expander:
            st.write("--- Episode " + str(i+1) + " / " + str(n_episodes) + ' training...')

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
        net_pl_history = []
        net_pl_pct_history = []
        
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
            net_pl_history.append(account_balance-initial_balance)
            net_pl_pct_history.append((100*(account_balance-initial_balance))/initial_balance)

            ### --- end of 1 episode --- ###
            if done:
                with train_log_expander:
                    #st.write("--- Episode {} of {} done | Total Reward: {:.2f} | Account_Balance: {:,.2f} THB | Profit/Loss: {:+,.2f} THB".format(
                        #i+1, n_episodes,acc_reward, account_balance, account_balance-initial_balance))
                    st.write("--- Episode {} of {} done | Total Reward: {:.2f} | Profit/Loss: {:+,.2f} THB".format(
                        i+1, n_episodes,acc_reward, account_balance-initial_balance))

                acc_reward_history_dict['episode_'+str(i+1)] = acc_reward_history
                action_history_dict['episode_'+str(i+1)] = action_history
                trade_exposure_history_dict['episode_'+str(i+1)] = trade_exposure_history
                account_balance_history_dict['episode_'+str(i+1)] = account_balance_history
                net_pl_history_dict['episode_'+str(i+1)] = net_pl_history
                net_pl_pct_history_dict['episode_'+str(i+1)] = net_pl_pct_history

                all_acc_reward_history.append([(i+1),acc_reward])
                all_balance_history.append([(i+1),account_balance])
                all_eps_history.append([(i+1),agent.epsilon])
                ### --- start next episode --- ###
    ### --- end of training --- ###
    st.success('Training DONE!')
    ################################################
    st.write('#####     ----- Train Result (last episode) -----')
    st.write('Reward History')
    acc_reward_history_df = pd.DataFrame(acc_reward_history_dict, index=ag_df_price_train[5:-1].index)
    alt_acc_reward = alt.Chart(acc_reward_history_df.iloc[:,-1].reset_index()
                              ).encode(x = alt.X('Date'),
                                       y = alt.Y(acc_reward_history_df.columns[-1], 
                                                 title='Reward (pts)', 
                                                 scale=alt.Scale(domain=[acc_reward_history_df.iloc[:,-1].min()-2,
                                                                         acc_reward_history_df.iloc[:,-1].max()+2])),
                                       tooltip=[alt.Tooltip('Date', title='Date'),
                                                alt.Tooltip(acc_reward_history_df.columns[-1], title='Reward (pts)')]
                                      )
    st.altair_chart(alt_acc_reward.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                    use_container_width=True)
    ################################################
    st.write('Account Balance History')
    account_balance_history_df = pd.DataFrame(account_balance_history_dict, index=ag_df_price_train[5:-1].index)
    alt_acc_bal_hist = alt.Chart(account_balance_history_df.iloc[:,-1].reset_index()
                                ).encode(x = alt.X('Date'),
                                         y = alt.Y(account_balance_history_df.columns[-1],
                                                   title='Account Balance (THB)',
                                                   scale=alt.Scale(domain=[account_balance_history_df.iloc[:,-1].min()-10000,
                                                                           account_balance_history_df.iloc[:,-1].max()+10000])),
                                         tooltip=[alt.Tooltip('Date', title='Date'),
                                                  alt.Tooltip(account_balance_history_df.columns[-1], title='Account Balance (THB)')]
                                        )
    st.altair_chart(alt_acc_bal_hist.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                    use_container_width=True)
    ################################################
    st.write('Net Profit/Loss History')
    net_pl_history_df = pd.DataFrame(net_pl_history_dict, index=ag_df_price_train[5:-1].index)
    alt_net_pl_hist = alt.Chart(net_pl_history_df.iloc[:,-1].reset_index()
                               ).encode(x = alt.X('Date'),
                                        y = alt.Y(net_pl_history_df.columns[-1],
                                                   title='Profit/Loss (THB)',
                                                   scale=alt.Scale(domain=[net_pl_history_df.iloc[:,-1].min()-10000,
                                                                           net_pl_history_df.iloc[:,-1].max()+10000])),
                                         tooltip=[alt.Tooltip('Date', title='Date'),
                                                  alt.Tooltip(net_pl_history_df.columns[-1], title='Profit/Loss (THB)')]
                                        )
    st.altair_chart(alt_net_pl_hist.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                    use_container_width=True)
    ################################################
    st.write('Net Profit/Loss (%) History')
    net_pl_pct_history_df = pd.DataFrame(net_pl_pct_history_dict, index=ag_df_price_train[5:-1].index)
    alt_net_pl_pct_hist = alt.Chart(net_pl_pct_history_df.iloc[:,-1].reset_index()
                               ).encode(x = alt.X('Date'),
                                        y = alt.Y(net_pl_pct_history_df.columns[-1],
                                                   title='Profit/Loss (%)',
                                                   scale=alt.Scale(domain=[net_pl_pct_history_df.iloc[:,-1].min()-2,
                                                                           net_pl_pct_history_df.iloc[:,-1].max()+2])),
                                         tooltip=[alt.Tooltip('Date', title='Date'),
                                                  alt.Tooltip(net_pl_pct_history_df.columns[-1], title='Profit/Loss (%)')]
                                        )
    st.altair_chart(alt_net_pl_pct_hist.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                    use_container_width=True)
    ################################################
    eps_trained = n_episodes
    result_train_pl = account_balance_history_dict['episode_'+str(n_episodes)][-1] - initial_balance
#END###### ---------------TRAIN_MODEL--------------- ##########

########## ---------------TEST_MODEL--------------- ##########
def test_model(ag_df_price_test,
               ag_name,
               ag_gamma,
               ag_eps,
               ag_eps_dec,
               ag_eps_min,
               ag_lr,
               ag_ini_bal,
               ag_trade_size_pct,
               ag_com_fee_pct,
               ag_train_episode):
    global result_test_pl
    ### --- environment parameters
    action_space = 2      # consist of 0(Sell) , 1(Buy)
    window_size = 5      # n-days of prices used as observation or state
    n_episodes = 1

    test_prices = ag_df_price_test['Close'].to_numpy()

    ### --- trading parameters
    initial_balance = ag_ini_bal
    trading_size_pct = ag_trade_size_pct
    commission_fee_pct = ag_com_fee_pct
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
    account_balance_history_dict = {}
    net_pl_history_dict = {}
    net_pl_pct_history_dict = {}
    
    agent_check = False
    
    try:
        agent_check_state = test_prices[ 0 : 5 ]
        agent.q_eval.predict(np.array([agent_check_state]), verbose=0)
        agent_check = True
    except:
        st.error('No model detected.')
        st.warning('Please train your model before testing it.')
    
    if agent_check == True:
        test_log_expander = st.expander('Testing Logs',expanded=True)
        ## --- loop through episodes
        for i in range(n_episodes):
            with test_log_expander:
                st.write("--- Episode " + str(i+1) + " / " + str(n_episodes) + ' ---' )

            # slider window
            start_tick = window_size
            end_tick = len(test_prices) - 2 
            current_tick = start_tick
            done = False

            # bundle test_prices data into state and new_state
            state = test_prices[ (current_tick - window_size) : current_tick ]
            new_state = test_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

            # initiate episodial variables
            acc_reward_history = []
            action_history = []
            trade_exposure_history = []
            account_balance_history = []
            nom_return_history = []
            real_return_history = []
            net_pl_history = []
            net_pl_pct_history = []

            acc_reward = 0
            account_balance = initial_balance
            trade_exposure = False
            trade_exposure_ledger = []
            last_buy = []
    ########
            while not done:
                pred_action = agent.q_eval.predict(np.array([state]), verbose=0)
                action = np.argmax(pred_action)

                if action == 1: # buy
                    reward = test_prices[current_tick+1] - test_prices[current_tick]
                    acc_reward += reward
                    if trade_exposure == False:
                        last_buy.append(test_prices[current_tick])
                        trade_exposure = True
                    st.write("Step: {} | Buy at: {} | Reward: {:+,.2f} | Profit/Loss: {:+,.2f}".format(current_tick-4,
                                                                                                       test_prices[current_tick],
                                                                                                       reward,
                                                                                                       account_balance-initial_balance))

                elif action == 0: # sell
                    reward = test_prices[current_tick] - test_prices[current_tick+1]
                    acc_reward += reward
                    if trade_exposure == True:
                        return_pct = (test_prices[current_tick] - last_buy[-1]) / last_buy[-1]
                        market_value = (return_pct+1) * trade_size
                        nom_return = return_pct * trade_size
                        real_return = (return_pct * trade_size) - (market_value * commission_fee) - (trade_size * commission_fee)
                        account_balance += real_return
                        nom_return_history.append([int(current_tick),nom_return])
                        real_return_history.append([int(current_tick),real_return])
                        trade_exposure = False
                    st.write("Step: {} | Sell at: {} | Reward: {:+,.2f} | Profit/Loss: {:+,.2f}".format(current_tick-4,
                                                                                                       test_prices[current_tick],
                                                                                                       reward,
                                                                                                       account_balance-initial_balance))
                done = True if current_tick == end_tick else False

                current_tick += 1
                state = new_state
                new_state = test_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

                # append history lists
                acc_reward_history.append(acc_reward)
                action_history.append(action)
                trade_exposure_history.append(trade_exposure)
                account_balance_history.append(account_balance)
                net_pl_history.append(account_balance-initial_balance)
                net_pl_pct_history.append((100*(account_balance-initial_balance))/initial_balance)

                ### --- end of 1 episode --- ###
                if done:
                    with test_log_expander:
                        st.success('Testing DONE!')
                        st.write('Testing result')
                        st.write("--- Total Reward: {:+,.2f} | Net Profit/Loss: {:+,.2f} THB".format(acc_reward,
                                                                                                 account_balance-initial_balance))

                    acc_reward_history_dict['episode_'+str(i+1)] = acc_reward_history
                    action_history_dict['episode_'+str(i+1)] = action_history
                    trade_exposure_history_dict['episode_'+str(i+1)] = trade_exposure_history
                    account_balance_history_dict['episode_'+str(i+1)] = account_balance_history
                    net_pl_history_dict['episode_'+str(i+1)] = net_pl_history
                    net_pl_pct_history_dict['episode_'+str(i+1)] = net_pl_pct_history

                    all_acc_reward_history.append([(i+1),acc_reward])
                    all_balance_history.append([(i+1),account_balance])
                    all_eps_history.append([(i+1),agent.epsilon])
                    ### --- start next episode --- ###
        ### --- end of training --- ###
        st.success('Testing DONE!')
        ######################################
        st.write('#####     ----- Test Result -----')
        st.write('Reward History')
        acc_reward_history_df = pd.DataFrame(acc_reward_history_dict, index=ag_df_price_test[5:-1].index)
        alt_acc_reward = alt.Chart(acc_reward_history_df.iloc[:,-1].reset_index()
                                  ).encode(x = alt.X('Date'),
                                           y = alt.Y(acc_reward_history_df.columns[-1], 
                                                     title='Reward (pts)',
                                                     scale=alt.Scale(domain=[acc_reward_history_df.iloc[:,-1].min()-2,
                                                                             acc_reward_history_df.iloc[:,-1].max()+2])),
                                           tooltip=[alt.Tooltip('Date', title='Date'),
                                                    alt.Tooltip(acc_reward_history_df.columns[-1], title='Reward (pts)')])
        st.altair_chart(alt_acc_reward.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                        use_container_width=True)
        ######################################
        st.write('Account Balance History')
        account_balance_history_df = pd.DataFrame(account_balance_history_dict, index=ag_df_price_test[5:-1].index)
        alt_acc_bal_hist = alt.Chart(account_balance_history_df.iloc[:,-1].reset_index()
                                    ).encode(x = alt.X('Date'),
                                             y = alt.Y(account_balance_history_df.columns[-1],
                                                       title='Account Balance (THB)',
                                                       scale=alt.Scale(domain=[account_balance_history_df.iloc[:,-1].min()-10000,
                                                                               account_balance_history_df.iloc[:,-1].max()+10000])),
                                             tooltip=[alt.Tooltip('Date', title='Date'),
                                                      alt.Tooltip(account_balance_history_df.columns[-1], title='Account Balance (THB)')])
        st.altair_chart(alt_acc_bal_hist.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                        use_container_width=True)
        ######################################
        st.write('Net Profit/Loss History')
        net_pl_history_df = pd.DataFrame(net_pl_history_dict, index=ag_df_price_train[5:-1].index)
        alt_net_pl_hist = alt.Chart(net_pl_history_df.iloc[:,-1].reset_index()
                                   ).encode(x = alt.X('Date'),
                                            y = alt.Y(net_pl_history_df.columns[-1],
                                                       title='Profit/Loss (THB)',
                                                       scale=alt.Scale(domain=[net_pl_history_df.iloc[:,-1].min()-10000,
                                                                               net_pl_history_df.iloc[:,-1].max()+10000])),
                                             tooltip=[alt.Tooltip('Date', title='Date'),
                                                      alt.Tooltip(net_pl_history_df.columns[-1], title='Profit/Loss (THB)')]
                                            )
        st.altair_chart(alt_net_pl_hist.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                        use_container_width=True)
        ######################################
        net_pl_pct_history_df = pd.DataFrame(net_pl_pct_history_dict, index=ag_df_price_train[5:-1].index)
        alt_net_pl_pct_hist = alt.Chart(net_pl_pct_history_df.iloc[:,-1].reset_index()
                                   ).encode(x = alt.X('Date'),
                                            y = alt.Y(net_pl_pct_history_df.columns[-1],
                                                       title='Profit/Loss (%)',
                                                       scale=alt.Scale(domain=[net_pl_pct_history_df.iloc[:,-1].min()-2,
                                                                               net_pl_pct_history_df.iloc[:,-1].max()+2])),
                                             tooltip=[alt.Tooltip('Date', title='Date'),
                                                      alt.Tooltip(net_pl_pct_history_df.columns[-1], title='Profit/Loss (%)')]
                                            )
        st.altair_chart(alt_net_pl_pct_hist.mark_line().interactive().configure_axis(labelFontSize=14,titleFontSize=16),
                        use_container_width=True)
        ######################################
        result_test_pl = account_balance_history_dict['episode_1'][-1] - initial_balance
        return result_test_pl
#END###### ---------------TEST_MODEL--------------- ##########

########## ---------------SAVE_MODEL--------------- ##########
def save_model_local(save_username):
    try:
        path = 'model/'+str(save_username)+'_'+str(agent.model_file_name)+'.h5'
        agent.q_eval.save(path)
        st.success('Save model on local DONE!')
    except:
        st.error('ERROR: save_model_local')
#END###### ---------------SAVE_MODEL--------------- ##########

########## ---------------GENERATE_ADVICE--------------- ##########
def generate_advice(ag_df_price_advice,
                    save_username,
                    ag_name,
                    ag_quote):
    ### --- Price Data
    advice_prices = ag_df_price_advice['Close'].to_numpy()
    
    ### --- Environment parameters
    action_space = 2
    window_size = 5
    adv_episodes = 1

    ### --- History Dict
    action_history_dict = {}
    position_history_dict = {}
    exposure_history_dict = {}
    
    agent = Agent(gamma=0.99,
                  epsilon=1.00,
                  epsilon_dec=0.01,
                  lr=0.001,
                  input_dims=window_size,
                  n_actions=action_space,
                  mem_size=1000000,
                  batch_size=32,
                  epsilon_end=0.01,
                  fname=ag_name)
    
    ### --- push start Agent
    push_start_state = advice_prices[ 0 : 5 ]
    push_start_pred = agent.q_eval.predict(np.array([push_start_state]), verbose=0)
    
    ### --- Load Weights
    local_path = 'model/'+str(save_username)+'_'+str(ag_name)+'.h5'
    agent.q_eval.load_weights(local_path)
####
    #####_LOOP_THROUGH_1_EPISODE_#########################
    for i in range(adv_episodes):
        # slider window
        start_tick = window_size
        end_tick = len(advice_prices)-1
        current_tick = start_tick
        done = False

        # bundle advice_prices data into first state
        state = advice_prices[ (current_tick - window_size) : current_tick ]

        # initiate episodial variables
        action_history = []
        position_history = []
        plot_history = []

        trade_exposure = False

        #####_MOVING_PRICE_WINDOW_#########################
        while not done:
            pred_action = agent.q_eval.predict(np.array([state]), verbose=0)
            action = np.argmax(pred_action)

            if action == 1: # buy
                position = 'Buy'
                if trade_exposure == False:
                    trade_exposure = True
                    plot_history.append(trade_exposure)
                elif trade_exposure == True:
                    plot_history.append(False)

            elif action == 0: # sell
                position = 'Sell'
                if trade_exposure == True:
                    plot_history.append(trade_exposure)
                    trade_exposure = False
                elif trade_exposure == False:
                    plot_history.append(trade_exposure)
                    
            #####_APPEND_HISTORY_LIST_#####
            action_history.append(action)
            position_history.append(position)
            
            if current_tick == end_tick:
                done = True
            else:
                current_tick +=1
                state = advice_prices[ (current_tick - window_size) : current_tick ]
            
            if done:
                advice_df_dict = {'Close':ag_df_price_advice[5:]['Close'].to_list(),
                                 'position':position_history,
                                  'plot':plot_history}
                advice_df = pd.DataFrame(advice_df_dict, index=ag_df_price_advice[5:].index)
        #####_END_OF_PRICE_WINDOW_#########################
    #####_END_LOOP_##############################
    
    #####_ALTAIR_CHART_##################################################
    #####_PRICE_LINE_#####
    base = alt.Chart(advice_df.reset_index()).encode(
        x = alt.X('Date'),
        y = alt.Y('Close', title='Price  (THB)',
                  scale=alt.Scale(domain=[advice_df['Close'].min()-2,
                                          advice_df['Close'].max()+2])),
        tooltip=[alt.Tooltip('Date',title='Date'),
                 alt.Tooltip('Close',title='Price (THB)')] )
    #####_ACTION_OVERLAY_#####
    base2 = alt.Chart(advice_df.reset_index()).encode(
        x = alt.X('Date'),
        y = alt.Y('Close', title='Price  (THB)',
                  scale=alt.Scale(domain=[advice_df['Close'].min()-2,
                                          advice_df['Close'].max()+2])),
        color = alt.Color('position',
                          scale=alt.Scale(domain=['Buy','Sell'],
                                          range=['green','red']),
                          legend=alt.Legend(title="Model Advice")),
        tooltip=[alt.Tooltip('Date', title='Date'),
                 alt.Tooltip('Close', title='Price (THB)'),
                 alt.Tooltip('position', title='Advice')] )
    #####_LAYERED_CHART_#####
    layer1 = base.mark_line()
    layer2 = base2.mark_circle(size=50).transform_filter(alt.FieldEqualPredicate(field='plot',equal=True))
    bundle = alt.layer(layer1,layer2).configure_axis(labelFontSize=16,titleFontSize=18)
    #####_SHOW_ADVICE_CHART_#####
    st.altair_chart(bundle, use_container_width=True)
    
    #####_ADVICE_REPORT_#####
    st.write('#### Model advice: ')
    st.write('Date: {}'.format(datetime.date.today()))
    if advice_df['position'][-1] == 'Buy':
        st.success('#### BUY {} at current price of {} THB per share'.format(ag_quote,advice_df['Close'][-1]))
    else:
        st.error('#### SELL {} at current price of {} THB per share'.format(ag_quote,advice_df['Close'][-1]))
        
                



