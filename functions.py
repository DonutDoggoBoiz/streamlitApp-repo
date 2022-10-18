# ------ IMPORT LIBRARY ------ #
import numpy as np
import pandas as pd
import yfinance as yf
from dqn_object import Agent
import streamlit as st
import altair as alt
import datetime
import time
from deta import Deta

### --- DATABASE CONNECTION --- ###
deta = Deta(st.secrets["deta_key"])
stock_db = deta.Base("stock_db")

stock_df = pd.DataFrame(stock_db.fetch().items)
stock_list = stock_df['symbol'].sort_values(ascending=True)
#stock_list_sorted = stock_list.sort()

# ---------------------------- #
# DEMO username
username = 'admin99'

### ------------ MODEL DATABASE ------------ ###
model_list = []
model1 = {'username':'admin', 'model_name':'bbl_01', 'stock_quote':'bbl'.upper()}
model2 = {'username':'admin', 'model_name':'ptt_04', 'stock_quote':'ptt'.upper()}
model_list.append(model1)
model_list.append(model2)
model_df = pd.DataFrame(model_list)

#### ------ PRICE FETCHING MODULE ------ ###
def fetch_price_data():
  global stock_name, df_price, df_length
  #stock_name = st.selectbox('Select your Stock', ('BBL', 'PTT', 'ADVANC','KBANK') )
  stock_name = st.selectbox('Select your Stock', options=stock_list, index=86)
  company_name = stock_df[stock_df['symbol']==stock_name]['company_name'].to_string(index=False)
  market_name = stock_df[stock_df['symbol']==stock_name]['market'].to_string(index=False)
  industry_name = stock_df[stock_df['symbol']==stock_name]['industry'].to_string(index=False)
  sector_name = stock_df[stock_df['symbol']==stock_name]['sector'].to_string(index=False)
  st.write('{}'.format(company_name))
  st.write('Market:  {}'.format(market_name))
  st.write('Industry:  {}'.format(industry_name))
  st.write('Sector:  {}'.format(sector_name))
  int_year = int(datetime.date.today().year)
  int_last_year = int(datetime.date.today().year) - 1
  int_month = int(datetime.date.today().month)
  int_day = int(datetime.date.today().day)
  #start_date = st.date_input("Select start date: ", datetime.date(2021, 9, 20))
  #end_date = st.date_input("Select end date: ", datetime.date(2022, 9, 20))
  start_date = st.date_input("Select start date: ", datetime.date( int_last_year, int_month, int_day) )
  end_date = st.date_input("Select end date: ", datetime.date( int_year, int_month, int_day) )
  stock_code = stock_name + '.BK'
  df_price = yf.download(stock_code,
                        start=start_date,
                        end=end_date,
                        progress=True)
  df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
  df_length = df_price['Close'].count()
  #return df_price, df_length
  
def observe_price():
  #alt chart with scale
  c = (alt.Chart(df_price['Close'].reset_index()
                )
          .mark_line()
          .encode(x = alt.X('Date') ,
                  y = alt.Y('Close', title='Price', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                  tooltip=[alt.Tooltip('Date', title='Date'), 
                           alt.Tooltip('Close', title='Price')
                          ]
                 )
          .interactive()
      )
  st.altair_chart(c, use_container_width=True)

  #show split slider widget
  st.write('This dataset contains ' + str(df_length) + ' days of historical prices')
  global split_point
  split_point = st.slider('Select the split point between Train set and Test set:', 0, int(df_length), int(df_length/2))
  train_size_pct = (split_point/df_length)*100
  test_size_pct = 100-train_size_pct
  st.write('Dataset will be split into {} records of training set and {} records of test set'.format(split_point, df_length-split_point) )
  #st.write('train set will be considered as {:.2f}% of dataset while the other {:.2f}% is test set'.format(train_size_pct,test_size_pct) )
  st.write('the training set is {:.2f}% of the dataset while the test set is {:.2f}%'.format(train_size_pct,test_size_pct) )
  #return split_point
  
def split_dataset2():
  global df_price_train, df_price_test, train_prices, test_prices
  df_price['split'] = 'split'
  df_price.loc[:split_point, 'split'] = 'Train set'
  df_price.loc[split_point:, 'split'] = 'Test set'
  df_price_train = df_price[:split_point]
  df_price_test = df_price[split_point:]
  train_prices = df_price_train['Close'].to_numpy()
  test_prices = df_price_test['Close'].to_numpy()
  alt_split = alt.Chart(df_price.reset_index()).mark_line().encode(x = alt.X('Date'), 
                      y = alt.Y('Close',title='Price', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                      color = 'split' ,
                      tooltip=[alt.Tooltip('Date', title='Date'), 
                            alt.Tooltip('Close', title='Price'),
                            alt.Tooltip('split', title='Dataset')
                          ] 
                        ).interactive()
  st.write("Splited dataset")
  st.altair_chart(alt_split, use_container_width=True)
  st.success('Please proceed to "Set Parameters" tab')
  
  
### ------ MACHINE LEARNING MODULE ------ ###
## --- parameters setting --- ##
def set_parameters():
# st.title('Create DQN Trading Model 💡')
# st.sidebar.markdown('## Create Model 💡')
  #st.write("#### Set these following parameters for your trading model")
  global agent_name, agent_gamma, agent_epsilon, agent_epsilon_dec, agent_epsilon_end, agent_lr, initial_balance, trading_size_pct, commission_fee_pct
  st.write("##### Model parameters")
  agent_name = st.text_input("Model name: ", "model_01")
  agent_gamma = st.slider("Gamma: ", 0.00, 1.00, 0.90)
  agent_epsilon = st.slider("Starting epsilon (random walk probability): ", 0.00, 1.00, 1.00)
  agent_epsilon_dec = st.select_slider("Epsilon decline rate (random walk probability decline): ",
                                       options=[0.001,0.002,0.005,0.010], value=0.001)
  agent_epsilon_end = st.slider("Minimum epsilon: ", 0.01, 0.10, 0.01)
  agent_lr = st.select_slider("Learning rate: ", options=[0.001, 0.002, 0.005, 0.010], value=0.001)

  st.write("##### Trading parameters")
  initial_balance = st.number_input("Initial account balance (THB):", min_value=0, step=1000, value=1000000)
  trading_size_pct = st.slider("Trading size as a percentage of initial account balance (%):", 0, 100, 10)
  commission_fee_pct = st.number_input("Commission fee (%):", min_value=0.000, step=0.001, value=0.157, format='%1.3f')

  set_param_button = st.button("Set Parameters")
  if set_param_button:
          st.write("Your model is successfully created with these parameters...")
          st.write("##### Model parameters")
          st.write("Model name: {}".format(agent_name) )
          st.write("Gamma: {}".format(agent_gamma) )
          st.write("Starting epsilon: {}".format(agent_epsilon) )
          st.write("Epsilon decline rate: {}".format(agent_epsilon_dec) )
          st.write("Minimum epsilon: {}".format(agent_epsilon_end) )
          st.write("Learning rate: {}".format(agent_lr) )

          st.write("##### Trading parameters")
          st.write("Initial account balance:      {} ฿".format(initial_balance) )
          st.write("Trading size:                 {}%".format(trading_size_pct) )
          st.write("Commission fee:               {}%".format(commission_fee_pct) )

### ------ TRAINING MODULE ------ ###
def set_train_episodes():
    global train_episodes
    train_episodes = st.number_input('Number of training episodes:', value=2, step=1, min_value=0)
    
def train_model():
  global action_history, acc_reward_history, account_balance_history, n_episodes, window_size
  ### --- environment parameters
  action_space = 2      # consist of 0(Sell) , 1(Buy)
  window_size = 5      # n-days of prices used as observation or state
  n_episodes = train_episodes      # 10ep use around 6 mins

  ### --- trading parameters
  #initial_balance = 1000000
  #trading_size_pct = 10
  #commission_fee_pct = 0.157
  trade_size = (trading_size_pct/100) * initial_balance
  commission_fee = (commission_fee_pct/100) * 1.07

  ### --- episodic History
  total_acc_reward_history = []
  end_balance_history = []
  eps_history = []

  ### --- trading History
  acc_reward_history = []
  action_history = []
  account_balance_history = []
  nom_return_history = []
  real_return_history = []
  
  global agent
  agent = Agent(gamma=agent_gamma, 
                epsilon=agent_epsilon, 
                epsilon_dec=agent_epsilon_dec,
                lr=agent_lr, 
                input_dims=window_size,
                n_actions=action_space, 
                mem_size=1000000, 
                batch_size=32,
                epsilon_end=agent_epsilon_end)

  ## --- loop through episodes
  for i in range(n_episodes):
      ### --- start episode --- ###
      #print ("---------- Episode " + str(i+1) + " / " + str(n_episodes) + ' ----------' )
      st.write("--- Episode " + str(i+1) + " / " + str(n_episodes) + ' ---' )

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
        action = agent.choose_action(state)

        if action == 1: # buy
            reward = train_prices[current_tick+1] - train_prices[current_tick]
            acc_reward += reward
            if trade_exposure == False:
              last_buy.append(train_prices[current_tick])     # memorize bought price
              #account_balance -= trade_size * commission_fee  # pay fees on purchase
              trade_exposure = True 

        elif action == 0: # sell
            reward = train_prices[current_tick] - train_prices[current_tick+1]
            acc_reward += reward
            if trade_exposure == True:
              return_pct = (train_prices[current_tick] - last_buy[-1]) / last_buy[-1]   # profit/loss percentage on investment
              market_value = (return_pct+1) * trade_size                                # market value of investment
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
        account_balance_history.append(account_balance)

        if done: 
          # print ("-----------------------------------------")
          #print ("Total Reward: {:.2f} , Account_Balance: {:2f}".format(acc_reward, account_balance) )
          #print ("-----------------------------------------")
          st.write("---Episode {} of {} done...".format(i+1, n_episodes) )
          st.write("---Total Reward: {:.2f} | Account_Balance: {:.2f}".format(acc_reward, account_balance) )
        ### --- end of 1 episode --- ###

          total_acc_reward_history.append(acc_reward)
          end_balance_history.append(account_balance)
          eps_history.append(agent.epsilon)
      
  ### --- end of all episodes --- ###
def train_result():
  record_num = np.array(action_history).shape[0]
  np_acc_reward_history = np.reshape( np.array(acc_reward_history) , ( int(record_num/n_episodes) , int(n_episodes) ) )
  np_account_balance_history = np.reshape( np.array(account_balance_history) , ( int(record_num/n_episodes) , int(n_episodes) ) )
  st.write('Reward History of last episode')
  st.line_chart(np_acc_reward_history) #[-1]
  train_date_col = df_price_train.reset_index()[window_size:len(df_price_train)-1]['Date']
  reward_history_df = pd.DataFrame(np_acc_reward_history, index=train_date_col)
  reward_history_df = reward_history_df.reset_index()
  #st.write('acc reward shape: {}'.format(np_acc_reward_history.shape) )
  #st.write('train date col len: {}'.format(len(train_date_col)) )
  #st.dataframe(reward_history_df)
  #alt_data = pd.DataFrame({
              #'Date': reward_history_df['Date'],
              #'Episode 1': reward_history_df['0'],
              #'Episode 2': reward_history_df['1'] })
  
  #alt_chart = alt.Chart(source).transform_fold(['Episode 1', 'Episode 2'],
                                               #as_=['Episode', 'acc_reward']
                                              #).mark_line().encode(x='Date',
                                               #                    y='acc_reward',
                                                #                   color='Episode')
  #alt_reward_history = alt.Chart(reward_history_df).mark_line()
                      #.encode(x = alt.X('Date'))#, 
                      #y = alt.Y('Close', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                      #color = 'split' ,
                      #tooltip=['Date','Close','split'] ).interactive()
  #st.altair_chart(alt_chart, use_container_width=True)
  st.write('Account Balance History of last episode')
  st.line_chart(np_account_balance_history) #[-1])

  
  
# --- reshape history data to array ---
def reshape_history():
  record_num = np.array(action_history).shape[0]
  np_acc_reward_history = np.reshape( np.array(acc_reward_history) , ( int(record_num/n_episodes) , int(n_episodes) ) )
  np_action_history = np.reshape( np.array(action_history) , ( int(record_num/n_episodes) , int(n_episodes) ) )
  np_account_balance_history = np.reshape( np.array(account_balance_history) , ( int(record_num/n_episodes) , int(n_episodes) ) )
  # --- print shape of history arrays ---
  print('np_acc_reward_history.shape: {}'.format(np_acc_reward_history.shape))
  print('np_action_history.shape: {}'.format(np_action_history.shape))
  print('np_account_balance_history.shape: {}'.format(np_account_balance_history.shape))

# plot last 10 episodes of acc_reward_history
def last10_history():  # ********
  for i in range(n_episodes-10,n_episodes):
    pd.DataFrame(np_acc_reward_history[i]).plot(figsize=(6,3), title='episode'+str(i+1), legend=False)
    
    
### ------ TESTING MODULE ------ ###
def test_model():
  global x_episodes, eval_action_history, eval_acc_reward_history, eval_account_balance_history
  ### --- environment parameters
  action_space = 2      # consist of 0(Sell) , 1(Buy)
  window_size = 5      # n-days of prices used as observation or state
  x_episodes = 1      # 10ep use around 6 mins

  ### --- trading parameters
  #initial_balance = 1000000
  #trading_size_pct = 10
  #commission_fee_pct = 0.157
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

  ## --- loop through episodes
  for i in range(x_episodes):
      ### --- start episode --- ###
      #print ("---------- Episode " + str(i+1) + " / " + str(x_episodes) + ' ----------' )
      st.write("--- Episode " + str(i+1) + " / " + str(x_episodes) + ' ---' )

      # slider window
      start_tick = window_size
      end_tick = len(test_prices) - 2 
      current_tick = start_tick
      ##last_trade_tick = current_tick - 1
      done = False

      # bundle test_prices data into state and new_state
      state = test_prices[ (current_tick - window_size) : current_tick ]
      new_state = test_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

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
            reward = test_prices[current_tick+1] - test_prices[current_tick]
            acc_reward += reward
            if trade_exposure == False:
              last_buy.append(test_prices[current_tick])     # memorize bought price
              #account_balance -= trade_size * commission_fee  # pay fees on purchase
              trade_exposure = True 

        elif action == 0: # sell
            reward = test_prices[current_tick] - test_prices[current_tick+1]
            acc_reward += reward
            if trade_exposure == True:
              return_pct = (test_prices[current_tick] - last_buy[-1]) / last_buy[-1]   # profit/loss percentage on investment
              market_value = (return_pct+1) * trade_size                                # market value of investment
              nom_return = return_pct * trade_size
              real_return = (return_pct * trade_size) - (market_value * commission_fee) - (trade_size * commission_fee)
              account_balance += real_return
              eval_nom_return_history.append([int(current_tick),nom_return])
              eval_real_return_history.append([int(current_tick),real_return])
              trade_exposure = False

        done = True if current_tick == end_tick else False

        #agent.store_transition(state, action, reward, new_state, done)
        #agent.learn()

        current_tick += 1
        state = new_state
        new_state = test_prices[ (current_tick - window_size) + 1 : current_tick+1 ]

        # append history lists
        eval_acc_reward_history.append(acc_reward)
        eval_action_history.append(action)
        eval_account_balance_history.append(account_balance)

        if done: 
          # print ("-----------------------------------------")
          #print ("Total Reward: {:.2f} , Account_Balance: {:2f}".format(acc_reward, account_balance) )
          #print ("-----------------------------------------")
          st.write("---Episode {} of {} done...".format(i+1, x_episodes) )
          st.write("---Total Reward: {:.2f} | Account_Balance: {:.2f}".format(acc_reward, account_balance) )
        ### --- end of 1 episode --- ###

          eval_total_acc_reward_history.append(acc_reward)
          eval_end_balance_history.append(account_balance)
          eval_eps_history.append(agent.epsilon)
      
  ### --- end of all episodes --- ###
def test_result():
  record_num = np.array(eval_action_history).shape[0]
  np_eval_acc_reward_history = np.reshape( np.array(eval_acc_reward_history) , ( int(record_num/x_episodes) , int(x_episodes) ) )
  np_eval_account_balance_history = np.reshape( np.array(eval_account_balance_history) , ( int(record_num/x_episodes) , int(x_episodes) ) )
  st.write('Reward History of testing episode')
  st.line_chart(np_eval_acc_reward_history) #[-1])
  #alt_reward_history = alt.Chart(df_price.reset_index()).mark_line().encode(x = alt.X('Date'), 
                      #y = alt.Y('Close', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                      #color = 'split' ,
                      #tooltip=['Date','Close','split'] ).interactive()
  st.write('Account Balance History of testing episode')
  st.line_chart(np_eval_account_balance_history) #[-1])

def save_model():
  #save_path = "models/" + str(agent.model_file)
  #agent.q_eval.save('models/' + 'h5_file.h5')
  new_model = {'username':username, 'model_name':agent_name, 'stock_quote':stock_name.upper()}
  model_df.loc[len(model_df.index)] = new_model
  st.write(model_df)

  
###### --------------------
def login_form():
  ### --- DATABASE CONNECTION --- ###
  deta = Deta(st.secrets["deta_key"])
  user_db = deta.Base("user_db")

  user_frame = pd.DataFrame(user_db.fetch().items)
  user_list = user_frame['username'].values.tolist()
  password_list = user_frame['password'].values.tolist()

  ### --- SESSION STATE --- ###
  if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
  if 'username' not in st.session_state:
    st.session_state['username'] = None

  def login_func():
    st.session_state['login_status'] = True

  def logout_func():
    st.session_state['login_status'] = False

  ### --- INTERFACE --- ###
  placeholder = st.empty()

  if st.session_state['login_status'] == False:
      with placeholder.container():
          login_form = st.form('Login')
          login_form.subheader('Login 📝')
          username = login_form.text_input('Username', placeholder='your username')
          password = login_form.text_input('Password', type='password', placeholder='your password')
          if login_form.form_submit_button('Login'):
              if len(username) <= 0:
                st.warning("Please enter a username")
              elif len(password) <= 0:
                st.warning("Please enter your password")
              elif len(username) > 0 and len(password) > 0:
                if username not in user_list:
                  st.warning('User not found. Please check your username')
                else: 
                  if user_frame.loc[user_frame['username'] == username,'password'].values != password:
                    st.error("Password incorrect. Please try again")
                  else:
                    st.success("Login Successful!")
                    st.session_state['username'] = username
                    login_func()
                    time.sleep(2)
                    with placeholder.container():
                      st.write('Welcome na krub, {}'.format(username))
                      st.write('Welcome na sess, {}'.format(st.session_state['username']))
                    ### --- SIDEBAR --- ###
                    st.sidebar.write('Welcome, {}'.format(st.session_state['username']))
                    st.sidebar.button('Logout', on_click=logout_func)
                    st.sidebar.button('Reset Password')
  else:
      st.sidebar.write('Welcome, {}'.format(st.session_state['username']))
      logout_button_side = st.sidebar.button('Logout', on_click=logout_func)
      reset_pass_button_side = st.sidebar.button('Reset Password')
      with placeholder.container():
        st.write('Welcome na sess, {}'.format(st.session_state['username']))
