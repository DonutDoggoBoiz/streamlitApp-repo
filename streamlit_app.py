### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
import time
import datetime
import yfinance as yf
import altair as alt
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

from functions import fetch_price_data, observe_price, split_dataset2, set_parameters
from functions import set_train_episodes, train_model, train_result, test_model, test_result
from functions import save_model
from func.generateAdvice import generate_advice

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
if 'show_register_form' not in st.session_state:
  st.session_state['show_register_form'] = False
  
  ### --- NAV BUTTON STATUS --- ###
if 'sign_in_b_disable' not in st.session_state:
  st.session_state['sign_in_b_disable'] = True
if 'sign_up_b_disable' not in st.session_state:
  st.session_state['sign_up_b_disable'] = False

if 'model_manage_b_status' not in st.session_state:
  st.session_state['model_manage_b_status'] = False  
if 'del_mod_button_status' not in st.session_state:
  st.session_state['del_mod_button_status'] = False
if 'edit_mod_button_status' not in st.session_state:
  st.session_state['edit_mod_button_status'] = False
  
if 'model_b_status' not in st.session_state:
  st.session_state['model_b_status'] = False
if 'observe_button_status' not in st.session_state:
  st.session_state['observe_button_status'] = False
if 'split_button_status' not in st.session_state:
  st.session_state['split_button_status'] = False
if 'train_button_status' not in st.session_state:
  st.session_state['train_button_status'] = False
if 'test_button_status' not in st.session_state:
  st.session_state['test_button_status'] = False
  
if 'advice_b_status' not in st.session_state:
  st.session_state['advice_b_status'] = False
  
if 'user_manage_b_status' not in st.session_state:
  st.session_state['user_manage_b_status'] = False
### --- ^^ SESSION STATE --- ###

### -------------------- ###
def login_func():
  st.session_state['login_status'] = True

def logout_func():
  st.session_state['login_status'] = False
  st.session_state['username'] = None
  #reset_session_state
  st.session_state['show_register_form'] = False
  st.session_state['model_b_status'] = False
  st.session_state['advice_b_status'] = False
  st.session_state['user_manage_b_status'] = False
  st.session_state['observe_button_status'] = False
  st.session_state['split_button_status'] = False
  st.session_state['train_button_status'] = False
  st.session_state['test_button_status'] = False

def dis_login_button():
  st.session_state['sign_in_b_disable'] = True
  st.session_state['sign_up_b_disable'] = False
  st.session_state['show_register_form'] = False

def dis_regis_button():
  st.session_state['sign_in_b_disable'] = False
  st.session_state['sign_up_b_disable'] = True
  
def rerun():
  st.experimental_rerun()
### -------------------- ###
  
### --- NOT LOGGED IN --- ###
if st.session_state['login_status'] == False:
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    with placeholder1.container():
      ph1_col1, _, ph1_col3 = st.columns([1,5,1])
      with ph1_col1:
        login_button_ph1 = st.button('Sign in', disabled=st.session_state['sign_in_b_disable'], on_click=dis_login_button)
      with ph1_col3:
        register_button_ph1 = st.button('Sign up', disabled=st.session_state['sign_up_b_disable'], on_click=dis_regis_button)
    with placeholder2.container():
        login_form = st.form('Login')
        login_form.subheader('Login 📝')
        username = login_form.text_input('Username', placeholder='your username')
        password = login_form.text_input('Password', type='password', placeholder='your password')
        login_button = login_form.form_submit_button('Login')
        if login_button:
            if len(username) <= 0:
              st.warning("Please enter a username")
            elif len(password) <= 0:
              st.warning("Please enter your password")
            elif len(username) > 0 and len(password) > 0:
              if username not in user_list:
                #st.warning('User not found. Please check your username')
                st.error("Username or Password is incorrect. Please try again")
              else: 
                if user_frame.loc[user_frame['username'] == username,'password'].values != password:
                  st.error("Username or Password is incorrect. Please try again")
                else:
                  st.success("Login Successful!")
                  st.session_state['username'] = username
                  login_func()
                  time.sleep(4)
                  rerun()
                  
    ### --- SIGN UP BUTTON --- ###
    if register_button_ph1 or st.session_state['show_register_form']:
      st.session_state['show_register_form'] = True
      with placeholder2.container():
        register_form = st.form('Register')
        register_form.subheader('Register 📝')
        new_name = register_form.text_input('Name', placeholder='eg. Michael Burry')
        new_username = register_form.text_input('Username', placeholder='eg. mikeburr1971')
        new_password = register_form.text_input('Password', type='password', placeholder='your password')
        register_button = register_form.form_submit_button('Register')
        if register_button:
          if len(new_username) <= 0:
            st.warning("Please enter a username")
          elif len(new_password) <= 0:
            st.warning("Please enter your password")
          elif len(new_username) > 0 and len(new_password) > 0:
            if new_username not in user_list:
              user_db.put({'username':new_username, 'password':new_password})
              st.success("Register Successful!")
              st.session_state['show_register_form'] = False
              time.sleep(4)
              rerun()
            else: st.warning("Username already exists. Please enter a new username")
    if login_button_ph1:
      st.session_state['show_register_form'] = False
      rerun()
      
### --- LOGGED IN --- ###
else:
    st.sidebar.write('Welcome, {}'.format(st.session_state['username']))
    ### --- SIDEBAR --- ###
    logout_button_side = st.sidebar.button('Logout', on_click=logout_func)
    st.sidebar.write('Menu:')
    user_manage_side_b = st.sidebar.button('Manage Account', key='user_manage_side')
    manage_model_side_b = st.sidebar.button('Manage Model', key='model_manage_side')
    model_side_b = st.sidebar.button('Develop Model', key='model_side_b')
    advice_side_b = st.sidebar.button('Generate Advice', key='advice_side_b')
    
    ### --- WELCOME NOTE --- ###
    placeholder_1 = st.empty()
    with placeholder_1.container():
        st.write('### Welcome, {}'.format(st.session_state['username']))
    
     ### --- MAIN TAB BUTTON --- ###
    menuholder = st.empty()
    with menuholder.container():
      col1, _, col2, col3, col4, col5= st.columns([1,1,4,4,4,4])
      with col1:
        st.write('##### MENU:')                                          
      with col2:
        user_manage_b = st.button('Manage Account')
      with col3:
        model_manage_b = st.button('Manage Model')
      with col4:
        model_b = st.button('Develop Model')
      with col5:
        advice_b = st.button('Generate Advice', key='gen_advice_tab')
    
    placeholder_2 = st.empty()
    placeholder_3 = st.empty()
    placeholder_4 = st.empty()
    
    ### --- MANAGE ACCOUNT MENU --- ###
    if user_manage_b or user_manage_side_b or st.session_state['user_manage_b_status']:
      st.session_state['user_manage_b_status'] = True
      st.session_state['model_manage_b_status'] = False
      st.session_state['model_b_status'] = False
      st.session_state['advice_b_status'] = False
      placeholder_2.empty()
      placeholder_3.empty()
      placeholder_4.empty()
      with placeholder_2.container():
        with st.form('change_password'):
          st.write('##### Change Password')
          old_password = st.text_input('Old Password', type='password', placeholder='your old password')
          new_password = st.text_input('New Password', type='password', placeholder='your new password')
          change_pass_button = st.form_submit_button('Change Password')
          
    ### --- MANAGE MODEL MENU --- ###
    if model_manage_b or manage_model_side_b or st.session_state['model_manage_b_status']:
      st.session_state['model_manage_b_status'] = True
      st.session_state['user_manage_b_status'] = False
      st.session_state['model_b_status'] = False
      st.session_state['advice_b_status'] = False
      #######
      #st.session_state['edit_mod_button_status'] = False
      placeholder_2.empty()
      placeholder_3.empty()
      placeholder_4.empty()
      with placeholder_2.container():
        st.write('#### Model Management')
########
        datamodel_dict = {'model_name': ['bbl_01','bbl_02','ppt_05','scg_111','mint_01'],
             'gamma': [0.90,0.80,0.85,0.75,0.95],
             'learning_rate': [0.001,0.002,0.005,0.04,0.099],
             'initial_balance': [1000000,1200000,1980000,2550000,3390000],
             'trading_size': [0.10,0.25,0.15,0.30,0.50] }
        datamodel_df = pd.DataFrame(datamodel_dict)

########
        gb = GridOptionsBuilder.from_dataframe(datamodel_df)
        gb.configure_selection('single', use_checkbox=True, pre_selected_rows=[0])
        gridoptions = gb.build()
        grid_response = AgGrid(datamodel_df,
                               fit_columns_on_grid_load=True,
                               gridOptions=gridoptions)
        grp_data = grid_response['data']
        selected_row = grid_response['selected_rows'] 

        with placeholder_3.container():
          ph2col1, ph2col2, _ = st.columns([1,1,6])
          with ph2col1:
            edit_mod_button = st.button('Edit')
          with ph2col2:
            del_mod_button = st.button('Delete')
          ### --- edit button --- ###
          if edit_mod_button or st.session_state['edit_mod_button_status']:
            st.session_state['edit_mod_button_status'] = True
            placeholder_4.empty()
            with placeholder_4.container():
              with st.form('edit parameter form'):
                st.write("##### Model parameters")
                new_agent_name = st.text_input("Model name: ", "model_01")
                new_agent_gamma = st.slider("Gamma: ", 0.00, 1.00, 0.90)
                new_agent_epsilon = st.slider("Starting epsilon (random walk probability): ", 0.00, 1.00, 1.00)
                new_agent_epsilon_dec = st.select_slider("Epsilon decline rate (random walk probability decline): ",
                                                     options=[0.001,0.002,0.005,0.010], value=0.001)
                new_agent_epsilon_end = st.slider("Minimum epsilon: ", 0.01, 0.10, 0.01)
                new_agent_lr = st.select_slider("Learning rate: ", options=[0.001, 0.002, 0.005, 0.010], value=0.001)

                st.write("##### Trading parameters")
                new_initial_balance = st.number_input("Initial account balance (THB):", min_value=0, step=1000, value=1000000)
                new_trading_size_pct = st.slider("Trading size as a percentage of initial account balance (%):", 0, 100, 10)
                new_trade_size = initial_balance * trading_size_pct / 100
                #st.write('{}% of initial investment is {:,.0f} THB'.format(trading_size_pct, trade_size))
                new_commission_fee_pct = st.number_input("Commission fee (%):", min_value=0.000, step=0.001, value=0.157, format='%1.3f')
                edit_param_button = st.form_submit_button("Edit")
                if edit_param_button:
                  st.session_state['edit_mod_button_status'] = False
                  st.success('Edit parameters successful!')
                  
          ### --- delete button --- ###
          if del_mod_button or st.session_state['del_mod_button_status']:
            st.session_state['del_mod_button_status'] = True
            with placeholder_3.container():
              with st.form('del_make_sure'):
                st.write('Are you sure?')
                make_sure_radio = st.radio('Please confirm your choice:', 
                                           options=('No', 'Yes') )
                confirm_button = st.form_submit_button('Confirm')
                if confirm_button:
                  if make_sure_radio == 'Yes':
                    st.session_state['del_mod_button_status'] = False
                    st.error('Model {} has been successfully deleted'.format(selected_row[0]['model_name']))
                    time.sleep(3)
                    st.experimental_rerun()
                  elif make_sure_radio == 'No':
                    st.session_state['del_mod_button_status'] = False
                    placeholder_3.empty()
########
        try:
          placeholder_4.empty()
          with placeholder_4.container():
            with st.expander('More model information:'):
                st.write('Name : {}'.format(selected_row[0]['model_name']))
                st.write('Gamma : {:.2f}'.format(selected_row[0]['gamma']))
                st.write('Learning Rate : {:.3f}'.format(selected_row[0]['learning_rate']))
                st.write('Initial Balance : {:,} THB'.format(selected_row[0]['initial_balance']))
                st.write('Trading Size : {:.2f}%'.format(selected_row[0]['trading_size']*100))
        except:
          with placeholder_4.container():
            st.success('Loading...')
    
    ####### ---------------------- #######
    
    ### --- GENERATE ADVICE MENU --- ###
    if advice_b or advice_side_b or st.session_state['advice_b_status']:
      st.session_state['model_manage_b_status'] = False
      st.session_state['user_manage_b_status'] = False
      st.session_state['model_b_status'] = False
      st.session_state['advice_b_status'] = True
      placeholder_2.empty()
      placeholder_3.empty()
      placeholder_4.empty()
      with placeholder_2.container():
          st.markdown("### Generate Investment Advice 📈")
          selected_model = st.selectbox('Choose your model',
                                        options=['BBL_01', 'BBL_02', 'PTT_07'])
          generate_advice_button = st.button('Generate Advice')
          if generate_advice_button:
            stock_name = 'BBL'
            #start_date = datetime.date(datetime.date.today().year-1, datetime.date.today().month, datetime.date.today().day )
            start_date = ( datetime.date.today() - datetime.timedelta(days=180) )
            end_date = datetime.date.today()
            stock_code = stock_name + '.BK'
            df_price = yf.download(stock_code,
                                  start=start_date,
                                  end=end_date,
                                  progress=True)
            df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
            last_price = df_price['Close'][-1]
            #### ----- ####
            pos_list = []
            for i in range(len(df_price)):
              rand_num = np.random.randn()
              if rand_num >= 0:
                pos_list.append('Buy')
              else:
                pos_list.append('Sell')
                
            expos_list = []
            for i in range(len(df_price)):
              rand_num = np.random.randn()
              if rand_num >= 0:
                expos_list.append(True)
              else:
                expos_list.append(False)
            
            df_price['pos'] = pos_list
            df_price['expos'] = expos_list
            #### ----- ####
            base = alt.Chart(df_price.reset_index()).encode(
              x = alt.X('Date'),
              y = alt.Y('Close', title='Price  (THB)', 
                        scale=alt.Scale(domain=[df_price['Close'].min()-2, df_price['Close'].max()+2])),
              tooltip=[alt.Tooltip('Date', title='Date'),alt.Tooltip('Close', title='Price (THB)')] )
            
            base2 = alt.Chart(df_price.reset_index()).encode(
              x = alt.X('Date') ,
              y = alt.Y('Close', title='Price  (THB)', scale=alt.Scale(domain=[df_price['Close'].min()-2, df_price['Close'].max()+2])),
                              color = alt.Color('pos', 
                                                scale=alt.Scale(domain=['Buy','Sell'],range=['green','red']),
                                               legend=alt.Legend(title="Model Advice")),
              tooltip=[alt.Tooltip('Date', title='Date'),
                       alt.Tooltip('Close', title='Price (THB)'),
                       alt.Tooltip('pos', title='Action')] )
                  
            c_line = (alt.Chart(df_price.reset_index())
                      .mark_line()
                      .encode(x = alt.X('Date') ,
                              y = alt.Y('Close', title='Price  (THB)', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                              tooltip=[alt.Tooltip('Date', title='Date'),
                                       alt.Tooltip('Close', title='Price (THB)')])
                      .interactive() )
            c_point = (alt.Chart(df_price[df_price['expos'] == True].reset_index())
                      .mark_circle()
                      .encode(x = alt.X('Date') ,
                              y = alt.Y('Close', title='Price  (THB)', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                              color = 'pos',
                              tooltip=[alt.Tooltip('pos', title='Action')])
                      .interactive() )
            c_all = alt.layer(c_line, c_point)
            st.write('#### Model performance compared to actual trading data in the past year')
            #st.altair_chart(c_line, use_container_width=True)
            #st.altair_chart(c_point, use_container_width=True)
            #st.altair_chart((base.mark_line() + base.mark_point()).resolve_scale(y='independent'))
            st.altair_chart( (base.mark_line() + base.mark_circle()), use_container_width=True)
            #bundle = (base.mark_line() + base.mark_circle().transform_filter(alt.FieldEqualPredicate(field='expos', equal=True)))
            #st.altair_chart(bundle, use_container_width=True) .add_selection(alt.selection_interval(bind='scales'))
            bundle2 = (base.mark_line() + base2.mark_circle().transform_filter(alt.FieldEqualPredicate(field='expos', equal=True)))
            layer1 = base.mark_line()
            layer2 = base2.mark_circle(size=50).transform_filter(alt.FieldEqualPredicate(field='expos', equal=True))
            bundle3 = alt.layer(layer1,layer2).configure_axis(labelFontSize=16,titleFontSize=18)
            st.altair_chart(bundle3, use_container_width=True)
            #st.altair_chart(c_all, use_container_width=True)
            
            #rand_num = np.random.randn()
            st.write('Model advice: ')
            #if rand_num > 0:
            if pos_list[-1] == 'Buy':
              st.success('#### BUY {} at current price of {} THB per share'.format('BBL',last_price) )
            else:
              st.error('#### SELL {} at current price of {} THB per share'.format('BBL',last_price) )
    
    ### --- DEVELOP MODEL MENU --- ###
    if model_b or model_side_b or st.session_state['model_b_status']:
      st.session_state['model_manage_b_status'] = False
      st.session_state['user_manage_b_status'] = False
      st.session_state['model_b_status'] = True
      st.session_state['advice_b_status'] = False
      placeholder_2.empty()
      placeholder_3.empty()
      placeholder_4.empty()
      with placeholder_3.container():
        #tab_list = ["Select Data 📈", "Set Parameters 💡", "Train Model 🚀", "Test Model 🧪", "Save Model 💾","PENDING"]
        #select_data_tab, set_para_tab, train_tab, test_tab, save_tab, pending_tab = st.tabs(tab_list)
        tab_list = ["Select Data 📈", "Set Parameters 💡", "Train Model 🚀", "Test Model 🧪", "Save Model 💾"]
        select_data_tab, set_para_tab, train_tab, test_tab, save_tab = st.tabs(tab_list)
        with select_data_tab:
            st.header("Select stock and time period 📈")
            fetch_price_data()
            #observe_button = st.checkbox('View Dataset 🔍')
            observe_button = st.button('View Dataset 🔍')
            if observe_button or st.session_state['observe_button_status']:
              st.session_state['observe_button_status'] = True
              observe_price()
              #split_button = st.checkbox("Split dataset ✂️")
              split_button = st.button("Split dataset ✂️")
              if split_button or st.session_state['split_button_status']:
                st.session_state['split_button_status'] = True
                split_dataset2()

        with set_para_tab:
            st.header("Set parameters for your trading model 💡")
            set_parameters()

        with train_tab:
            st.header("Train your model with train set 🚀")
            to_train_model = st.selectbox('Choose your model to train:', options=['BBL_01', 'BBL_02', 'PTT_07'])
            with st.expander('Model Information'):
              st.write("##### Model Parameters")
              st.write("Model name: {}".format(to_train_model) )
              st.write("Gamma: {}".format(0.99) )
              st.write("Starting epsilon: {:.2f}".format(1.00) )
              st.write("Epsilon decline rate: {:.4f}".format(0.005) )
              st.write("Minimum epsilon: {:.2f}".format(0.01) )
              st.write("Learning rate: {:.4f}".format(0.001) )
              st.write('  ')
              st.write("##### Trading Parameters")
              st.write("Initial account balance:  {:,} ฿".format(1500000) )
              st.write("Trading size (%):  {}%".format(10) )
              st.write("Trading size (THB):  {:,}".format(150000) )
              st.write("Commission fee:  {:.3f}%".format(0.157) )
            col1 , col2 = st.columns(2)
            with col1:
                set_train_episodes()
            with col2:
                st.write('  ')
                st.write('  ')
                train_button = st.button("Start Training 🏃")
            if train_button: #or st.session_state['train_button_status']:
              st.session_state['train_button_status'] = True
              train_model()
              if st.session_state['train_button_status']:
                train_result()

        with test_tab:
            st.header("Test your model on test set 🧪")
            test_button = st.button("Start Testing 🏹")
            if test_button:
                st.session_state['test_button_status'] = True
                st.write("Test Result")
                test_model()
                if st.session_state['test_button_status']:
                  test_result()
                  st.write('test report: ---')
                  st.write('parameters: ---')
                  st.write('train_episodes: ---')

        with save_tab:
            st.header("Save your model")
            #show_model_list_checkbox = st.checkbox('Show model list')
            #if show_model_list_checkbox:
              #st.write(model_df)
            with st.container():
              st.write('----- Model Information -----')
              st.write("##### Model Parameters")
              st.write("Model name: {}".format(to_train_model) )
              st.write("Gamma: {}".format(0.99) )
              st.write("Starting epsilon: {:.2f}".format(1.00) )
              st.write("Epsilon decline rate: {:.4f}".format(0.005) )
              st.write("Minimum epsilon: {:.2f}".format(0.01) )
              st.write("Learning rate: {:.4f}".format(0.001) )
              st.write('  ')
              st.write("##### Trading Parameters")
              st.write("Initial account balance:  {:,} ฿".format(1500000) )
              st.write("Trading size (%):  {}%".format(10) )
              st.write("Trading size (THB):  {:,}".format(150000) )
              st.write("Commission fee:  {:.3f}%".format(0.157) )
              st.write('  ')
              st.write("##### Train Result")
              st.write("Trained episodes:  {:,}".format(10) )
              st.write("Last session profit/loss:  {:+,.2f}".format(11576.23) )
              st.write('  ')
              st.write("##### Test Result")
              st.write("Profit/Loss on test set:  {:+,.2f}".format(1078.84) )
            save_button = st.button("Save 💾")
            if save_button:
                #save_model()
                time.sleep(2)
                st.success('Your model is saved successfully. Proceed to "Generate Advice" menu to use your model')

        #with pending_tab:
         #   st.header("PENDING adjustment...")
          #  st.success("select data ---- DONE")
           # st.warning("parameter -- adjust interface and input choice")
            #st.warning("parameter -- add info to each input")
            #st.success("train model -- add input field for n_episodes ---- DONE")
            #st.warning("train / test model -- better result visualization")
            #st.warning("save model -- integrate to cloud infrastructure")
            #st.warning("generate advice -- add load_model function")
            #st.warning("generate advice -- compulsory stock quote")
            #st.warning("generate advice -- formally written buy/sell advice")
            #st.error("overall -- user database and management system")
            #st.error("overall -- stock quote database")
            #st.error("overall -- set up cloud infrastructure")
