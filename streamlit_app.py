#########_IMPORT_LIBRARY_######################################################
import streamlit as st
import pandas as pd
from deta import Deta
import time
import datetime
import yfinance as yf
import altair as alt
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

from func.trainModel import train_model, test_model, deta_update_train, deta_update_test, save_model_local, generate_advice
from func.googleCloud import upload_model_gcs, download_model_gcs
#################################################################################

#########_DATABASE_CONNECTION_####################################
deta = Deta(st.secrets["deta_key"])
user_db = deta.Base("user_db")
model_db = deta.Base("model_db")
#model_frame = pd.DataFrame(model_db.fetch().items)

user_frame = pd.DataFrame(user_db.fetch().items)
user_list = user_frame['username'].values.tolist()
password_list = user_frame['password'].values.tolist()
name_list = user_frame['name'].values.tolist()

stock_db = deta.Base("stock_db")
stock_df = pd.DataFrame(stock_db.fetch().items)
stock_list = stock_df['symbol'].sort_values(ascending=True)
###############################################################

#########_SESSION_STATE_####################################
if 'login_status' not in st.session_state:
  st.session_state['login_status'] = False
if 'username' not in st.session_state:
  st.session_state['username'] = None
if 'name' not in st.session_state:
  st.session_state['name'] = None

if 'sess_model_name' not in st.session_state:
  st.session_state['sess_model_name'] = None
  
    #########_PRE_LOGIN_#########
if 'sign_in_b_disable' not in st.session_state:
  st.session_state['sign_in_b_disable'] = True
if 'sign_up_b_disable' not in st.session_state:
  st.session_state['sign_up_b_disable'] = False
if 'show_register_form' not in st.session_state:
  st.session_state['show_register_form'] = False
  
    #########_POST_LOGIN_#########
      ###_MANAGE_MODEL_###
if 'model_manage_b_status' not in st.session_state:
  st.session_state['model_manage_b_status'] = False  
if 'del_mod_button_status' not in st.session_state:
  st.session_state['del_mod_button_status'] = False
if 'edit_mod_button_status' not in st.session_state:
  st.session_state['edit_mod_button_status'] = False
      
      ###_DEVELOP_MODEL_###
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
if 'show_save_box' not in st.session_state:
  st.session_state['show_save_box'] = False
  
if 'xselect_exist_model_button_status' not in st.session_state:
  st.session_state['xselect_exist_model_button_status'] = False
  
      ###_GENERATE_ADVICE_##
if 'advice_b_status' not in st.session_state:
  st.session_state['advice_b_status'] = False
  
      ###_MANAGE_ACCOUNT_##
if 'user_manage_b_status' not in st.session_state:
  st.session_state['user_manage_b_status'] = False
###############################################################

#########_DEFINE_FUNCTION_####################################
      #########_PRE_LOGIN_#########
def login_func():
  st.session_state['login_status'] = True

def on_click_login_button():
  st.session_state['sign_in_b_disable'] = True
  st.session_state['sign_up_b_disable'] = False
  st.session_state['show_register_form'] = False

def on_click_regis_button():
  st.session_state['sign_in_b_disable'] = False
  st.session_state['sign_up_b_disable'] = True
  st.session_state['show_register_form'] = True
  

def on_click_logout():
  st.session_state['login_status'] = False
  st.session_state['username'] = None
  st.session_state['name'] = None
  st.session_state['sess_model_name'] = None
  st.session_state['sign_in_b_disable'] = True
  st.session_state['sign_up_b_disable'] = False
  st.session_state['show_register_form'] = False
  st.session_state['model_manage_b_status'] = False  
  st.session_state['del_mod_button_status'] = False
  st.session_state['edit_mod_button_status'] = False
  st.session_state['model_b_status'] = False
  st.session_state['observe_button_status'] = False
  st.session_state['split_button_status'] = False
  st.session_state['train_button_status'] = False
  st.session_state['test_button_status'] = False
  st.session_state['show_save_box'] = False
  st.session_state['xselect_exist_model_button_status'] = False
  st.session_state['advice_b_status'] = False
  st.session_state['user_manage_b_status'] = False
  

      #########_POST_LOGIN_#########
def on_click_home():
  st.session_state['model_manage_b_status'] = False  
  st.session_state['del_mod_button_status'] = False
  st.session_state['edit_mod_button_status'] = False
  st.session_state['model_b_status'] = False
  st.session_state['observe_button_status'] = False
  st.session_state['split_button_status'] = False
  st.session_state['train_button_status'] = False
  st.session_state['test_button_status'] = False

      #########_CLICK_TOP_AND_SIDEBAR_BUTTON_#########
def on_click_user_manage_b():
  st.session_state['user_manage_b_status'] = True
  st.session_state['model_manage_b_status'] = False
  st.session_state['model_b_status'] = False
  st.session_state['advice_b_status'] = False
  on_click_empty_ph_123()

def on_click_model_manage_b():
  st.session_state['user_manage_b_status'] = False
  st.session_state['model_manage_b_status'] = True
  st.session_state['model_b_status'] = False
  st.session_state['advice_b_status'] = False
  on_click_empty_ph_123()
  
def on_click_model_b():
  st.session_state['user_manage_b_status'] = False
  st.session_state['model_manage_b_status'] = False
  st.session_state['model_b_status'] = True
  st.session_state['advice_b_status'] = False
  on_click_empty_ph_123()

      #########_DEVELOP_MODEL_TAB_#########
def on_click_observe_b():
  st.session_state['observe_button_status'] = True
  
def on_click_split_b():
    st.session_state['split_button_status'] = True
    
def on_click_select_exist_model_b():
  st.session_state['xselect_exist_model_button_status'] = True
  
def on_click_show_save_box():
  st.session_state['show_save_box'] = True
    
      #########_GENERATE_ADVICE_TAB_#########
def on_click_advice_b():
  st.session_state['user_manage_b_status'] = False
  st.session_state['model_manage_b_status'] = False
  st.session_state['model_b_status'] = False
  st.session_state['advice_b_status'] = True
  on_click_empty_ph_123()
  
def on_change_date_select():
  st.session_state['observe_button_status'] = False
  st.session_state['split_button_status'] = False
  

      #########_UTILITY_FUNCTIONS_#########
def update_user_frame():
  global user_frame
  user_frame = pd.DataFrame(user_db.fetch().items)
  
def update_model_frame_u():
  global model_frame_u
  model_frame_u = pd.DataFrame(model_db.fetch({'username':st.session_state['username']}).items)
  
def rerun():
  st.experimental_rerun()

def on_click_empty_ph_123():
    placeholder_2.empty()
    placeholder_3.empty()
    placeholder_4.empty()

#########<END>_DEFINE_FUNCTION_####################################
  
#########_PRE_LOGIN_#############################################
if st.session_state['login_status'] == False:
  pre_login_top_holder = st.empty()
  pre_login_form_holder = st.empty()
  with pre_login_top_holder.container():
    top_holder_col1, _, top_holder_col3 = st.columns([1,5,1])
    with top_holder_col1:
      login_button_top = st.button('Sign in',
                                   disabled=st.session_state['sign_in_b_disable'],
                                   on_click=on_click_login_button)
    with top_holder_col3:
      register_button_top = st.button('Sign up',
                                      disabled=st.session_state['sign_up_b_disable'],
                                      on_click=on_click_regis_button)
    with pre_login_form_holder.container():
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
            st.error("Username or Password is incorrect. Please try again")
          else: 
            if user_frame.loc[user_frame['username'] == username,'password'].values != password:
              st.error("Username or Password is incorrect. Please try again")
            else:
              st.success("Login Successful! ✔️")
              st.session_state['username'] = username
              st.session_state['name'] = user_frame.loc[user_frame['username'] == username,'name'].to_string(index=False)
              login_func()
              time.sleep(3)
              st.experimental_rerun()
              
    ######_SIGN_UP_BUTTON_######
    if register_button_top or st.session_state['show_register_form']:
      #st.session_state['show_register_form'] = True
      with pre_login_form_holder.container():
        register_form = st.form('Register')
        register_form.subheader('Register 📝')
        new_name = register_form.text_input('Name', placeholder='eg. Michael Burry', max_chars=32)
        new_username = register_form.text_input('Username', placeholder='eg. mikeburr1971', max_chars=32)
        new_password = register_form.text_input('Password', type='password', placeholder='your password', max_chars=32)
        register_button = register_form.form_submit_button('Register')
        if register_button:
          if len(new_name) <= 0:
            st.warning('Please enter your name')
          if len(new_username) <= 0:
            st.warning("Please enter a username")
          elif len(new_password) <= 0:
            st.warning("Please enter your password")
          elif len(new_name) > 0 and len(new_username) > 0 and len(new_password) > 0:
            if new_username not in user_list:
              user_db.put({'username':new_username, 'password':new_password, 'name':new_name})
              st.success("Register Successful! ✔️")
              st.session_state['show_register_form'] = False
              time.sleep(3)
              st.experimental_rerun()
            else: st.warning("Username already exists. Please enter a new username")
              
    #####_SIGN_IN_BUTTON_######
    if login_button_top:
      st.session_state['show_register_form'] = False
      st.experimental_rerun()
      
#########_POST_LOGIN_#############################################
else:
  ###########################
  model_frame_u = pd.DataFrame(model_db.fetch({'username':st.session_state['username']}).items)
  ###########################
  
  ######_SIDEBAR_######
  st.sidebar.write('Welcome, {}'.format(st.session_state['name']) )
  with st.sidebar.container():
    sb_button_1, sb_button_2, _ = st.columns([3,3,1])
    with sb_button_1:
      logout_button_side = st.button('Logout', on_click=on_click_logout)
    with sb_button_2:
      home_button_side = st.button('Home', on_click=on_click_home) 
  st.sidebar.write('Menu:')
  user_manage_side_b = st.sidebar.button('Manage Account', key='user_manage_side', on_click=on_click_user_manage_b)
  manage_model_side_b = st.sidebar.button('Manage Model', key='model_manage_side', on_click=on_click_model_manage_b)
  model_side_b = st.sidebar.button('Develop Model', key='model_side_b', on_click=on_click_model_b)
  advice_side_b = st.sidebar.button('Generate Advice', key='advice_side_b', on_click=on_click_advice_b)
  
  ######_WELCOME_NOTE_######
  welcome_note_holder = st.empty()
  with welcome_note_holder.container():
    st.write('### Welcome, {}'.format(st.session_state['name']) )

  ######_MAIN_MENU_######
  menuholder = st.empty()
  with menuholder.container():
    menu_1, _, menu_2, menu_3, menu_4, menu_5 = st.columns([1,1,4,4,4,4])
    with menu_1:
      st.write('##### MENU:')                                          
    with menu_2:
      user_manage_b = st.button('Manage Account', on_click=on_click_user_manage_b)
    with menu_3:
      model_manage_b = st.button('Manage Model',on_click=on_click_model_manage_b)
    with menu_4:
      model_b = st.button('Develop Model',on_click=on_click_model_b)
    with menu_5:
      advice_b = st.button('Generate Advice', key='gen_advice_tab',on_click=on_click_advice_b)

  ######_MAIN_PLACEHOLDER_######
  placeholder_2 = st.empty()
  placeholder_3 = st.empty()
  placeholder_4 = st.empty()
  
  ######_MANAGE_ACCOUNT_MENU_######
  if user_manage_b or user_manage_side_b or st.session_state['user_manage_b_status']:
    #####_CHANGE_NAME_FORM_#####
    with placeholder_2.container():
      with st.form('change_name'):
        st.write('##### Change Name')
        new_name = st.text_input('Your new name', placeholder=str(st.session_state['name']), max_chars=32)
        change_name_button = st.form_submit_button('Change Name')
        if change_name_button:
          if len(new_name) <= 0:
            st.warning('Please enter your name')
          else:
            st.session_state['name'] = new_name
            user_key_to_update = user_frame.loc[user_frame['username']==st.session_state['username'], 'key'].to_list()[0]
            user_db.update(updates={'name':new_name}, key=user_key_to_update)
            st.success('Change name successful! ✔️')
            update_user_frame()
            with st.spinner('Processing...'):
              time.sleep(2)
              st.experimental_rerun()
          
      #####_CHANGE_PASSWORD_FORM_#####
      with st.form('change_password'):
        st.write('##### Change Password')
        old_password = st.text_input('Old Password', type='password', placeholder='your old password')
        new_password = st.text_input('New Password', type='password', placeholder='your new password')
        change_pass_button = st.form_submit_button('Change Password')
        if change_pass_button:
          if len(old_password) <= 0 or len(new_password) <= 0:
            st.warning('Please type your password')
          else: #both passwords length > 0
            if old_password != user_frame.loc[user_frame['username']==st.session_state['username'],'password'].to_list()[0]:
              st.warning('Old password is incorrect.  Please try again')
            else: #correct old password
              user_key_to_update = user_frame.loc[user_frame['username']==st.session_state['username'],'key'].to_list()[0]
              user_db.update(updates={'password':new_password}, key=user_key_to_update)
              st.success('Change password successful! ✔️')
              update_user_frame()
              with st.spinner('Processing...'):
                time.sleep(2)
              st.experimental_rerun()
              
  ######_MANAGE_MODEL_MENU_######
  if model_manage_b or manage_model_side_b or st.session_state['model_manage_b_status']:
    st.write('#### Model Management')
    if len(model_frame_u) <= 0:
      _warning = "You don't have any created model."
      _info = 'You can create a new model in '+'"Develop Model "'+'menu.'
      st.warning(_warning)
      st.info(_info, icon="ℹ️")
      
    else: #len(model_frame_u) > 0
    ######_GRID_VIEWER_######
      with placeholder_2.container():
        #st.write('#### Model Management')
        shuffle_col = ['model_name','stock_quote','start_date','end_date','episode_trained','initial_balance','trading_size_pct','commission_fee_pct','gamma',]
        model_grid = model_frame_u.loc[:,shuffle_col]
        gb = GridOptionsBuilder.from_dataframe(model_grid)
        gb.configure_selection('single', use_checkbox=True, pre_selected_rows=[0])
        gridoptions = gb.build()
        try:
          grid_response = AgGrid(model_grid,
                                 fit_columns_on_grid_load=False,
                                 gridOptions=gridoptions)
          selected_row = grid_response['selected_rows']
        except:
          st.warning('Loading model database...')

      ######_SEE_MODEL_DETAIL_BUTTON_######
      with placeholder_3.container():
        view_model_detail = st.button('See Model Detail')
        if view_model_detail:
          selected_row_model_name = selected_row[0]['model_name']
          with st.expander('Model information:',expanded=True):
              st.write('Name : {}'.format(selected_row_model_name))
              st.write('Gamma : {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'gamma'].to_list()[0]))
              st.write('Epsilon : {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'epsilon_start'].to_list()[0]))
              st.write('Epsilon decline : {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'epsilon_decline'].to_list()[0]))
              st.write('Epsilon minimum : {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'epsilon_min'].to_list()[0]))
              st.write('Learning Rate : {:.3f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'learning_rate'].to_list()[0]))
              st.write('Initial Balance : {:,} THB'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'initial_balance'].to_list()[0]))
              st.write('Trading Size : {:.2f}%'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'trading_size_pct'].to_list()[0]))
              st.write('Commission Fee : {:.2f}%'.format(model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'commission_fee_pct'].to_list()[0]))

      ######_EDIT_AND_DELETE_BUTTON_######
      with placeholder_4.container():
        ph2col1, ph2col2, _ = st.columns([1,1,6])
        with ph2col1:
          edit_mod_button = st.button('Edit')
        with ph2col2:
          del_mod_button = st.button('Delete')

        ######_EDIT_BUTTON_######################################################
        if edit_mod_button or st.session_state['edit_mod_button_status']:
          st.session_state['edit_mod_button_status'] = True
          selected_row_model_name = selected_row[0]['model_name']
          with placeholder_4.container():
              edit_form_col1, _ = st.columns([2,1])
              with edit_form_col1:
                with st.form('edit parameter form'):
                  st.write("##### Model parameters")
                  edt_agent_name = st.text_input("Model name: ", placeholder=str(selected_row[0]['model_name']),
                                                value=selected_row_model_name)
                  edt_agent_gamma = st.slider("Gamma: ", min_value=0.00, max_value=1.00, 
                                              value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'gamma'].to_list()[0] )
                  edt_agent_epsilon = st.slider("Starting epsilon (random walk probability): ", min_value=0.00, max_value=1.00, 
                                                value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'epsilon_start'].to_list()[0] )
                  edt_agent_epsilon_dec = st.select_slider("Epsilon decline rate (random walk probability decline): ",
                                                       options=[0.001,0.002,0.005,0.010], 
                                                           value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'epsilon_decline'].to_list()[0] )
                  edt_agent_epsilon_end = st.slider("Minimum epsilon: ", min_value=0.01, max_value=0.10, 
                                                    value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'epsilon_min'].to_list()[0] )
                  edt_agent_lr = st.select_slider("Learning rate: ", options=[0.001, 0.002, 0.005, 0.010], 
                                                  value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'learning_rate'].to_list()[0] )
                  st.write("##### Trading parameters")
                  edt_initial_balance = st.number_input("Initial account balance (THB):", min_value=0, step=1000, 
                                                        value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'initial_balance'].to_list()[0] )
                  edt_trading_size_pct = st.slider("Trading size as a percentage of initial account balance (%):", min_value=0, max_value=100, 
                                                   value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'trading_size_pct'].to_list()[0] )
                  edt_commission_fee_pct = st.number_input("Commission fee (%):", min_value=0.000, step=0.001, 
                                                           value=model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'commission_fee_pct'].to_list()[0], 
                                                           format='%1.3f')
                  edit_param_button = st.form_submit_button('Edit')

              ######_<FORM>_EDIT_BUTTON_######################################################
              if edit_param_button:
                key_to_update = model_frame_u.loc[model_frame_u['model_name']==selected_row_model_name,'key'].to_list()[0]
                update_dict = {'model_name':edt_agent_name,
                              'gamma':edt_agent_gamma,
                              'epsilon_start':edt_agent_epsilon,
                              'epsilon_decline':edt_agent_epsilon_dec,
                              'epsilon_min':edt_agent_epsilon_end,
                              'learning_rate':edt_agent_lr,
                              'initial_balance':edt_initial_balance,
                              'trading_size_pct':edt_trading_size_pct,
                              'commission_fee_pct':edt_commission_fee_pct,
                              'episode_trained':0}
                model_db.update(updates=update_dict, key=key_to_update)
                st.session_state['edit_mod_button_status'] = False
                st.success('Edit parameters successful! ✔️')
                with st.form('after edit ok'):
                  st.info('You can re-trained this model in "Develop Model" menu', icon="ℹ️")
                  edit_ok_button = st.form_submit_button('OK')
                if edit_ok_button:
                  time.sleep(2)
                  st.experimental_rerun()
        #####################################################################
  ######
        ######_DELETE_BUTTON_################################################
        if del_mod_button or st.session_state['del_mod_button_status']:
          st.session_state['del_mod_button_status'] = True
          with placeholder_4.container():
            with st.form('del_make_sure'):
              st.write('Are you sure?')
              make_sure_radio = st.radio('Please confirm your choice:', options=('No', 'Yes') )
              confirm_button = st.form_submit_button('Confirm')
              if confirm_button:
                if make_sure_radio == 'Yes':
                  st.session_state['del_mod_button_status'] = False
                  selected_model_name = selected_row[0]['model_name']
                  key_to_del = model_frame_u.loc[model_frame_u['model_name']==selected_model_name,'key'].to_list()[0]
                  model_db.delete(key_to_del)
                  st.error('Model {} has been successfully deleted'.format(selected_model_name))
                  time.sleep(3)
                  st.experimental_rerun()
                elif make_sure_radio == 'No':
                  st.session_state['del_mod_button_status'] = False
                  st.experimental_rerun()
########################################################################
  
  ######_GENERATE_ADVICE_MENU_################################################
  if advice_b or advice_side_b or st.session_state['advice_b_status']:
    st.markdown("#### Generate Investment Advice 📈")
    if len(model_frame_u) <= 0:
      _warning = "You don't have any created model."
      _info = 'You can create a new model in '+'"Develop Model "'+'menu.'
      st.warning(_warning)
      st.info(_info, icon="ℹ️")
    else: #len(model_frame_u) > 0
      with placeholder_2.container():
        #st.markdown("#### Generate Investment Advice 📈")
        model_options = model_frame_u.loc[:,'model_name']
        selected_advice_model = st.selectbox('Choose your model',options=model_options)
        with st.expander('Model Information:'):
          st.write(' Model Name : {}'.format(selected_advice_model))
          st.write(' Stock Quote : {}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'stock_quote'].to_list()[0]))
          st.write(' Start Date : {}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'start_date'].to_list()[0]))
          st.write(' End Date : {}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'end_date'].to_list()[0]))
          st.write(' Episode Trained : {}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'episode_trained'].to_list()[0]))
          st.write(' Train Profit/Loss : {:+,.2f} THB'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'trained_result'].to_list()[0]))
          st.write(' Test Profit/Loss : {:+,.2f} THB'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'test_result'].to_list()[0]))
          st.write(' Initial Balance : {:,} THB'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'initial_balance'].to_list()[0]))
          st.write(' Trading Size (%): {:.2f}%'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'trading_size_pct'].to_list()[0]))
          st.write(' Trading Size (THB): {:.2f} THB'.format( (model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'initial_balance'].to_list()[0])*
                                                           ((model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'trading_size_pct'].to_list()[0])/100)))
          st.write(' Commission Fee : {:.2f}%'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'commission_fee_pct'].to_list()[0]))
          st.write(' Gamma : {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'gamma'].to_list()[0]))
          st.write(' Epsilon Start: {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'epsilon_start'].to_list()[0]))
          st.write(' Epsilon Decline rate : {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'epsilon_decline'].to_list()[0]))
          st.write(' Epsilon Minimum : {:.2f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'epsilon_min'].to_list()[0]))
          st.write(' Learning Rate : {:.3f}'.format(model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'learning_rate'].to_list()[0]))

          generate_advice_button = st.button('Generate Advice')

        #####_GENERATE_ADVICE_BUTTON_#####
        if generate_advice_button:
          stock_name = model_frame_u.loc[model_frame_u['model_name']==selected_advice_model,'stock_quote'].to_list()[0]
          start_date = ( datetime.date.today() - datetime.timedelta(days=180) )
          end_date = datetime.date.today()
          stock_code = stock_name + '.BK'
          df_price = yf.download(stock_code,
                                start=start_date,
                                end=end_date,
                                progress=True)
          with st.spinner('Generating advice...'):
            download_model_gcs(save_username=st.session_state['username'],
                               ag_name=selected_advice_model)
            generate_advice(ag_df_price_advice=df_price,
                            save_username=st.session_state['username'],
                            ag_name=selected_advice_model,
                            ag_quote=stock_name)
######################################################################################################

  ######_DEVELOP_MODEL_MENU_##########################################
  if model_b or model_side_b or st.session_state['model_b_status']:
    with placeholder_3.container():
      tab_list = ["Select Dataset 📈", "Train Model 🚀", "Test Model 🧪", "Save Model 💾"]
      select_data_tab, train_tab, test_tab, save_tab = st.tabs(tab_list)
      
      ######_SELECT_DATA_TAB_######
      with select_data_tab:
        st.write("#### Select stock and time period 📈")
        stock_name = st.selectbox('Select your Stock', options=stock_list, index=86)
        company_name = stock_df[stock_df['symbol']==stock_name]['company_name'].to_string(index=False)
        market_name = stock_df[stock_df['symbol']==stock_name]['market'].to_string(index=False)
        industry_name = stock_df[stock_df['symbol']==stock_name]['industry'].to_string(index=False)
        sector_name = stock_df[stock_df['symbol']==stock_name]['sector'].to_string(index=False)
        with st.expander('Company Information', expanded=True):
          st.write('{}'.format(company_name))
          st.write('Market: {}'.format(market_name))
          st.write('Industry: {}'.format(industry_name))
          st.write('Sector: {}'.format(sector_name))
        start_date = st.date_input("Select start date: ",
                                   (datetime.date.today()-datetime.timedelta(days=365)),
                                  on_change=on_change_date_select)
        end_date = st.date_input("Select end date: ",
                                 datetime.date.today(),
                                on_change=on_change_date_select)
        select_data_menu_holder = st.empty()
        select_data_chart_holder = st.empty()
        split_data_chart_holder = st.empty()
        success_box_holder = st.empty()
        select_data_slider_holder = st.empty()
        with select_data_menu_holder.container():
          col_observe_b, col_describe = st.columns([1,3])
          with col_observe_b:
            observe_button = st.button('View Dataset 🔍', on_click=on_click_observe_b)
########
        if observe_button or st.session_state['observe_button_status']:
          stock_code = stock_name + '.BK'
          df_price = yf.download(stock_code, start=start_date, end=end_date, progress=True)
          df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
          df_length = df_price['Close'].count()
          
          with col_describe:
            st.write('This dataset contains {} days of historical prices'.format(df_length))
          alt_price_range = (alt.Chart(df_price['Close'].reset_index()).mark_line().encode(
            x = alt.X('Date'),
            y = alt.Y('Close',
                      title='Price  (THB)',
                      scale=alt.Scale(domain=[df_price['Close'].min()-2, df_price['Close'].max()+2])),
            tooltip=[alt.Tooltip('Date', title='Date'),
                     alt.Tooltip('Close', title='Price (THB)')]).configure_axis(labelFontSize=14,titleFontSize=16))
          
          with select_data_chart_holder.container():
            st.altair_chart(alt_price_range.interactive(), use_container_width=True)
            with st.form('split_slider'):
              st.write('This dataset contains {} days of historical prices'.format(df_length))
              split_point = st.slider('Select the split point between Train set and Test set:', 0, int(df_length), int(df_length/2))
              ##############################
              train_size_pct = (split_point/df_length)*100
              test_size_pct = 100-train_size_pct
              df_price['split'] = 'split'
              df_price.loc[:split_point, 'split'] = 'Train set'
              df_price.loc[split_point:, 'split'] = 'Test set'
              df_price_train = df_price[:split_point]
              df_price_test = df_price[split_point:]
              train_prices = df_price_train['Close'].to_numpy()
              test_prices = df_price_test['Close'].to_numpy()
              ##############################
              alt_split = (alt.Chart(df_price.reset_index()).mark_line().encode(
                x = alt.X('Date'),
                y = alt.Y('Close',
                          title='Price  (THB)',
                          scale=alt.Scale(domain=[df_price['Close'].min()-2,
                                                  df_price['Close'].max()+2])),
                color = alt.Color('split',
                                  scale=alt.Scale(domain=['Train set','Test set'],range=['#4682b4','orange']),
                                  legend=alt.Legend(title="Dataset")),
                tooltip=[alt.Tooltip('Date', title='Date'),
                         alt.Tooltip('Close', title='Price (THB)'),
                         alt.Tooltip('split', title='Dataset')]).configure_axis(labelFontSize=14,titleFontSize=16))
              ##############################
              split_button = st.form_submit_button("Split dataset ✂️", on_click=on_click_split_b)

              if split_button or st.session_state['split_button_status']:
                with split_data_chart_holder.container():
                  st.altair_chart(alt_split.interactive(), use_container_width=True)
                  st.write('Dataset will be split into {} records ({:.2f}%) as training set and {} records ({:.2f}%) as test set'.format(
                    split_point,train_size_pct,df_length-split_point,test_size_pct) )
                  st.success('Your Datasets are ready!')
                  _info = 'Please proceed to "Train Model 🚀" tab to create/train a model'
                  st.info(_info, icon="ℹ️")
                  
      ######_TRAIN_TAB_######
      with train_tab:
        st.write('#### Train model with train set 🚀')
        if st.session_state['observe_button_status'] == False:
          st.warning('No dataset detected.')
          st.info('Please select dataset in "Select Dataset 📈" tab', icon="ℹ️")
        elif st.session_state['split_button_status'] == False:
          st.warning('No splited dataset detected.')
          st.info('Please split your dataset in "Select Dataset 📈" tab', icon="ℹ️")
        else:
          select_model_radio = st.radio('Which model do you want to train?',
                                        options=['New Model', 'Existing Model'],
                                        horizontal=True)
          train_allowed = False
          ######_RADIO_NEW_MODEL_######
          if select_model_radio == 'New Model':
            with st.form('set_param_new_model'):
              _l, col1_set_para, _r = st.columns([1,7,1])
              with col1_set_para:
                st.write("##### Model parameters")
                nm_agent_name = st.text_input("Model name: ", max_chars=32, placeholder="eg. model_01")
                nm_agent_gamma = st.slider("Gamma: ", 0.00, 1.00, 0.90)
                nm_agent_epsilon = st.slider("Starting epsilon (random walk probability): ", 0.00, 1.00, 1.00)
                nm_agent_epsilon_dec = st.select_slider("Epsilon decline rate (random walk probability decline):",
                                                     options=[0.001,0.002,0.005,0.010], value=0.001)
                nm_agent_epsilon_end = st.slider("Minimum epsilon: ", 0.01, 0.10, 0.01)
                nm_agent_lr = st.select_slider("Learning rate: ", options=[0.001, 0.002, 0.005, 0.010], value=0.001)

                st.write("##### Trading parameters")
                nm_initial_balance = st.number_input("Initial account balance (THB):", min_value=100000, step=10000, value=1000000)
                nm_trading_size_pct = st.slider("Trading size as a percentage of initial account balance (%):", 0, 100, 10)
                nm_trade_size = nm_initial_balance * nm_trading_size_pct / 100
                nm_commission_fee_pct = st.number_input("Commission fee (%):", min_value=0.000, step=0.001, value=0.157, format='%1.3f')
                nm_create_model = st.form_submit_button("Create Model")
                if nm_create_model:
                  if len(nm_agent_name) <= 0:
                    st.warning('Please name your model')
                  elif len(model_frame_u) > 0:
                    if (nm_agent_name in model_frame_u['model_name'].to_list()) == True:
                      st.warning('Model name is already exist. Please type new model name')
                  else:
                    st.session_state['sess_model_name'] = nm_agent_name
                    model_param_dict = {'username': st.session_state['username'],
                                        'model_name': nm_agent_name,
                                        'stock_quote': stock_name,
                                        'start_date': str(start_date),
                                        'end_date': str(end_date),
                                        'split_point': split_point,
                                        'episode_trained': 0,
                                        'trained_result': 0,
                                        'test_result': 0,
                                        'gamma': nm_agent_gamma,
                                        'epsilon_start': nm_agent_epsilon,
                                        'epsilon_decline': nm_agent_epsilon_dec,
                                        'epsilon_min': nm_agent_epsilon_end,
                                        'learning_rate': nm_agent_lr,
                                        'initial_balance': nm_initial_balance,
                                        'trading_size_pct': nm_trading_size_pct,
                                        'commission_fee_pct': nm_commission_fee_pct}
                    model_db.put(model_param_dict)
                    update_model_frame_u()
                    st.success('Create Model Successful!')
                    _info = 'You can set episodes and start training in a box below'
                    st.info(_info, icon="ℹ️")
                    train_allowed = True

          ######_RADIO_EXISTING_MODEL_######
          if select_model_radio == 'Existing Model':
            if len(model_frame_u) <= 0:
              _warning = "You don't have any created model."
              _info = 'select '+'"New Model "'+'option to create a new model.'
              st.warning(_warning)
              st.info(_info, icon="ℹ️")
            else: #len(model_frame_u) > 0
              with st.form('select_existing_model'):
                ex_model_list = model_frame_u['model_name'].sort_values(ascending=True)
                ex_to_train_model = st.selectbox('Select your existing model',
                                                 options=ex_model_list)
                ex_agent_name = ex_to_train_model
                ######### TEST REUSE VARIABLE ######
                nm_agent_name = ex_to_train_model
                nm_agent_gamma = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'gamma'])
                nm_agent_epsilon = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'epsilon_start'])
                nm_agent_epsilon_dec = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'epsilon_decline'])
                nm_agent_epsilon_end = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'epsilon_min'])
                nm_agent_lr = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'learning_rate'])
                nm_initial_balance = int(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'initial_balance'])
                nm_trading_size_pct = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'trading_size_pct'])
                nm_commission_fee_pct = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'commission_fee_pct'])
                ####################################
                info_trade_size_nom = nm_initial_balance * (nm_trading_size_pct/100)
                ex_select_exist_model = st.form_submit_button('Select Model')

                if ex_select_exist_model:
                  train_allowed = True
                  with st.expander('Model Information', expanded=True):
                    st.write("##### Model Parameters")
                    st.write("Model name: {}".format(nm_agent_name))
                    st.write("Gamma: {:.2f}".format(nm_agent_gamma))
                    st.write("Starting epsilon: {:.2f}".format(nm_agent_epsilon))
                    st.write("Epsilon decline rate: {:.4f}".format(nm_agent_epsilon_dec))
                    st.write("Minimum epsilon: {:.2f}".format(nm_agent_epsilon_end))
                    st.write("Learning rate: {:.4f}".format(nm_agent_lr))
                    st.write('  ')
                    st.write("##### Trading Parameters")
                    st.write("Initial account balance:  {:,} ฿".format(nm_initial_balance))
                    st.write("Trading size (%):  {}%".format(nm_trading_size_pct))
                    st.write("Trading size (THB):  {:,}".format(info_trade_size_nom))
                    st.write("Commission fee:  {:.3f}%".format(nm_commission_fee_pct))

          with st.form('train_form'):
            st.write('How many episodes to train?')
            t_form_col1 , t_form_col2 = st.columns(2)
            with t_form_col1:
              xtrain_episodes = st.number_input('Number of training episodes:', value=2, step=1, min_value=0)
            with t_form_col2:
              st.write('  ')
              st.write('  ')
              xtrain_button = st.form_submit_button("Start Training 🏃", disabled=not(train_allowed))
            if xtrain_button:
              if select_model_radio == 'New Model' or select_model_radio == 'Existing Model':
                train_model(ag_df_price_train=df_price_train,
                            ag_name=nm_agent_name,
                            ag_gamma=nm_agent_gamma,
                            ag_eps=nm_agent_epsilon,
                            ag_eps_dec=nm_agent_epsilon_dec,
                            ag_eps_min=nm_agent_epsilon_end,
                            ag_lr=nm_agent_lr,
                            ag_ini_bal=nm_initial_balance,
                            ag_trade_size_pct=nm_trading_size_pct,
                            ag_com_fee_pct=nm_commission_fee_pct,
                            ag_train_episode=xtrain_episodes)
                deta_update_train(username=st.session_state['username'],
                                  deta_key=st.session_state['deta_key'])
                update_model_frame_u()
                _info = 'Please proceed to "Test Model 🧪" tab to test your model'
                st.info(_info, icon="ℹ️")
                  #st.warning('Please create or select existing model to train.')
              
################################################################################################################
      ######_TEST_TAB_######
      with test_tab:
        st.write("#### Test your model on test set 🧪")
        if st.session_state['observe_button_status'] == False:
          st.warning('No dataset detected.')
          st.info('Please select dataset in "Select Dataset 📈" tab', icon="ℹ️")
        elif st.session_state['split_button_status'] == False:
          st.warning('No splited dataset detected.')
          st.info('Please split your dataset in "Select Dataset 📈" tab', icon="ℹ️")
        else:
          test_button = st.button("Start Testing 🏹")
          if test_button:
            test_model(ag_df_price_test=df_price_test,
                       ag_name=nm_agent_name,
                       ag_gamma=nm_agent_gamma,
                       ag_eps=nm_agent_epsilon,
                       ag_eps_dec=nm_agent_epsilon_dec,
                       ag_eps_min=nm_agent_epsilon_end,
                       ag_lr=nm_agent_lr,
                       ag_ini_bal=nm_initial_balance,
                       ag_trade_size_pct=nm_trading_size_pct,
                       ag_com_fee_pct=nm_commission_fee_pct,
                       ag_train_episode=xtrain_episodes)
            deta_update_test(username=st.session_state['username'],
                             deta_key=st.session_state['deta_key'])
            update_model_frame_u()
            _info = 'Please proceed to "Save 💾" tab to save your model'
            st.info(_info, icon="ℹ️")
######################################################################################################
######
      ######_SAVE_TAB_######
      with save_tab:
        st.write("#### Save your model")
        save_model_button = st.button('Save 💾', on_click=on_click_show_save_box)
        if st.session_state['show_save_box'] == True:
          model_name_sv = st.session_state['sess_model_name']
          with st.form('save model'):
            with st.expander('----- Model Information -----'):
              st.write("##### Model Parameters")
              st.write("Model name: {}".format(model_name_sv) )
              st.write("Gamma: {}".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                  'gamma'].values)))
              st.write("Starting epsilon: {:.2f}".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                                 'epsilon_start'].values)))
              st.write("Epsilon decline rate: {:.4f}".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                                     'epsilon_decline'].values)))
              st.write("Minimum epsilon: {:.2f}".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                                'epsilon_min'].values)))
              st.write("Learning rate: {:.4f}".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                              'learning_rate'].values)))
              st.write('  ')
              st.write("##### Trading Parameters")
              st.write("Initial account balance:  {:,} ฿".format(int(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                                       'initial_balance'].values)))
              st.write("Trading size (%):  {}%".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                               'trading_size_pct'].values)))
              st.write("Trading size (THB):  {:,}".format(int(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                                'initial_balance'].values)*
                                                         (float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                                  'trading_size_pct'].values)/100)
                                                         )
                      )
              st.write("Commission fee:  {:.3f}%".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,
                                                                                 'commission_fee_pct'].values)))
              st.write('  ')
              st.write("##### Train Result")
              st.write("Profit/Loss:  {:.2f}THB".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,'trained_result'].values)))
              st.write("Profit/Loss (%):  {:+,.2f}%".format(100*float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,'trained_result'].values)/
                                                         int(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,'initial_balance'].values)))
              st.write('  ')
              st.write("##### Test Result")
              st.write("Profit/Loss:  {:.2f}THB".format(float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,'test_result'].values)))
              st.write("Profit/Loss (%):  {:+,.2f}%".format(100*float(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,'test_result'].values)/
                                                         int(model_frame_u.loc[model_frame_u['model_name']==model_name_sv,'initial_balance'].values)))
            save_submit = st.form_submit_button('Confirm')
            if save_submit:
              save_model_local(save_username=st.session_state['username'])
              upload_model_gcs(save_username=st.session_state['username'],
                               ag_name=model_name_sv)
              time.sleep(2)
              st.success('Save model successful')
              time.sleep(1)
              st.info('You can use your model at "Generate Advice" menu', icon="ℹ️")
              
######################################################################################################
