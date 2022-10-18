# with placeholder = st.empty()

### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
import time

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
