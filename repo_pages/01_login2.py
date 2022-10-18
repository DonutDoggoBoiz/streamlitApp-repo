### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta

### --- DATABASE CONNECTION --- ###
deta = Deta(st.secrets["deta_key"])
user_db = deta.Base("user_db")

user_frame = pd.DataFrame(user_db.fetch().items)
user_list = user_frame['username'].values.tolist()
password_list = user_frame['password'].values.tolist()

login_form = st.form('Login')
login_form.subheader('Login Form 📝')
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
        st.write('Welcome, {}'.format(username))
        st.sidebar.button('Logout')
        st.sidebar.button('Reset Password')
