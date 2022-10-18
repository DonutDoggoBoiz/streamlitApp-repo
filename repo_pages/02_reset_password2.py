### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta

### --- DATABASE CONNECTION --- ###
deta = Deta(st.secrets["deta_key"])
user_db = deta.Base("user_db")

user_frame = pd.DataFrame(user_db.fetch().items)
user_list = user_frame['username'].values.tolist()

reset_password_form = st.form('Reset Password')
reset_password_form.subheader('Reset Password Form 📝')
username = reset_password_form.text_input('Username', placeholder='your username')
password = reset_password_form.text_input('Password', type='password', placeholder='your password')
new_password = reset_password_form.text_input('Password', type='password', placeholder='your new password')

if reset_password_form.form_submit_button('Reset Password'):
  # if fields are empty
  if len(username) <= 0:
    st.warning("Please enter a username")
  elif len(password) <= 0:
    st.warning("Please enter your password")
  elif len(new_password) <= 0:
    st.warning("Please enter your new password")
  # fields are not empty
  elif len(username) > 0 and len(password) and len(new_password) > 0:
    if username not in user_list:
      st.warning('User not found. Please check your username')
    elif password == new_password:
      st.error('Your new password is similar to your current password. Please try a new one')
    elif user_frame.loc[user_frame['username'] == username,'password'].values != password:
      st.error("Current password incorrect. Please try again")
    else:
      user_key = user_frame.loc[user_frame['username'] == username, 'key'].to_string(index=False)
      user_db.update({'password':new_password}, user_key)
      st.success("Your password is successfully reset!")
