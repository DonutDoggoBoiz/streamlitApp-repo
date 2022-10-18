### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta

### --- DATABASE CONNECTION --- ###
deta = Deta(st.secrets["deta_key"])
user_db = deta.Base("user_db")

user_frame = pd.DataFrame(user_db.fetch().items)
user_list = user_frame['username'].values.tolist()

register_form = st.form('Register')
register_form.subheader('Registration Form 📝')
new_username = register_form.text_input('Username', placeholder='your username')
new_password = register_form.text_input('Password', type='password', placeholder='your password')

if register_form.form_submit_button('Register'):
  if len(new_username) <= 0:
    st.warning("Please enter a username")
  elif len(new_password) <= 0:
    st.warning("Please enter your password")
  elif len(new_username) > 0 and len(new_password) > 0:
    if new_username not in user_list:
      user_db.put({'username':new_username, 'password':new_password})
      st.success("Register Successful!")
    else: st.warning("Username already exists. Please enter a new username")
