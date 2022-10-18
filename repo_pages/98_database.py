import streamlit as st
import pandas as pd
from deta import Deta

deta = Deta(st.secrets["deta_key"])
db = deta.Base("user_db")

register_form = st.form('Register')
register_form.subheader('Registration Form')
new_username = register_form.text_input('Username').lower()
new_password = register_form.text_input('Password', type='password')

if register_form.form_submit_button('Register'):
  if len(new_username) <= 0:
    st.warning("Please enter a username")
  elif len(new_password) <= 0:
    st.warning("Please enter your password")
  elif len(new_username) > 0 and len(new_password) > 0:
    db.put({'username':new_username, 'password':new_password})
    st.success("Register Successful!")

if st.button('show user database'):
  st.write('Here is the latest user database')
  aaa = db.fetch().items
  st.write(type(aaa))
  st.write(aaa)
  daframe = pd.DataFrame(aaa)
  st.write(daframe[1])
  #st.dataframe(db.fetch().items)

### --- MODEL DATABASE DEMO --- ###
n_episodes = 22
# agent parameters
agent_name = 'model_22'
agent_gamma = 0.99
agent_epsilon = 1.0
agent_epsilon_dec = 0
agent_epsilon_end = 0.01
agent_lr = 0.001
# trading parameters
initial_balance = 1000000
trading_size_pct = 10
commission_fee_pct = 0.157


db2 = deta.Base("model_db")
if st.button('add model to database2'):
  db2.put({'username':'admin99',
           'model_name':agent_name, 
            'gamma':agent_gamma,
            'start_epsilon':agent_epsilon ,
            'epsilon_decline':agent_epsilon_dec ,
            'epislon_min':agent_epsilon_end ,
            'learning_rate':agent_lr ,
            'initial_balance':initial_balance ,
            'trading_size_pct':trading_size_pct ,
            'commission_fee_pct':commission_fee_pct ,
            'episode_trained': n_episodes})
  db2_frame = pd.DataFrame(db2.fetch().items)
  st.write(db2_frame)
