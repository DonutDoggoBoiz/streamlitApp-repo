### --- IMPORT LIBRARY --- ###
import streamlit as st
import pandas as pd
from deta import Deta
import time
from functions import login_form

login_form()

if st.sidebar.button('check sess'):
  st.write('check session state, {}'.format(st.session_state['username']))
