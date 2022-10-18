import streamlit as st
import altair as alt
import datetime
import numpy as np
import pandas as pd

st.title('Create DQN Trading Model 💡')
st.sidebar.markdown('## Create Model 💡')

st.write("#### Set these following parameters for your trading model")
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
trading_size_pct = st.slider("Trading size as a percentage of trading account (%):", 0, 100, 10)
commission_fee_pct = st.number_input("Commission fee as percent rate (%):", min_value=0.000, step=0.001, value=0.157, format='%1.3f')

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
