import streamlit as st
import numpy as np
import pandas as pd

#st.markdown("### This is a Login Page 🔑")
#st.sidebar.markdown("### Login 🔑")
#st.write('--------')

# --- mocked up User Database ---
if 'users' not in st.session_state:
    st.session_state['users'] = ['admin']
if 'passwords' not in st.session_state:
    st.session_state['passwords'] = ['admin']
    
#if st.sidebar.checkbox('database'):
#  users = ['admin']
#  passwords = ['admin']

# st.title("Login Page 🔑")


st.markdown("## Login Page 🔑")
# --- login form
login_form = st.form('Login')
login_form.subheader('Login Form')
username = login_form.text_input('Username').lower()
password = login_form.text_input('Password', type='password')

if login_form.form_submit_button('Login'):
  if len(username) <= 0:
    st.warning("Please enter a username")
  elif len(password) <= 0:
    st.warning("Please enter your password")
  elif len(username) > 0 and len(password) > 0:
    st.success("Login Successful!")
# --- --- ---

menu = ["Home", "Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Home")
    users_df = pd.DataFrame({'users':st.session_state['users'],
                             'passwords':st.session_state['passwords'] }
                           )
    st.dataframe(users_df)
  
elif choice == "Login":
  st.subheader("Login Section")
  username = st.sidebar.text_input("Username")
  password = st.sidebar.text_input("Password", type='password')
  if st.sidebar.checkbox("Login"):
    if len(username) <= 0:
        st.sidebar.error("Please enter username")
    if len(password) <= 0:
        st.sidebar.error("Please enter password")
    #if password == '12345':
    #if passwords[users.index(username)] == password:
    if len(username) > 0 and len(password) > 0:
        if st.session_state['passwords'][st.session_state['users'].index(username)] == password:
          st.success("Logged In as {}".format(username))

          task = st.selectbox("Task", ["Add Post", "Analytics", "Profiles"])
          if task == "Add Post":
            st.subheader("Add Your Post")
          elif task == "Analytics":
            st.subheader("Analytics")
          elif task == "Profiles":
            st.subheader("Profiles")
            users_df = pd.DataFrame({'users':st.session_state['users'],
                                     'passwords':st.session_state['passwords']}
                                   )
            st.dataframe(users_df)
        else:
          st.warning("Incorrect Username/Password")
  
elif choice == "Sign Up":
  st.subheader("Create New Account")
  new_user = st.text_input("Username")
  new_password = st.text_input("Password", type='password')
  
  if st.button("Signup"):
    if len(new_user) <= 0:
        st.error("Please provide username")
    st.session_state['users'].append(new_user)
    st.session_state['passwords'].append(new_password)
    st.success("You have successfully created a valid account")
    st.info("Go to Login Menu to login")
