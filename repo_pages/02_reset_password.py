import streamlit as st

# --- reset password form
reset_password_form = st.form('Reset Password')
reset_password_form.subheader('Reset Password')
username = reset_password_form.text_input('Username').lower()
password = reset_password_form.text_input('Current password', type='password')
new_password = reset_password_form.text_input('New password', type='password')

if reset_password_form.form_submit_button('Reset Password'):
  if len(username) <= 0:
    st.warning("Please enter a username")
  elif len(password) <= 0:
    st.warning("Please enter current password")
  elif len(new_password) <= 0:
    st.warning("Please enter new password")
  elif len(username) > 0 and len(password) > 0 and len(new_password) > 0:
    st.success("Reset Password Successful!")
# --- --- ---
