import streamlit as st
import pandas as pd

st.markdown("## View, Edit, Delete Model 🛠️")
st.sidebar.markdown("### model management 🛠️")

view_model_button = st.checkbox('View Model')
if view_model_button:
    model_dict = {'model name': ['BBL_01', 'BBL_02', 'PTT_07'],
                 'account balance': [1000000, 1500000, 3344200],
                 'stock': ['BBL','BBL','PTT']
                 }
    model_df = pd.DataFrame(model_dict)
    st.dataframe(model_df)
