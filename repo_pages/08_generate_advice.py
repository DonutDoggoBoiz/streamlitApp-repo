import streamlit as st
import datetime
import yfinance as yf
import altair as alt
import numpy as np

st.markdown("# Generate Advice 📈")
st.sidebar.markdown("# Generate Advice 📈")

selected_model = st.selectbox('Choose your model',
                              options=['BBL_01', 'BBL_02', 'PTT_07']
                             )

generate_advice_button = st.button('Generate Advice')
if generate_advice_button:
  stock_name = 'BBL'
  start_date = datetime.date(datetime.date.today().year-1, datetime.date.today().month, datetime.date.today().day )
  end_date = datetime.date.today()
  stock_code = stock_name + '.BK'
  df_price = yf.download(stock_code,
                        start=start_date,
                        end=end_date,
                        progress=True)
  df_price.drop(columns=['Adj Close','Volume'] , inplace=True)
  last_price = df_price['Close'][-1]

  c = (alt.Chart(df_price['Close'].reset_index() 
                )
            .mark_line()
            .encode(x = alt.X('Date') ,
                    y = alt.Y('Close', scale=alt.Scale(domain=[df_price['Close'].min()-10, df_price['Close'].max()+10]) ) ,
                    tooltip=['Date','Close']
                   )
            .interactive()
        )
  st.altair_chart(c, use_container_width=True)
  
  rand_num = np.random.randn()
  st.write('Model recommend: ')
  if rand_num > 0:
    st.success('#### BUY at current price of {}'.format(last_price) )
  else:
    st.error('#### SELL at current price of {}'.format(last_price) )
