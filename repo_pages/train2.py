######
      with train_tab2:
        select_model_radio = st.radio('Which model do you want to train?',
                                      options=['New Model', 'Existing Model'],
                                      horizontal=True)
########
        if select_model_radio == 'New Model':
          with st.form('set_param_new_model'):
            _l, col1_set_para, _r = st.columns([1,7,1])
            with col1_set_para:
              st.write("##### Model parameters")
              nm_agent_name = st.text_input("Model name: ", max_chars=32, placeholder="eg. model_01")
              nm_agent_gamma = st.slider("Gamma: ", 0.00, 1.00, 0.90)
              nm_agent_epsilon = st.slider("Starting epsilon (random walk probability): ", 0.00, 1.00, 1.00)
              nm_agent_epsilon_dec = st.select_slider("Epsilon decline rate (random walk probability decline):",
                                                   options=[0.001,0.002,0.005,0.010], value=0.001)
              nm_agent_epsilon_end = st.slider("Minimum epsilon: ", 0.01, 0.10, 0.01)
              nm_agent_lr = st.select_slider("Learning rate: ", options=[0.001, 0.002, 0.005, 0.010], value=0.001)

              st.write("##### Trading parameters")
              nm_initial_balance = st.number_input("Initial account balance (THB):", min_value=100000, step=10000, value=1000000)
              nm_trading_size_pct = st.slider("Trading size as a percentage of initial account balance (%):", 0, 100, 10)
              nm_trade_size = nm_initial_balance * nm_trading_size_pct / 100
              nm_commission_fee_pct = st.number_input("Commission fee (%):", min_value=0.000, step=0.001, value=0.157, format='%1.3f')
              nm_create_model = st.form_submit_button("Create Model")
              if nm_create_model:
                if len(nm_agent_name) <= 0:
                  st.warning('Please name your model')
                elif (nm_agent_name in model_frame_u['model_name'].to_list()) == True:
                  st.warning('Model name is already exist. Please type new model name')
                else:
####################
                  st.session_state['sess_model_name'] = nm_agent_name
                  model_param_dict = {'username': st.session_state['username'],
                                  'model_name': nm_agent_name,
                                  'stock_quote': stock_name,
                                  'start_date': str(start_date),
                                  'end_date': str(end_date),
                                  'split_point': split_point,
                                  'episode_trained': 0,
                                  'trained_result': 0,
                                  'test_result': 0,
                                  'gamma': nm_agent_gamma,
                                  'epsilon_start': nm_agent_epsilon,
                                  'epsilon_decline': nm_agent_epsilon_dec,
                                  'epsilon_min': nm_agent_epsilon_end,
                                  'learning_rate': nm_agent_lr,
                                  'initial_balance': nm_initial_balance,
                                  'trading_size_pct': nm_trading_size_pct,
                                  'commission_fee_pct': nm_commission_fee_pct}
####################
                  model_db.put(model_param_dict)
                  update_model_frame_u()
                  st.success('Create Model Successful!')
########
      if select_model_radio == 'Existing Model':
          with st.form('select_existing_model'):
            ex_model_list = model_frame_u['model_name'].sort_values(ascending=True)
            ex_to_train_model = st.selectbox('Select your existing model',
                                    options=ex_model_list)
            ex_agent_name = ex_to_train_model
            ######### TEST REUSE VARIABLE ######
            nm_agent_name = ex_to_train_model
            nm_agent_gamma = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'gamma'])
            nm_agent_epsilon = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'epsilon_start'])
            nm_agent_epsilon_dec = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'epsilon_decline'])
            nm_agent_epsilon_end = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'epsilon_min'])
            nm_agent_lr = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'learning_rate'])
            nm_initial_balance = int(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'initial_balance'])
            nm_trading_size_pct = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'trading_size_pct'])
            nm_commission_fee_pct = float(model_frame_u.loc[model_frame_u['model_name']==ex_to_train_model,'commission_fee_pct'])
            ####################################
            info_trade_size_nom = nm_initial_balance * nm_trading_size_pct
            ex_select_exist_model = st.form_submit_button('Select Model')

          if ex_select_exist_model:
            with st.expander('Model Information', expanded=True):
              st.write("##### Model Parameters")
              st.write("Model name: {}".format(nm_agent_name))
              st.write("Gamma: {:.2f}".format(nm_agent_gamma))
              st.write("Starting epsilon: {:.2f}".format(nm_agent_epsilon))
              st.write("Epsilon decline rate: {:.4f}".format(nm_agent_epsilon_dec))
              st.write("Minimum epsilon: {:.2f}".format(nm_agent_epsilon_end))
              st.write("Learning rate: {:.4f}".format(nm_agent_lr))
              st.write('  ')
              st.write("##### Trading Parameters")
              st.write("Initial account balance:  {:,} ฿".format(nm_initial_balance))
              st.write("Trading size (%):  {}%".format(nm_trading_size_pct))
              st.write("Trading size (THB):  {:,}".format(info_trade_size_nom))
              st.write("Commission fee:  {:.3f}%".format(nm_commission_fee_pct))
########
      with st.form('train_form'):
        st.write('How many episodes to train?')
        t_form_col1 , t_form_col2 = st.columns(2)
        with t_form_col1:
            xtrain_episodes = st.number_input('Number of training episodes:', value=2, step=1, min_value=0)
        with t_form_col2:
            st.write('  ')
            st.write('  ')
            xtrain_button = st.form_submit_button("Start Training 🏃")
        if xtrain_button:
############
          if select_model_radio == 'New Model' or select_model_radio == 'Existing Model':
            train_model(ag_df_price_train=df_price_train,
                        ag_name=nm_agent_name,
                        ag_gamma=nm_agent_gamma,
                        ag_eps=nm_agent_epsilon,
                        ag_eps_dec=nm_agent_epsilon_dec,
                        ag_eps_min=nm_agent_epsilon_end,
                        ag_lr=nm_agent_lr,
                        ag_ini_bal=nm_initial_balance,
                        ag_trade_size_pct=nm_trading_size_pct,
                        ag_com_fee_pct=nm_commission_fee_pct,
                        ag_train_episode=xtrain_episodes)
            st.success('Training DONE!')
            
############################################################################################################
base = alt.Chart(advice_df.reset_index()).encode(
        x = alt.X('Date'),
        y = alt.Y('Close', title='Price  (THB)',
                  scale=alt.Scale(domain=[advice_df['Close'].min()-2,
                                          advice_df['Close'].max()+2])),
        tooltip=[alt.Tooltip('Date',title='Date'),
                 alt.Tooltip('Close',title='Price (THB)')] )
    #####_ACTION_OVERLAY_#####
    buy_plot = alt.Chart(advice_df.reset_index()).encode(
        x = alt.X('Date'),
        y = alt.Y('Close', title='Price  (THB)',
                  scale=alt.Scale(domain=[advice_df['Close'].min()-2,
                                          advice_df['Close'].max()+2])),
        color = alt.Color('position',
                          scale=alt.Scale(domain=['Buy','Sell'],
                                          range=['green','red']),
                          legend=alt.Legend(title="Model Advice")),
        tooltip=[alt.Tooltip('Date', title='Date'),
                 alt.Tooltip('Close', title='Price (THB)'),
                 alt.Tooltip('position', title='Advice')] )
    
    sell_plot = alt.Chart(advice_df.reset_index()).encode(
        x = alt.X('Date'),
        y = alt.Y('Close', title='Price  (THB)',
                  scale=alt.Scale(domain=[advice_df['Close'].min()-2,
                                          advice_df['Close'].max()+2])),
        color = alt.Color('position',
                          scale=alt.Scale(domain=['Buy','Sell'],
                                          range=['green','red']),
                          legend=alt.Legend(title="Model Advice")),
        tooltip=[alt.Tooltip('Date', title='Date'),
                 alt.Tooltip('Close', title='Price (THB)'),
                 alt.Tooltip('position', title='Advice')] )
    #####_LAYERED_CHART_#####
    layer1 = base.mark_line()
    layer2 = base2.mark_circle(size=50).transform_filter(alt.FieldEqualPredicate(field='exposure',equal=True))
    bundle = alt.layer(layer1,layer2).configure_axis(labelFontSize=16,titleFontSize=18)
    #####_SHOW_ADVICE_CHART_#####
    st.altair_chart(bundle, use_container_width=True)
              
