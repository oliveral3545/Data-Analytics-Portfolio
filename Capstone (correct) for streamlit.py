#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px


# In[11]:


btc_macro_df = pd.read_csv('btc_macroeconomic.csv')



# In[15]:





# In[3]:


#np.nan used to stop errors in mathematical operations.
btc_macro_df = btc_macro_df.replace('No data', np.nan)
btc_macro_df


# In[4]:


btc_macro_df['date'] = pd.to_datetime(btc_macro_df['date'], format='%d/%m/%Y')


# In[5]:


feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']

X = btc_macro_df[feature_cols]
y = btc_macro_df['btc_price_usd']

total_rows = len(X)
na_rows = X.isna().any(axis=1).sum()
print(f"Total rows: {total_rows}")
print(f"Rows with at least one NaN: {na_rows} ({na_rows/total_rows:.2%})")

mask = ~X.isna().any(axis=1)

X_clean = X[mask]
y_clean = y[mask]

print(f"Rows after dropping NaNs: {len(X_clean)}")


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, random_state=123)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# In[19]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Function to generate sample data
def generate_data():
    np.random.seed(42)
    gold_prices = np.random.uniform(0, 3000, 100)
    btc_prices = 10000 + 15 * gold_prices + np.random.normal(0, 3000, 100)
    return pd.DataFrame({
        'gold_price_usd': gold_prices,
        'btc_price_usd': btc_prices
    })

# Function to train model
def train_model():
    df = generate_data()
    X = df[['gold_price_usd']]
    y = df['btc_price_usd']
    model = LinearRegression()
    model.fit(X, y)
    return model

def main():
    st.title('BTC price predictor against Macro conditions')
    st.write("Macroeconomics affect BTC's price")  

    # Train model
    model = train_model()

    # User input
    gold_price = st.number_input('Gold Price (USD)', 
                          min_value=0, 
                          max_value=4000, 
                          value=1500)

    if st.button('Predict price'):
        # Perform prediction
        prediction = model.predict([[gold_price]])

        # Show result
        st.success(f'Estimated BTC price: ${prediction[0]:,.2f}')

        # Visualization
        df = generate_data()
        fig = px.scatter(df, x='gold_price_usd', y='btc_price_usd',trendline="ols", 
                       title='Gold Price vs BTC Price Relationship')

        # Add prediction point
        import plotly.graph_objects as go
        fig.add_trace(
            go.Scatter(
                x=[gold_price], 
                y=[prediction[0]], 
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Prediction'
            )
        )

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




