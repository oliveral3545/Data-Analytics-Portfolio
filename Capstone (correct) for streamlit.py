import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Function to generate sample data with all macro variables
def generate_data():
    np.random.seed(42)
    n_samples = 100
    
    # Generate features within realistic ranges
    gold_prices = np.random.uniform(0, 3000, n_samples)
    sp500_values = np.random.uniform(2000, 5000, n_samples)
    fed_funds_rates = np.random.uniform(0, 5, n_samples)
    us_m2_supply = np.random.uniform(11000, 22000, n_samples)
    us_inflation = np.random.uniform(-1, 9, n_samples)
    
    # Generate BTC prices with dependencies on all features
    btc_prices = (10000 + 
                 15 * gold_prices + 
                 3 * sp500_values +
                 -2000 * fed_funds_rates + 
                 1000 * us_inflation +
                 0.5 * us_m2_supply +
                 np.random.normal(0, 3000, n_samples))
    
    return pd.DataFrame({
        'gold_price_usd': gold_prices,
        'sp500': sp500_values,
        'fed_funds_rate': fed_funds_rates,
        'us_m2_supply_billions': us_m2_supply,
        'us_inflation': us_inflation,
        'btc_price_usd': btc_prices
    })

# Function to train model with all features
def train_model():
    df = generate_data()
    X = df[['gold_price_usd', 'sp500', 'fed_funds_rate', 'us_m2_supply_billions', 'us_inflation']]
    y = df['btc_price_usd']
    model = LinearRegression()
    model.fit(X, y)
    return model, df

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Macroeconomics affect BTC's price")
    
    # Train model with all features
    model, df = train_model()
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # User inputs
    with col1:
        gold_price = st.number_input('Gold Price (USD)', 
                                min_value=0, 
                                max_value=4000, 
                                value=1500)
        
        sp500 = st.number_input('S&P 500 Index', 
                               min_value=1500, 
                               max_value=6000, 
                               value=3000)
                               
        fed_rate = st.number_input('Fed Funds Rate (%)', 
                                 min_value=0.0, 
                                 max_value=10.0, 
                                 value=2.5,
                                 step=0.1)
    
    with col2:
        m2_supply = st.number_input('US M2 Money Supply (Billions $)', 
                                   min_value=10000, 
                                   max_value=25000, 
                                   value=18000)
        
        inflation = st.number_input('US Inflation Rate (%)', 
                                   min_value=-2.0, 
                                   max_value=10.0, 
                                   value=2.0,
                                   step=0.1)
    
    if st.button('Predict BTC Price'):
        # Create input array with all features for prediction
        input_features = [[gold_price, sp500, fed_rate, m2_supply, inflation]]
        
        # Perform prediction
        prediction = model.predict(input_features)
        
        # Show result
        st.success(f'Estimated BTC price: ${prediction[0]:,.2f}')
        
        # Calculate feature importance (coefficients)
        feature_names = ['Gold Price', 'S&P 500', 'Fed Funds Rate', 'M2 Supply', 'Inflation']
        coefficients = model.coef_
        
        # Create bar chart of coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': coefficients
        })
        
        st.subheader('Feature Importance')
        fig_coef = px.bar(coef_df, x='Feature', y='Impact',
                     title='Impact of Macro Factors on BTC Price')
        st.plotly_chart(fig_coef)
        
        # Create visualization for the most influential feature
        most_important_idx = abs(coefficients).argmax()
        most_important_feature = feature_names[most_important_idx]
        feature_cols = ['gold_price_usd', 'sp500', 'fed_funds_rate', 'us_m2_supply_billions', 'us_inflation']
        most_important_col = feature_cols[most_important_idx]
        
        fig = px.scatter(df, x=most_important_col, y='btc_price_usd',
                       trendline="ols",
                       title=f'{most_important_feature} vs BTC Price Relationship')
        
        # Add prediction point
        fig.add_trace(
            go.Scatter(
                x=[input_features[0][most_important_idx]],
                y=[prediction[0]],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Your Prediction'
            )
        )
        
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()

