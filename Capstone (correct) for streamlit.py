import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Function to load real data, with fallback to synthetic data
def load_data():
    try:
        # Attempt to read the CSV file
        df = pd.read_csv('btc_macroeconomic.csv')
        
        # Clean up the data
        df = df.replace('No data', np.nan)
        
        # Convert date to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            
        # Drop rows with NaN values in the features we need
        feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']
        df_clean = df.dropna(subset=feature_cols + ['btc_price_usd'])
        
        st.success("Successfully loaded real BTC macro data!")
        return df_clean
    except Exception as e:
        st.warning(f"Could not load CSV file: {e}. Using synthetic data instead.")
        # Fall back to synthetic data if loading fails
        return generate_synthetic_data()

# Function to generate synthetic data as a fallback
def generate_synthetic_data():
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
        'SP500': sp500_values,
        'fed_funds_rate': fed_funds_rates,
        'US_M2_money_supply_in_billions': us_m2_supply,
        'US_inflation': us_inflation,
        'btc_price_usd': btc_prices
    })

# Function to train model with actual data
def train_model():
    df = load_data()
    # Make sure column names match those in your CSV
    X = df[['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']]
    y = df['btc_price_usd']
    model = LinearRegression()
    model.fit(X, y)
    return model, df

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Macroeconomics affect BTC's price")
    
    # Train model with all features
    model, df = train_model()
    
    # Show data summary
    with st.expander("View Data Summary"):
        st.write("Number of data points:", len(df))
        st.write("BTC Price Range:", f"${df['btc_price_usd'].min():,.2f} to ${df['btc_price_usd'].max():,.2f}")
        st.dataframe(df.describe())
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # Get min and max values from the actual data for input ranges
    gold_min, gold_max = float(df['gold_price_usd'].min()), float(df['gold_price_usd'].max())
    sp500_min, sp500_max = float(df['SP500'].min()), float(df['SP500'].max())
    fed_min, fed_max = float(df['fed_funds_rate'].min()), float(df['fed_funds_rate'].max())
    m2_min, m2_max = float(df['US_M2_money_supply_in_billions'].min()), float(df['US_M2_money_supply_in_billions'].max())
    infl_min, infl_max = float(df['US_inflation'].min()), float(df['US_inflation'].max())
    
    # User inputs
    with col1:
        gold_price = st.number_input('Gold Price (USD)', 
                                min_value=gold_min, 
                                max_value=gold_max, 
                                value=(gold_min + gold_max)/2)
        
        sp500 = st.number_input('S&P 500 Index', 
                               min_value=sp500_min, 
                               max_value=sp500_max, 
                               value=(sp500_min + sp500_max)/2)
                               
        fed_rate = st.number_input('Fed Funds Rate (%)', 
                                 min_value=fed_min, 
                                 max_value=fed_max, 
                                 value=(fed_min + fed_max)/2,
                                 step=0.1)
    
    with col2:
        m2_supply = st.number_input('US M2 Money Supply (Billions $)', 
                                   min_value=m2_min, 
                                   max_value=m2_max, 
                                   value=(m2_min + m2_max)/2)
        
        inflation = st.number_input('US Inflation Rate (%)', 
                                   min_value=infl_min, 
                                   max_value=infl_max, 
                                   value=(infl_min + infl_max)/2,
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
        feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_M2_money_supply_in_billions', 'US_inflation']
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
        
        # Add time series chart if date column exists
        if 'date' in df.columns:
            st.subheader('BTC Price Over Time')
            time_fig = px.line(df.sort_values('date'), x='date', y='btc_price_usd',
                           title='Bitcoin Price Historical Trend')
            st.plotly_chart(time_fig)

if __name__ == '__main__':
    main()
