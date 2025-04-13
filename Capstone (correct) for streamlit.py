import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to load and process the real BTC macro data
def load_data():
    try:
        # Load the BTC macro data
        btc_macro_df = pd.read_csv('btc_macroeconomic.csv')
        
        # Clean up the data - convert all non-numeric values to NaN
        btc_macro_df = btc_macro_df.replace(['No data', 'No', 'NaN', 'nan', 'NULL', 'null', ''], np.nan)
        
        # Convert date to datetime if it exists
        if 'date' in btc_macro_df.columns:
            btc_macro_df['date'] = pd.to_datetime(btc_macro_df['date'], format='%d/%m/%Y', errors='coerce')
            
        # Ensure all numeric columns are properly converted to float
        feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions', 'btc_price_usd']
        for col in feature_cols:
            if col in btc_macro_df.columns:
                btc_macro_df[col] = pd.to_numeric(btc_macro_df[col], errors='coerce')
        
        return btc_macro_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to train the model
def train_model():
    # Load the real data
    btc_macro_df = load_data()
    
    if btc_macro_df is None:
        st.error("Failed to load the data file. Please make sure 'btc_macroeconomic.csv' is available.")
        st.stop()
    
    # Use all macro features for prediction
    feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']
    X = btc_macro_df[feature_cols]
    y = btc_macro_df['btc_price_usd']
    
    # Count rows with missing values
    total_rows = len(X)
    na_rows = X.isna().any(axis=1).sum()
    st.write(f"Total rows: {total_rows}")
    st.write(f"Rows with at least one NaN: {na_rows} ({na_rows/total_rows:.2%})")
    
    # Filter out rows with missing values
    mask = ~X.isna().any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    st.write(f"Rows after dropping NaNs: {len(X_clean)}")
    
    # Check if we have enough data
    if len(X_clean) < 10:
        st.error("Not enough complete data rows for reliable modeling.")
        st.stop()
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, random_state=123)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, btc_macro_df[mask], feature_cols

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Macroeconomics affect BTC's price")
    
    # Train model with real data
    model, clean_data, feature_cols = train_model()
    
    # Show data summary
    with st.expander("View Data Summary"):
        st.write("Number of clean data points:", len(clean_data))
        st.write("BTC Price Range:", f"${clean_data['btc_price_usd'].min():,.2f} to ${clean_data['btc_price_usd'].max():,.2f}")
        st.dataframe(clean_data.describe())
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # Get min and max values from the actual data for input ranges
    gold_min, gold_max = float(clean_data['gold_price_usd'].min()), float(clean_data['gold_price_usd'].max())
    sp500_min, sp500_max = float(clean_data['SP500'].min()), float(clean_data['SP500'].max())
    fed_min, fed_max = float(clean_data['fed_funds_rate'].min()), float(clean_data['fed_funds_rate'].max())
    m2_min, m2_max = float(clean_data['US_M2_money_supply_in_billions'].min()), float(clean_data['US_M2_money_supply_in_billions'].max())
    infl_min, infl_max = float(clean_data['US_inflation'].min()), float(clean_data['US_inflation'].max())
    
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
        # Create input array with all features
        input_features = [[gold_price, sp500, fed_rate, m2_supply, inflation]]
        
        # Perform prediction
        prediction = model.predict(input_features)[0]
        
          
        # Show result
        st.success(f'Estimated BTC price: ${prediction:,.2f}')
        
        # Show feature importance
        feature_names = ['Gold Price', 'S&P 500', 'Fed Funds Rate', 'M2 Supply', 'Inflation']
        coefficients = model.coef_
        
        # Create bar chart of coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': coefficients
        })
        
        st.subheader('Feature Importance (Coefficients)')
        fig_coef = px.bar(coef_df, x='Feature', y='Impact',
                     title='Impact of Macro Factors on BTC Price')
        st.plotly_chart(fig_coef)
        
        # Create visualization with the actual data
        fig = px.scatter(clean_data, x='gold_price_usd', y='btc_price_usd', 
                       title='Gold Price vs BTC Price Relationship')
        
        # Add prediction point
        fig.add_trace(
            go.Scatter(
                x=[gold_price], 
                y=[prediction], 
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Prediction'
            )
        )
        
        st.plotly_chart(fig)
        
        # If date column exists, show time series
        if 'date' in clean_data.columns:
            st.subheader('BTC Price Over Time')
            time_fig = px.line(clean_data.sort_values('date'), x='date', y='btc_price_usd',
                           title='Bitcoin Price Historical Trend')
            st.plotly_chart(time_fig)

if __name__ == '__main__':
    main()
