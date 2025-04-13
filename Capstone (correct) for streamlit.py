import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math

# Function to load real data only
def load_data():
    try:
        # Attempt to read the CSV file
        df = pd.read_csv('btc_macroeconomic.csv')
        
        # Clean up the data - convert all non-numeric values to NaN
        df = df.replace(['No data', 'No', 'NaN', 'nan', 'NULL', 'null', ''], np.nan)
        
        # Convert date to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
            
        # Ensure all numeric columns are properly converted to float
        feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions', 'btc_price_usd']
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Count rows with NaN values
        total_rows = len(df)
        feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']
        X = df[feature_cols]
        na_rows = X.isna().any(axis=1).sum()
        
        st.write(f"Total rows in dataset: {total_rows}")
        st.write(f"Rows with at least one NaN: {na_rows} ({na_rows/total_rows:.2%})")
        
        if total_rows > 0:
            st.success(f"Successfully loaded BTC macro data!")
            return df
        else:
            st.error("No data found in the CSV file.")
            return None
            
    except Exception as e:
        st.error(f"Could not load CSV file properly: {e}")
        return None

# Function to train model with proper NaN handling
def train_model():
    df = load_data()
    
    if df is None:
        st.error("Cannot train model: no data available.")
        st.stop()
    
    # Define feature columns
    feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']
    X = df[feature_cols]
    y = df['btc_price_usd']
    
    # Handle missing values exactly as in your code
    mask = ~X.isna().any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    
    st.write(f"Rows after dropping NaNs: {len(X_clean)}")
    
    if len(X_clean) < 10:
        st.error("Not enough clean data rows for reliable modeling (minimum 10 required).")
        st.stop()
    
    # Check for outliers in BTC price
    q1 = y_clean.quantile(0.25)
    q3 = y_clean.quantile(0.75)
    iqr = q3 - q1
    
    # Filter out extreme outliers
    outlier_mask = (y_clean >= q1 - 1.5 * iqr) & (y_clean <= q3 + 1.5 * iqr)
    X_filtered = X_clean[outlier_mask]
    y_filtered = y_clean[outlier_mask]
    
    if len(X_filtered) < len(X_clean):
        st.info(f"Removed {len(X_clean) - len(X_filtered)} outliers from the dataset.")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=123)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, df[mask], feature_cols, rmse, r2

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Macroeconomics affect BTC's price")
    
    # Train model with all features
    try:
        model, df_clean, feature_cols, rmse, r2 = train_model()
    except:
        st.error("Failed to train model. Please ensure your data file is correctly formatted.")
        st.stop()
    
    # Show model performance
    st.info(f"Model performance: RMSE = ${rmse:.2f}, R² = {r2:.3f}")
    
    # Show data summary
    with st.expander("View Data Summary"):
        st.write("Number of clean data points:", len(df_clean))
        st.write("BTC Price Range:", f"${df_clean['btc_price_usd'].min():,.2f} to ${df_clean['btc_price_usd'].max():,.2f}")
        st.dataframe(df_clean.describe())
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # Get min and max values from the actual data for input ranges
    gold_min, gold_max = float(df_clean['gold_price_usd'].min()), float(df_clean['gold_price_usd'].max())
    sp500_min, sp500_max = float(df_clean['SP500'].min()), float(df_clean['SP500'].max())
    fed_min, fed_max = float(df_clean['fed_funds_rate'].min()), float(df_clean['fed_funds_rate'].max())
    m2_min, m2_max = float(df_clean['US_M2_money_supply_in_billions'].min()), float(df_clean['US_M2_money_supply_in_billions'].max())
    infl_min, infl_max = float(df_clean['US_inflation'].min()), float(df_clean['US_inflation'].max())
    
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
        
        # Constrain prediction to be within a reasonable range of historical values
        min_btc = df_clean['btc_price_usd'].min()
        max_btc = df_clean['btc_price_usd'].max()
        
        if prediction < min_btc * 0.5:
            st.warning(f"Raw prediction (${prediction:.2f}) was below historical minimum. Adjusting to a reasonable value.")
            prediction = min_btc * 0.5
        elif prediction > max_btc * 1.5:
            st.warning(f"Raw prediction (${prediction:.2f}) was above historical maximum. Adjusting to a reasonable value.")
            prediction = max_btc * 1.5
        
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
            
        # Create scatter plot of actual vs predicted BTC prices
        df_clean['predicted_price'] = model.predict(df_clean[feature_cols])
        
        fig = px.scatter(df_clean, x='btc_price_usd', y='predicted_price',
                      title='Actual vs Predicted BTC Prices',
                      labels={
                          'btc_price_usd': 'Actual BTC Price',
                          'predicted_price': 'Predicted BTC Price'
                      })
        
        # Add diagonal line for perfect predictions
        max_val = max(df_clean['btc_price_usd'].max(), df_clean['predicted_price'].max())
        min_val = min(df_clean['btc_price_usd'].min(), df_clean['predicted_price'].min())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        # Add prediction point
        fig.add_trace(
            go.Scatter(
                x=[prediction],
                y=[prediction],
                mode='markers',
                marker=dict(size=15, color='green', symbol='star'),
                name='Your Prediction'
            )
        )
        
        st.plotly_chart(fig)
        
        # Add time series chart if date column exists
        if 'date' in df_clean.columns:
            st.subheader('BTC Price Over Time')
            time_fig = px.line(df_clean.sort_values('date'), x='date', y=['btc_price_usd', 'predicted_price'],
                           title='Bitcoin Price: Actual vs Model Predictions',
                           labels={
                               'value': 'Price (USD)',
                               'variable': 'Data Type'
                           })
            st.plotly_chart(time_fig)

if __name__ == '__main__':
    main()
