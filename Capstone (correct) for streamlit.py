import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math

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
    gold_prices = np.random.uniform(1000, 2500, n_samples)
    sp500_values = np.random.uniform(2000, 4500, n_samples)
    fed_funds_rates = np.random.uniform(0, 5, n_samples)
    us_m2_supply = np.random.uniform(11000, 22000, n_samples)
    us_inflation = np.random.uniform(-1, 9, n_samples)
    
    # Generate BTC prices with dependencies on all features (more realistic)
    btc_prices = (10000 + 
                 5 * gold_prices + 
                 2 * sp500_values +
                 -1000 * fed_funds_rates + 
                 500 * us_inflation +
                 0.1 * us_m2_supply +
                 np.random.normal(0, 2000, n_samples))
    
    return pd.DataFrame({
        'gold_price_usd': gold_prices,
        'SP500': sp500_values,
        'fed_funds_rate': fed_funds_rates,
        'US_M2_money_supply_in_billions': us_m2_supply,
        'US_inflation': us_inflation,
        'btc_price_usd': btc_prices
    })

# Function to train model with feature scaling
def train_model(model_type='linear'):
    df = load_data()
    
    # Check for outliers in BTC price
    q1 = df['btc_price_usd'].quantile(0.25)
    q3 = df['btc_price_usd'].quantile(0.75)
    iqr = q3 - q1
    
    # Filter out extreme outliers
    df_filtered = df[
        (df['btc_price_usd'] >= q1 - 1.5 * iqr) & 
        (df['btc_price_usd'] <= q3 + 1.5 * iqr)
    ]
    
    if len(df_filtered) < len(df):
        st.info(f"Removed {len(df) - len(df_filtered)} outliers from the dataset.")
        df = df_filtered
    
    # Make sure column names match those in your CSV
    feature_cols = ['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']
    X = df[feature_cols]
    y = df['btc_price_usd']
    
    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model based on selected type
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, df, feature_cols, rmse, r2

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Macroeconomics affect BTC's price")
    
    # Model selection
    model_type = st.radio(
        "Select prediction model:",
        ["Linear Regression", "Random Forest"],
        index=1,  # Default to Random Forest for better predictions
        horizontal=True
    )
    
    model_key = 'linear' if model_type == "Linear Regression" else 'random_forest'
    
    # Train model with all features
    model, scaler, df, feature_cols, rmse, r2 = train_model(model_key)
    
    # Show model performance
    st.info(f"Model performance: RMSE = ${rmse:.2f}, R² = {r2:.3f}")
    
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
        # Create input array with all features
        input_features = np.array([[gold_price, sp500, fed_rate, m2_supply, inflation]])
        
        # Scale the input features using the same scaler used for training
        input_scaled = scaler.transform(input_features)
        
        # Perform prediction
        prediction = model.predict(input_scaled)
        
        # Show result
        st.success(f'Estimated BTC price: ${prediction[0]:,.2f}')
        
        # For Linear Regression, show feature importance
        if model_type == "Linear Regression":
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
        else:
            # For Random Forest, show feature importance
            feature_names = ['Gold Price', 'S&P 500', 'Fed Funds Rate', 'M2 Supply', 'Inflation']
            importances = model.feature_importances_
            
            # Create bar chart of feature importances
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.subheader('Feature Importance')
            fig_imp = px.bar(imp_df, x='Feature', y='Importance',
                         title='Importance of Macro Factors for BTC Price')
            st.plotly_chart(fig_imp)
        
        # Compare your inputs to historical data
        input_df = pd.DataFrame({
            'Gold Price': [gold_price],
            'S&P 500': [sp500],
            'Fed Funds Rate': [fed_rate],
            'M2 Supply': [m2_supply],
            'Inflation': [inflation]
        })
        
        # Show where the inputs fall within historical ranges
        st.subheader('Your Inputs vs Historical Ranges')
        for feature, value in zip(feature_names, input_features[0]):
            col_name = feature_cols[feature_names.index(feature)]
            hist_min = df[col_name].min()
            hist_max = df[col_name].max()
            hist_mean = df[col_name].mean()
            
            # Calculate percentile of input
            percentile = (value - hist_min) / (hist_max - hist_min) * 100 if hist_max > hist_min else 50
            
            st.write(f"**{feature}**: Your input of {value:.2f} is at the {percentile:.1f}th percentile of historical data.")
        
        # Create scatter plot of actual vs predicted BTC prices
        X_scaled = scaler.transform(df[feature_cols])
        df['predicted_price'] = model.predict(X_scaled)
        
        fig = px.scatter(df, x='btc_price_usd', y='predicted_price',
                      title='Actual vs Predicted BTC Prices',
                      labels={
                          'btc_price_usd': 'Actual BTC Price',
                          'predicted_price': 'Predicted BTC Price'
                      })
        
        # Add diagonal line for perfect predictions
        max_val = max(df['btc_price_usd'].max(), df['predicted_price'].max())
        min_val = min(df['btc_price_usd'].min(), df['predicted_price'].min())
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
                x=[prediction[0]],
                y=[prediction[0]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='star'),
                name='Your Prediction'
            )
        )
        
        st.plotly_chart(fig)
        
        # Add time series chart if date column exists
        if 'date' in df.columns:
            st.subheader('BTC Price Over Time')
            time_fig = px.line(df.sort_values('date'), x='date', y=['btc_price_usd', 'predicted_price'],
                           title='Bitcoin Price: Actual vs Model Predictions',
                           labels={
                               'value': 'Price (USD)',
                               'variable': 'Data Type'
                           })
            st.plotly_chart(time_fig)

if __name__ == '__main__':
    main()
