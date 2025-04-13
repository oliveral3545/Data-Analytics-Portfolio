import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to load data
def load_data():
    # Replace with your actual data loading logic
    try:
        df = pd.read_csv('btc_macroeconomic.csv')
        # Ensure date column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        st.warning("btc_macroeconomic.csv not found. Using sample data.")
        dates = pd.date_range(start='2018-01-01', end='2024-10-01', freq='M')
        n = len(dates)
        
        np.random.seed(42)  # For reproducible results
        gold_price = np.random.uniform(1200, 2200, n)
        btc_price = gold_price * 15 + np.random.normal(0, 5000, n)
        sp500 = np.random.uniform(2500, 4800, n)
        fed_rate = np.random.uniform(0, 5, n)
        inflation = np.random.uniform(1, 9, n)
        m2_supply = np.random.uniform(15000, 22000, n)
        
        df = pd.DataFrame({
            'date': dates,
            'gold_price_usd': gold_price,
            'btc_price_usd': btc_price,
            'SP500': sp500,
            'fed_funds_rate': fed_rate,
            'US_inflation': inflation,
            'US_M2_money_supply_in_billions': m2_supply
        })
        
        return df

# Function to preprocess data and ensure it's suitable for training
def preprocess_data(df, feature_cols, target_col='btc_price_usd'):
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure all feature columns and target column exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df_copy.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
        return None, None, None
    
    # Remove rows with NaN values in features or target
    df_clean = df_copy.dropna(subset=feature_cols + [target_col])
    
    if len(df_clean) < 10:
        st.error("Not enough clean data points for reliable model training")
        return None, None, None
    
    # Ensure all data is numeric (convert to float to avoid issues with integers)
    for col in feature_cols + [target_col]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Drop any rows that couldn't be converted to numeric
    df_clean = df_clean.dropna(subset=feature_cols + [target_col])
    
    # Extract features and target
    X = df_clean[feature_cols].astype(float).values  # Explicit conversion to numpy array of floats
    y = df_clean[target_col].astype(float).values    # Explicit conversion to numpy array of floats
    
    return X, y, df_clean

# Function to train the model
def train_model(df, feature_cols=['gold_price_usd']):
    # Preprocess data
    X, y, df_clean = preprocess_data(df, feature_cols)
    
    if X is None or y is None:
        return None, None, None
    
    try:
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        
        return model, r_squared, df_clean
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None

# Function to generate visualization data
def generate_prediction_data(model, feature_col, min_val, max_val):
    if model is None:
        return None
    
    # Create range for prediction line
    x_range = np.linspace(min_val, max_val, 100)
    x_pred = x_range.reshape(-1, 1)  # Reshape for prediction
    
    try:
        y_pred = model.predict(x_pred)
        
        pred_df = pd.DataFrame({
            feature_col: x_range,
            'btc_price_predicted': y_pred
        })
        
        return pred_df
    except Exception as e:
        st.error(f"Error generating prediction data: {e}")
        return None

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Explore how macroeconomic factors affect Bitcoin's price")
    
    # Load data
    btc_macro_df = load_data()
    
    if btc_macro_df is None or btc_macro_df.empty:
        st.error("Failed to load data. Please check your data source.")
        return
    
    # Sidebar for feature selection
    st.sidebar.header("Model Configuration")
    
    # Identify available numeric columns for features
    numeric_cols = btc_macro_df.select_dtypes(include=['number']).columns.tolist()
    # Filter out btc_price_usd as it's our target
    feature_options = [col for col in numeric_cols if col != 'btc_price_usd']
    
    if not feature_options:
        st.error("No numeric feature columns found in the dataset.")
        return
    
    # By default, use gold_price_usd if available, otherwise the first feature
    default_index = 0
    if 'gold_price_usd' in feature_options:
        default_index = feature_options.index('gold_price_usd')
    
    selected_feature = st.sidebar.selectbox(
        'Select Macro Feature for Prediction',
        feature_options,
        index=default_index
    )
    
    # Display data overview
    with st.expander("View Data Overview"):
        st.dataframe(btc_macro_df.describe())
        
        # Display data statistics
        st.write(f"Total data points: {len(btc_macro_df)}")
        if selected_feature in btc_macro_df.columns and 'btc_price_usd' in btc_macro_df.columns:
            missing_values = btc_macro_df[[selected_feature, 'btc_price_usd']].isna().sum()
            st.write(f"Missing values in {selected_feature}: {missing_values[selected_feature]}")
            st.write(f"Missing values in BTC price: {missing_values['btc_price_usd']}")
    
    # Train model
    model, r_squared, clean_df = train_model(btc_macro_df, [selected_feature])
    
    if model is None or clean_df is None or clean_df.empty:
        st.error("Could not train model. Please check your data.")
        return
    
    # Display model info
    st.write(f"Model R-squared: {r_squared:.4f}")
    st.write(f"Coefficient for {selected_feature}: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.2f}")
    
    # User input for prediction
    st.subheader("Make a Prediction")
    
    # Get min/max values for the slider
    min_val = float(clean_df[selected_feature].min())
    max_val = float(clean_df[selected_feature].max())
    current_val = float(clean_df[selected_feature].median())
    
    # Input widget
    input_value = st.slider(
        f'{selected_feature}',
        min_value=min_val,
        max_value=max_val,
        value=current_val,
        step=(max_val - min_val) / 100
    )
    
    # Perform prediction
    try:
        prediction = model.predict([[float(input_value)]])[0]
        
        # Show result
        st.success(f'Estimated BTC price: ${prediction:,.2f}')
        
        # Visualization
        st.subheader("Data Visualization")
        
        # Create scatter plot
        fig = px.scatter(clean_df, x=selected_feature, y='btc_price_usd',
                        title=f'{selected_feature} vs BTC Price Relationship',
                        opacity=0.7)
        
        # Add regression line
        pred_data = generate_prediction_data(model, selected_feature, min_val, max_val)
        if pred_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=pred_data[selected_feature],
                    y=pred_data['btc_price_predicted'],
                    mode='lines',
                    name='Regression Line',
                    line=dict(color='blue')
                )
            )
        
        # Add prediction point
        fig.add_trace(
            go.Scatter(
                x=[input_value],
                y=[prediction],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Prediction'
            )
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title=selected_feature,
            yaxis_title='BTC Price (USD)',
            height=600
        )
        
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
    
    # Add disclaimer
    st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")

if __name__ == '__main__':
    main()
