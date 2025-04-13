import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Function to load data
def load_data():
    # Replace with your actual data loading logic
    # For now, let's create a sample dataset
    try:
        df = pd.read_csv('btc_macroeconomic.csv')
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

# Function to train the model
def train_model(df, feature_cols=['gold_price_usd']):
    # Remove rows with NaN values
    mask = ~df[feature_cols + ['btc_price_usd']].isna().any(axis=1)
    df_clean = df[mask]
    
    if len(df_clean) < 10:
        st.error("Not enough clean data points for reliable model training")
        return None
    
    X = df_clean[feature_cols]
    y = df_clean['btc_price_usd']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    return model, r_squared, df_clean

# Function to generate visualization data
def generate_prediction_data(df_clean, feature_col='gold_price_usd', model=None):
    if model is None:
        return df_clean
    
    min_val = df_clean[feature_col].min()
    max_val = df_clean[feature_col].max()
    
    # Create range for prediction line
    x_range = np.linspace(min_val, max_val, 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    
    pred_df = pd.DataFrame({
        feature_col: x_range,
        'btc_price_predicted': y_pred
    })
    
    return pred_df

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Explore how macroeconomic factors affect Bitcoin's price")
    
    # Load data
    btc_macro_df = load_data()
    
    # Sidebar for feature selection
    st.sidebar.header("Model Configuration")
    
    feature_options = [
        'gold_price_usd', 
        'SP500', 
        'fed_funds_rate', 
        'US_inflation', 
        'US_M2_money_supply_in_billions'
    ]
    
    # By default, only use gold price as predictor
    selected_feature = st.sidebar.selectbox(
        'Select Macro Feature for Prediction',
        feature_options,
        index=0
    )
    
    # Display data overview
    with st.expander("View Data Overview"):
        st.dataframe(btc_macro_df.describe())
        
        # Display data statistics
        st.write(f"Total data points: {len(btc_macro_df)}")
        missing_values = btc_macro_df[[selected_feature, 'btc_price_usd']].isna().sum()
        st.write(f"Missing values in {selected_feature}: {missing_values[selected_feature]}")
        st.write(f"Missing values in BTC price: {missing_values['btc_price_usd']}")
    
    # Train model
    model, r_squared, clean_df = train_model(btc_macro_df, [selected_feature])
    
    if model is None:
        st.error("Could not train model. Please check your data.")
        return
    
    # Display model info
    st.write(f"Model R-squared: {r_squared:.4f}")
    st.write(f"Coefficient for {selected_feature}: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.2f}")
    
    # User input for prediction
    st.subheader("Make a Prediction")
    
    # Get min/max values for the slider
    min_val = clean_df[selected_feature].min()
    max_val = clean_df[selected_feature].max()
    current_val = clean_df[selected_feature].median()
    
    # Input widget
    input_value = st.slider(
        f'{selected_feature}',
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(current_val),
        step=float((max_val - min_val) / 100)
    )
    
    # Perform prediction
    prediction = model.predict([[input_value]])[0]
    
    # Show result
    st.success(f'Estimated BTC price: ${prediction:,.2f}')
    
    # Visualization
    st.subheader("Data Visualization")
    
    # Create scatter plot
    fig = px.scatter(clean_df, x=selected_feature, y='btc_price_usd',
                    title=f'{selected_feature} vs BTC Price Relationship',
                    opacity=0.7)
    
    # Add regression line
    pred_data = generate_prediction_data(clean_df, selected_feature, model)
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
    
    # Add disclaimer
    st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")

if __name__ == '__main__':
    main()
