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

# Function to train the model and calculate RMSE
def train_model(df, feature_cols):
    # Preprocess data
    X, y, df_clean = preprocess_data(df, feature_cols)
    
    if X is None or y is None:
        return None, None, None, None
    
    try:
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        
        # Calculate RMSE (Root Mean Square Error)
        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        return model, r_squared, rmse, df_clean
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return None, None, None, None

# Function to make a prediction with multiple features
def make_prediction(model, feature_values):
    if model is None:
        return None
    
    try:
        # Ensure all features are float
        features = [float(val) for val in feature_values]
        # Reshape for sklearn
        features_array = np.array(features).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features_array)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    st.title('BTC Price Predictor Against Macro Conditions')
    st.write("Explore how macroeconomic factors affect Bitcoin's price")
    
    # Load data
    btc_macro_df = load_data()
    
    if btc_macro_df is None or btc_macro_df.empty:
        st.error("Failed to load data. Please check your data source.")
        return
    
    # Define the specific macro features to use
    macro_features = [
        'gold_price_usd',
        'SP500',
        'fed_funds_rate',
        'US_inflation',
        'US_M2_money_supply_in_billions'
    ]
    
    # Verify which features are available in the dataset
    available_features = [feat for feat in macro_features if feat in btc_macro_df.columns]
    
    if not available_features:
        st.error("None of the required macro features are in the dataset.")
        # Show available columns
        st.write("Available columns:", ", ".join(btc_macro_df.columns.tolist()))
        return
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Let user select features to include
    st.sidebar.subheader("Select Features to Include")
    selected_features = []
    
    for feature in available_features:
        if st.sidebar.checkbox(feature, value=True, key=f"feature_{feature}"):
            selected_features.append(feature)
    
    if not selected_features:
        st.error("Please select at least one feature for prediction.")
        return
    
    # Remove basic data statistics completely
    
    # Train model with selected features
    model, r_squared, rmse, clean_df = train_model(btc_macro_df, selected_features)
    
    if model is None or clean_df is None or clean_df.empty:
        st.error("Could not train model. Please check your data.")
        return
    
    # Display model info
    st.subheader("Model Information")
    st.write(f"Model R-squared: {r_squared:.4f}")
    st.write(f"RMSE (Root Mean Square Error): ${rmse:,.2f}")
    
    coef_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': model.coef_
    })
    st.write("Feature Coefficients:")
    st.dataframe(coef_df)
    
    # User input for prediction
    st.subheader("Make a Prediction")
    
    # Create input sliders for each feature
    feature_values = []
    
    for feature in selected_features:
        min_val = float(clean_df[feature].min())
        max_val = float(clean_df[feature].max())
        current_val = float(clean_df[feature].median())
        
        feature_val = st.slider(
            f'{feature}',
            min_value=min_val,
            max_value=max_val,
            value=current_val,
            step=(max_val - min_val) / 100,
            key=f"slider_{feature}"
        )
        
        feature_values.append(feature_val)
    
    # Predict button
    if st.button("Predict BTC Price"):
        prediction = make_prediction(model, feature_values)
        
        if prediction is not None:
            # Show result without prediction range
            st.success(f'Estimated BTC price: ${prediction:,.2f}')
            
            # Display feature importance (simple version - just use absolute coefficients)
            importance = np.abs(model.coef_)
            importance_normalized = importance / np.sum(importance)
            
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importance_normalized
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Feature Importance")
            fig_importance = px.bar(
                importance_df, 
                x='Feature', 
                y='Importance',
                title='Relative Importance of Each Feature'
            )
            st.plotly_chart(fig_importance)
    
    # Add a correlation matrix for selected features
    if len(selected_features) > 1:
        st.subheader("Feature Correlation Matrix")
        corr_matrix = clean_df[selected_features + ['btc_price_usd']].corr().round(2)  # Round to 2 decimal places
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',  # Format to always show 2 decimal places
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr)
    
    # Add disclaimer
    st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")

if __name__ == '__main__':
    main()
