import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
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
        return None

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

def train_model(df, feature_cols, random_state=123):
    # Preprocess data
    X, y, df_clean = preprocess_data(df, feature_cols)
    
    if X is None or y is None:
        return None, None, None, None
    
    try:
        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate R-squared on test data
        r_squared = model.score(X_test, y_test)
        
        # Calculate RMSE on test data
        y_pred = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
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
    st.write("""This model demonstrates how Bitcoin's price dynamics have evolved beyond the traditional 4-year cycle narrative in 2025. 
    It highlights the increasing influence of macroeconomic factors on BTC's valuation. Use the sliders to explore various economic scenarios—from 
    highly favorable to challenging conditions—and observe their significant impact on Bitcoin's predicted price movement.""")
    
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
    
    # Train model with selected features
    try:
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

        # Create expandable sections for each feature
        with st.container():
            st.markdown("**1. Inflation (coefficient: 541.73):**")
            st.write("For each percentage point increase in inflation, Bitcoin price is estimated to increase by $541.73, on average. This supports the narrative that Bitcoin serves as an inflation hedge - when fiat currencies lose purchasing power, Bitcoin tends to gain value.")
    
            st.markdown("**2. Fed funds rate (coefficient: -3,393.99):**")
            st.write("This large negative coefficient indicates that for each percentage point increase in the Fed funds rate, Bitcoin price is estimated to decrease by approximately $3,394. This relationship makes economic sense as higher interest rates:")
            st.markdown("""
            - Make yield-generating assets more attractive compared to non-yielding Bitcoin
            - Reduce liquidity in the financial system
            - Typically suppress risk appetite in markets
            """)
    
            st.markdown("**3. S&P 500 (coefficient: 27.39):**")
            st.write("For each point increase in the S&P 500 index, Bitcoin price tends to increase by about $27.39. This positive correlation suggests Bitcoin still behaves partially as a risk asset that rises with broader market optimism.")
    
            st.markdown("**4. Gold price (coefficient: 4.99):**")
            st.write("For each dollar increase in gold's price, Bitcoin price tends to increase by $4.99. This modest positive relationship suggests some connection between the two assets, supporting the \"digital gold\" narrative, but the effect is relatively small.")
    
            st.markdown("**5. M2 money supply (coefficient: -2.71):**")
            st.write("For each billion dollar increase in M2 money supply, Bitcoin price tends to decrease by $2.71. This slight negative relationship is counterintuitive since Bitcoin is often positioned as a hedge against monetary expansion.There might be lag effects not captured in the current model")

        
        # User input for prediction
        st.subheader("Make a Prediction")
        
        # Create input sliders for each feature
        feature_values = []
        
        # Make sure to use the actual column names from your dataset
        for feature in selected_features:  # Use the features that were selected for training
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
                st.success(f'Estimated BTC price: ${prediction:,.2f}')
        
        # Add disclaimer
        st.info("Disclaimer: This tool is for educational purposes only. Cryptocurrency investments carry significant risk.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == '__main__':
    main()


