# Replace the generate_data() function with this:
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
        
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Fall back to synthetic data if loading fails
        return generate_data()

# Then update your train_model() function:
def train_model():
    df = load_data()
    # Make sure column names match those in your CSV
    X = df[['gold_price_usd', 'SP500', 'fed_funds_rate', 'US_inflation', 'US_M2_money_supply_in_billions']]
    y = df['btc_price_usd']
    model = LinearRegression()
    model.fit(X, y)
    return model, df
