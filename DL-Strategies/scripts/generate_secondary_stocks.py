# Read in our appropriate libraries that we will use here.
import sys
import os
import yfinance as yf
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import load_config
from sklearn.preprocessing import StandardScaler


# core sectors are the divisions that our core stocks below belong to, just going to list them for use later.
CORE_SECTORS = {
    'Technology',
    'Communication Services',
    'Consumer Cyclical',
    'Consumer Staples',
    'Consumer Discretionary',
    'Healthcare',
    'Financials'
}

# Let's acquire our list of S&P 500 stock tickers.
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

# Let's also acquire our Nasdaq 100 stock tickers.
def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    tables = pd.read_html(url)
    df = tables[4]
    return df['Ticker'].tolist()

# Function to rescale features that need to be rescaled (if needed) for preprocessed data.
#def rescale_features(df):
    # These 5 columns are being unscaled in the download process when using preprocessed data.  We need to rescale them.
    #cols_to_rescale = ['Close_sec', 'Volume_sec', 'Open_sec', 'High_sec', 'Low_sec']
        
    # Initialize the scaler
    #scaler = StandardScaler()
    
    #df[cols_to_rescale] = scaler.fit_transform(df[cols_to_rescale])
    
    #print(f"Rescaled columns: {cols_to_rescale}")
        
    #return df

# Now we will now extract our new features from the pandas-ta library for our technical indicators.
def calculate_indicators(df, window_sma = 50, window_ema = 50, window_rsi = 14):
    # Simple Moving Average (SMA)
    df['SMA_sec'] = df['Close_sec'].rolling(window=window_sma).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_sec'] = df['Close_sec'].ewm(span=window_ema, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close_sec'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
    rs = gain / loss
    df['RSI_sec'] = 100 - (100 / (1 + rs))

    
    # Bollinger Bands, the calculation will automatically create the 3 feature columns for us.
    df['BBM_sec'] = df['Close_sec'].rolling(window=window_sma).mean()
    df['BBU_sec'] = df['BBM_sec'] + 2 * df['Close_sec'].rolling(window=window_sma).std()
    df['BBL_sec'] = df['BBM_sec'] - 2 * df['Close_sec'].rolling(window=window_sma).std()

    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close_sec'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close_sec'].ewm(span=26, adjust=False).mean()
    df['MACD_sec'] = ema_12 - ema_26
    df['MACD_Signal_sec'] = df['MACD_sec'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist_sec'] = df['MACD_sec'] - df['MACD_Signal_sec']
    
    # Average Directional Index (ADX)
    high_low = df['High_sec'] - df['Low_sec']
    high_close = abs(df['High_sec'] - df['Close_sec'].shift(1))
    low_close = abs(df['Low_sec'] - df['Close_sec'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    plus_dm = df['High_sec'].diff(1).where(lambda x: x > 0, 0)
    minus_dm = df['Low_sec'].diff(1).where(lambda x: x < 0, 0).abs()

    tr_14 = true_range.rolling(window=14).sum()
    plus_dm_14 = plus_dm.rolling(window=14).sum()
    minus_dm_14 = minus_dm.rolling(window=14).sum()

    plus_di = 100 * (plus_dm_14 / tr_14)
    minus_di = 100 * (minus_dm_14 / tr_14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX_14_sec'] = dx.rolling(window=14).mean()

    
    # Commodity Channel Index (CCI)
    typical_price = (df['High_sec'] + df['Low_sec'] + df['Close_sec']) / 3
    mean_typical_price = typical_price.rolling(window=20).mean()
    mean_deviation = (typical_price - mean_typical_price).abs().rolling(window=20).mean()
    df['CCI_20_sec'] = (typical_price - mean_typical_price) / (0.015 * mean_deviation)

    
    # Average True Range (ATR)
    high_low = df['High_sec'] - df['Low_sec']
    high_close = abs(df['High_sec'] - df['Close_sec'].shift(1))
    low_close = abs(df['Low_sec'] - df['Close_sec'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14_sec'] = true_range.rolling(window=14).mean()

    
    # Stochastic Oscillator (Stoch)
    lowest_low = df['Low_sec'].rolling(window=14).min()
    highest_high = df['High_sec'].rolling(window=14).max()
    df['Stoch_K_sec'] = 100 * ((df['Close_sec'] - lowest_low) / (highest_high - lowest_low))
    df['Stoch_D_sec'] = df['Stoch_K_sec'].rolling(window=3).mean()

    
    # Momentum indicators using different periods
    df['Momentum_1_sec'] = df['Close_sec'] - df['Close_sec'].shift(1)
    df['Momentum_3_sec'] = df['Close_sec'] - df['Close_sec'].shift(3)
    df['Momentum_7_sec'] = df['Close_sec'] - df['Close_sec'].shift(7)
    df['Momentum_30_sec'] = df['Close_sec'] - df['Close_sec'].shift(30)
    df['Momentum_50_sec'] = df['Close_sec'] - df['Close_sec'].shift(50)

    
    # On-Balance Volume (OBV)
    df['OBV_sec'] = (df['Volume_sec'] * ((df['Close_sec'] > df['Close_sec'].shift(1)).astype(int) - (df['Close_sec'] < df['Close_sec'].shift(1)).astype(int))).cumsum()
    
    
    
    return df

# Let's prepare a na value check and fill function to use in our main function for later.
def fill_missing_vals(df):
    df.isna().sum()
    df.ffill(inplace = True)
    df.bfill(inplace = True)
    df.interpolate(method = 'linear', inplace = True)
    
    return df


# Bringing it all together, let's use our previous function calls here along with the data settings in our config
# file to create a new csv dataframe of our secondary stocks to be used later.
def main(config):
    use_preprocessed = config['yfinance']['use_preprocessed'] # Uses already loaded pre-processed data.
    
    # Logic for if the config file has use_preprocessed set to True
    if use_preprocessed:
        print("Using pre-processed data.")
        csv_path = config['yfinance']['csv_paths']['sec_stock_unscaled']
        df_secondary_stocks = pd.read_csv(csv_path, parse_dates = ['Date'], index_col = 'Date')
        print(f"Pre-processed data loaded successfully with shape: {df_secondary_stocks.shape}")
        
        # Exclude core stocks from newly generated stocks
        core_tickers = config['yfinance']['core_tickers']
        df_secondary_stocks = df_secondary_stocks[~df_secondary_stocks['ticker'].isin(core_tickers)]
        
        # Implement our rescaling function here.
        #df_secondary_stocks = rescale_features(df_secondary_stocks)
        
    else:
        print("Fetching raw stock data.")
        max_secondary_stocks = config['yfinance']['max_secondary_stocks']
        start_date = config['yfinance']['start_date']
        end_date = config['yfinance']['end_date']
        core_tickers = config['yfinance']['core_tickers']
    
        sp500_tickers = get_sp500_tickers()
        nasdaq100_tickers = get_nasdaq100_tickers()
        all_tickers = list(set(sp500_tickers + nasdaq100_tickers))
    
        # Filter out core stocks from the new incoming list of secondary stocks if they get generated
        all_tickers = [ticker for ticker in all_tickers if ticker not in core_tickers]
    
        historical_data = {}
    
        successful_tickers = 0
    
        for ticker in all_tickers:
            if successful_tickers >= max_secondary_stocks:
                break
        
            try:
                data = yf.download(ticker, start = start_date, end = end_date)[['Close', 'Volume', 'Open', 'High', 'Low']]
            
                if data.empty or 'Close' not in data.columns:
                    print(f"No data or 'Close' column missing for {ticker}, skipping.")
                    continue
            
                data = data.rename(columns = {
                    'Close' : 'Close_sec',
                    'Volume' : 'Volume_sec',
                    'Open' : 'Open_sec',
                    'High' : 'High_sec',
                    'Low' : 'Low_sec',
                })
            
            
                # An important lines here, we will set the active first row to 03-14.  This offsets the initial rolling window from our data entry point of 01-01.
                data = data[data.index >= '2019-03-14']
            
                data = calculate_indicators(data)
                data = fill_missing_vals(data)
            
                historical_data[ticker] = data
                successful_tickers += 1
            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
        if not historical_data:
            print("No data was successfully downloaded.")
        
            return
    
        df_secondary_stocks = pd.concat(historical_data.values(), keys = historical_data.keys(), names = ['ticker', 'Date'])
        df_secondary_stocks = df_secondary_stocks.reset_index(level=['ticker', 'Date'])
    
    # Use the optimized feature list from the config file
    optimized_features = config['yfinance']['optimized_features']
    print(f"Optimized features: {optimized_features}")
    
    # Calculate the scores for each stock based on optimized features
    stock_scores = df_secondary_stocks.groupby('ticker')[optimized_features].mean()
    print(f"Stock scores:\n{stock_scores.head()}")    
    
    # Sort the stocks by their scores (using a sum of means across these same optimized features)
    stock_scores['total_score'] = stock_scores.sum(axis = 1)
    
    print(f"Selecting top {config['yfinance']['max_secondary_stocks']} stocks.")
    top_stocks = stock_scores.nlargest(config['yfinance']['max_secondary_stocks'], 'total_score').index
    print(f"Top stocks selected:\n{top_stocks}")
    
    # Filter the data for the optimized features, but keep ALL original features intact
    secondary_stocks_gen_filtered = df_secondary_stocks[df_secondary_stocks['ticker'].isin(top_stocks)]
    
    print(secondary_stocks_gen_filtered.head())
    
    # Save to the appropriate path
    secondary_stocks_gen_filtered.to_csv(config['yfinance']['csv_paths']['secondary_stocks_gen_filtered'])
    
    print(f"Final data saved to{config['yfinance']['csv_paths']['secondary_stocks_gen_filtered']}")
    print(f"Number of stocks in the final data: {len(secondary_stocks_gen_filtered['ticker'].unique())}")
    
    
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__),
    '..', 'config', 'config.yaml')
    
    config = load_config(config_path)
    
    main(config)

