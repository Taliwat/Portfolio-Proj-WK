# Read in our appropriate libraries that we will use here.
import sys
import os
import yfinance as yf
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import load_config
from statsmodels.tsa.stattools import coint


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

# Now from these lists let's filter for just the Technology stocks so we can use later.
def sec_stocks_filter(tickers, valid_sectors, core_tickers):
    sec_stocks = []
    for ticker in tickers:
        if ticker in core_tickers:
            continue
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', None)
            if sector in valid_sectors:
                sec_stocks.append({
                    'ticker' : ticker
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    return sec_stocks

# Now we will now extract our new features from the pandas-ta library for our technical indicators.
def calculate_indicators(df, window_sma = 50, window_ema = 50, window_rsi = 14):
    # Simple Moving Average (SMA)
    df['SMA_sec'] = df['Close'].rolling(window=window_sma).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_sec'] = df['Close'].ewm(span=window_ema, adjust=False).mean()
    
    # Relative Moving Average (RMA)
    df['RMA_sec'] = df['Close'] / df['EMA_sec']
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
    rs = gain / loss
    df['RSI_sec'] = 100 - (100 / (1 + rs))

    
    # Bollinger Bands, the calculation will automatically create the 3 feature columns for us.
    df['BBM_sec'] = df['Close'].rolling(window=window_sma).mean()
    df['BBU_sec'] = df['BBM_sec'] + 2 * df['Close'].rolling(window=window_sma).std()
    df['BBL_sec'] = df['BBM_sec'] - 2 * df['Close'].rolling(window=window_sma).std()

    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_sec'] = ema_12 - ema_26
    df['MACD_Signal_sec'] = df['MACD_sec'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist_sec'] = df['MACD_sec'] - df['MACD_Signal_sec']
    
    # Average Directional Index (ADX)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    plus_dm = df['High'].diff(1).where(lambda x: x > 0, 0)
    minus_dm = df['Low'].diff(1).where(lambda x: x < 0, 0).abs()

    tr_14 = true_range.rolling(window=14).sum()
    plus_dm_14 = plus_dm.rolling(window=14).sum()
    minus_dm_14 = minus_dm.rolling(window=14).sum()

    plus_di = 100 * (plus_dm_14 / tr_14)
    minus_di = 100 * (minus_dm_14 / tr_14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX_14_sec'] = dx.rolling(window=14).mean()

    
    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    mean_typical_price = typical_price.rolling(window=20).mean()
    mean_deviation = (typical_price - mean_typical_price).abs().rolling(window=20).mean()
    df['CCI_20_sec'] = (typical_price - mean_typical_price) / (0.015 * mean_deviation)

    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14_sec'] = true_range.rolling(window=14).mean()

    
    # Stochastic Oscillator (Stoch)
    lowest_low = df['Low'].rolling(window=14).min()
    highest_high = df['High'].rolling(window=14).max()
    df['Stoch_K_sec'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    df['Stoch_D_sec'] = df['Stoch_K_sec'].rolling(window=3).mean()

    
    # Momentum indicators using different periods
    df['Momentum_1_sec'] = df['Close'] - df['Close'].shift(1)
    df['Momentum_3_sec'] = df['Close'] - df['Close'].shift(3)
    df['Momentum_7_sec'] = df['Close'] - df['Close'].shift(7)
    df['Momentum_30_sec'] = df['Close'] - df['Close'].shift(30)
    df['Momentum_50_sec'] = df['Close'] - df['Close'].shift(50)

    
    # On-Balance Volume (OBV)
    df['OBV_sec'] = (df['Volume'] * ((df['Close'] > df['Close'].shift(1)).astype(int) - (df['Close'] < df['Close'].shift(1)).astype(int))).cumsum()
    
    
    
    return df

# Let's prepare a na value check and fill function to use in our main function for later.
def fill_missing_vals(df):
    df.isna().sum()
    df.ffill(inplace = True)
    df.bfill(inplace = True)
    df.interpolate(method = 'linear', inplace = True)
    
    return df

# Bringing it all together, let's use our previous function calls here along with the data settings in our config
# file to create a new csv dataframe of our secondary technology stocks to be used later.
def main(config):
    max_secondary_stocks = config['yfinance']['max_secondary_stocks']
    start_date = config['yfinance']['start_date']
    end_date = config['yfinance']['end_date']
    core_tickers = config['yfinance']['core_tickers']
    
    
    valid_sectors = CORE_SECTORS
    
    sp500_tickers = get_sp500_tickers()
    nasdaq100_tickers = get_nasdaq100_tickers()
    all_tickers = list(set(sp500_tickers + nasdaq100_tickers))
    
    sec_stocks = sec_stocks_filter(all_tickers, valid_sectors, core_tickers)    
        
    historical_data = {}
    
    successful_tickers = 0
    
    for stock in sec_stocks:
        if successful_tickers >= max_secondary_stocks:
            break
        
        try:
            data = yf.download(stock['ticker'], start = start_date, end = end_date)[['Close', 'Volume', 'Open', 'High', 'Low']]
            data = data.rename(columns = {
                'Close' : 'Close_sec',
                'Volume' : 'Volume_sec',
                'Open' : 'Open_sec',
                'High' : 'High_sec',
                'Low' : 'Low_sec',
            })
            
            print(f"Downloaded data for {stock['ticker']}: \n{data.head()}")
            
            if data.empty:
                print(f"No data available for {stock['ticker']}, skipping.")
                continue
            
            data = calculate_indicators(data)
            data = fill_missing_vals(data)
            
            data['Composite_Score'] = (
                0.4 * data['EMA_sec'] +
                0.3 * data['SMA_sec'] +
                0.2 * data['Close'] +
                0.05 * data['RSI_sec'] +
                0.05 * data['Volume']
            )
            print(f"Composite Score calculate for {stock['ticker']}:\n{data[['Composite_Score']].head()}")
                        
            historical_data[stock['ticker']] = data
            successful_tickers += 1
            
        except Exception as e:
            print(f"Error processing {stock['ticker']}: {e}")
    
    if not historical_data:
        print("No data was successfully downloaded.")
        
    else:
        df_secondary_stocks = pd.concat(historical_data.values(), keys = historical_data.keys(), names = ['ticker', 'Date'])
        df_secondary_stocks = df_secondary_stocks.reset_index(level=['ticker', 'Date'])
        df_secondary_stocks['Date'] = pd.to_datetime(df_secondary_stocks['Date'])
    
        sort_indicators = df_secondary_stocks.groupby('ticker').agg({
            'Composite_Score' : 'last'
        }).reset_index()
    
        final_indicators_sorted = sort_indicators.sort_values(by = 'Composite_Score', ascending = False)
        top_indicators = final_indicators_sorted.head(max_secondary_stocks)
        
        secondary_stocks_gen = df_secondary_stocks[df_secondary_stocks['ticker'].isin(top_indicators['ticker'])]
        
        print(secondary_stocks_gen.isna().sum())
        
        secondary_stocks_gen.reset_index(drop = True, inplace = True)
        secondary_stocks_gen.set_index('Date', inplace = True)
        
        print(secondary_stocks_gen.head())
        
        secondary_stocks_gen.to_csv(config['yfinance']['csv_paths']['secondary_stocks_gen'])
        
    
    
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__),
    '..', 'config', 'config.yaml')
    
    config = load_config(config_path)
    
    main(config)

