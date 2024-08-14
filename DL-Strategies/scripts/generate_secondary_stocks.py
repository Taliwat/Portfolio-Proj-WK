# Read in our appropriate libraries that we will use here.
import sys
import os
import yfinance as yf
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import load_config

# core sectors are the divisions that our core stocks below belong to, just going to list them for use later.
CORE_SECTORS = {
    'Technology',
    'Communication Services',
    'Consumer Cyclical'
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
def tech_stocks_filter(tickers, valid_sectors, core_tickers):
    tech_stocks = []
    for ticker in tickers:
        if ticker in core_tickers:
            continue
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', None)
            if sector in valid_sectors:
                tech_stocks.append({
                    'ticker' : ticker
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    return tech_stocks

# Now we will calculate and create our indicators to be used in the generation and filetering of our secondary stocks.  The SMA (Simple Moving Average), EMA (Exponential Moving Average), and the RSI (Relative Strength Index).
def calculate_indicators(df, window_sma = 50, window_ema = 50, window_rsi = 14):
    # Simple Moving Average (SMA)
    df.loc[:, 'SMA'] = df['Close'].rolling(window = window_sma).mean()
    
    # Exponential Moving Average (EMA)
    df.loc[:, 'EMA'] = df['Close'].ewm(span = window_ema, adjust = False).mean()
    
    # Relative Strength Index (RSI)    
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window = window_rsi, min_periods = 1).mean()
    avg_loss = loss.rolling(window = window_rsi, min_periods = 1).mean()
    
    rs = avg_gain / avg_loss
    df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
    
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
    
    tech_stocks = tech_stocks_filter(all_tickers, valid_sectors, core_tickers)    
        
    historical_data = {}
    
    successful_tickers = 0
    
    for stock in tech_stocks:
        if successful_tickers >= max_secondary_stocks:
            break
        
        try:
            data = yf.download(stock['ticker'], start = start_date, end = end_date)[['Close', 'Volume']]
            if data.empty:
                print(f"No dat available for {stock['ticker']}, skipping.")
                continue
            
            if data.index[0].strftime('%Y-%m-%d') > start_date:
                start_date_adjusted = data.index[0].strftime('%Y-%m-%d')
            else:
                start_date_adjusted = start_date
            
            if data.index[-1].strftime('%Y-%m-%d') < end_date:
                end_date_adjusted = data.index[-1].strftime('%Y-%m-%d')
            else:
                end_date_adjusted = end_date
            
            if start_date_adjusted != start_date or end_date_adjusted != end_date:
                data = yf.download(stock['ticker'], start = start_date_adjusted, end = end_date_adjusted)[['Close', 'Volume']]
            
            
            data = data.copy()
            data = calculate_indicators(data)        
            historical_data[stock['ticker']] = data
            successful_tickers += 1
        except Exception as e:
            print(f"Failed to download data for {stock['ticker']}: {e}")
    
    if not historical_data:
        print("No data was successfully downloaded.")
        
    else:
        df_secondary_stocks = pd.concat(historical_data.values(), keys = historical_data.keys(), names = ['ticker', 'Date'])
        df_secondary_stocks = df_secondary_stocks.reset_index(level=['ticker', 'Date'])
        df_secondary_stocks['Date'] = pd.to_datetime(df_secondary_stocks['Date'])
    
        sort_indicators = df_secondary_stocks.groupby('ticker').agg({
            'SMA' : 'last',
            'EMA' : 'last',
            'RSI' : 'last'
        }).reset_index()
    
        final_indicators_sorted = sort_indicators.sort_values(by = ['SMA', 'EMA', 'RSI'], ascending = [False, False, False])
        top_indicators = final_indicators_sorted.head(max_secondary_stocks)
        secondary_stocks_gen = df_secondary_stocks[df_secondary_stocks['ticker'].isin(top_indicators['ticker'])]
    
        secondary_stocks_gen.to_csv(config['yfinance']['csv_paths']['secondary_stocks_gen'])
    
    
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__),
    '..', 'config', 'config.yaml')
    
    config = load_config(config_path)
    
    main(config)

