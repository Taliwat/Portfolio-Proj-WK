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
                    'ticker' : ticker,
                    'marketCap' : info.get('marketCap', 0),
                    'PE_ratio' : info.get('forwardPE', info.get('trailingPE', None)),
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    return tech_stocks

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
    tech_stocks_sorted = sorted(tech_stocks, key = lambda x: x['marketCap'], reverse = True)
    top_tech_stocks = tech_stocks_sorted[:max_secondary_stocks]
    
    historical_data = {}
    for stock in top_tech_stocks:
        data = yf.download(stock['ticker'], start = start_date, end = end_date)[['Close', 'Volume']]
        
        
        historical_data[stock['ticker']] = data



    df_secondary_stocks = pd.concat(historical_data.values(), keys = historical_data.keys())
    df_secondary_stocks.reset_index(level=1, inplace = True)
    df_secondary_stocks['Date'] = pd.to_datetime(df_secondary_stocks['Date'])
    print(df_secondary_stocks.head())
    
    df_secondary_stocks.to_csv(config['yfinance']['csv_paths']['secondary_tickers_source'])
    
    
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__),
    '..', 'config', 'config.yaml')
    
    config = load_config(config_path)
    
    main(config)

