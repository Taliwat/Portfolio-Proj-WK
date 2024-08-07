# Read in our appropriate libraries that we will use here.
import yfinance as yf
import pandas as pd

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
    df = tables[3]
    return df['Ticker'].tolist()

# Now from these lists let's filter for just the Technology stocks so we can use later.
def tech_stocks_filter(tickers):
    tech_stocks = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if info['sector'] == 'Technology':
                tech_stocks.append((ticker, info.get('marketCap', 0)))
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    return tech_stocks

def main():
    sp500_tickers = get_index_tickers('^GSPC')
    nasdaq100_tickers = get_index_tickers('^NDX')
    
    all_tickers = list(set(sp500_tickers + nasdaq100_tickers))
    
    tech_stocks = tech_stocks_filter(all_tickers)
    
    tech_stocks_sorted = sorted(tech_stocks, key = lambda x: x[1], reverse = True)
    
    top_secondary_tech_stocks = [ticker for ticker, market_cap in tech_stocks_sorted[:100]]