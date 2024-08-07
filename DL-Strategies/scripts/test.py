import yfinance as yf

ticker = 'AAPL'

data = yf.download(ticker, start = '2023-01-01', end = '2023-12-31')

print(data.head())


def print_sector_info(tickers):
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            print(f"{ticker}: {info.get('sector')}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
print_sector_info(tickers)
