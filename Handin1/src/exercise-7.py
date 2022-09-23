# Exercise 7

import yfinance as yf
import matplotlib.pyplot as plt

def get_returns(ticker):
    stock = yf.Ticker(ticker)

    hist = stock.history(period="1y")
    #days = hist.index.values
    prices = hist["Close"]

    returns = prices.shift(1) / prices - 1

    return (hist, returns)

(hist1, returns1) = get_returns("MSFT")
(hist2, returns2) = get_returns("WEAT")

correlation = hist1["Close"].rolling(7).corr(hist2["Close"])
days = hist1.index.values

plt.figure(1)
plt.scatter(days, returns1)
plt.title("Daily returns of MSFT")
plt.show()

plt.figure(2)
plt.scatter(days, returns2)
plt.title("Daily returns of WEAT")
plt.show()

plt.figure(3)
plt.scatter(days, correlation)
plt.title("Correlation between the returns of MSFT and WEAT")
plt.show()