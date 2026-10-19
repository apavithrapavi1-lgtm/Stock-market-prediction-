import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

s = input("Ticker (TCS.NS): ") or "TCS.NS"
p = yf.download(s, period="1y", progress=False)["Close"].dropna().values.flatten()

X = [p[i-5:i] for i in range(5, len(p))]
m = LinearRegression().fit(X[:-1], p[5:-1])
n = m.predict([p[-5:]])[0]

print("Latest price:", round(p[-1], 2))
print("Predicted price:", round(n, 2))

plt.plot(p[-30:])
plt.axhline(n, linestyle="--")
plt.savefig("stock_prediction.png")
plt.show()