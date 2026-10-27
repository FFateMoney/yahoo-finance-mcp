"""Quick quote snapshot via Actions (cloud blocks Yahoo). Writes research/regime/data/quote_*.csv."""
import yfinance as yf, os
T = ["P", "SNDK", "MU", "WDC", "STX", "SPY", "ONB"]
os.makedirs("research/regime/data", exist_ok=True)
yf.download(T, period="6mo", auto_adjust=True, progress=False).round(4).to_csv("research/regime/data/quote_daily.csv")
yf.download(T, period="5d", interval="5m", prepost=True, auto_adjust=True, progress=False).round(4).to_csv("research/regime/data/quote_5m.csv")
print("ok")
