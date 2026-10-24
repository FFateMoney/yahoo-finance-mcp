"""Fetch macro series for the regime study (cloud container cannot reach Yahoo/FRED).
Writes research/regime/data/macro.csv: Date x [^TNX, ^IRX, ^VIX, TLT, IEF] adjusted closes."""
import os, yfinance as yf
T = ["^TNX", "^IRX", "^VIX", "TLT", "IEF"]
df = yf.download(T, start="1999-01-01", auto_adjust=True, progress=False)["Close"]
os.makedirs("research/regime/data", exist_ok=True)
df.round(4).to_csv("research/regime/data/macro.csv")
print(df.tail(), df.notna().sum())
