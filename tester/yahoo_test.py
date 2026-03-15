import pandas as pd
import yfinance as yf

df = yf.download("SOXL", period="1y")

# 모든 행(row)을 다 보여주도록 설정
pd.set_option('display.max_rows', None)

print(df)