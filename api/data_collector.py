import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def collect_all_finance_data():
    # 1. 경로 설정: 현재 파일(api/data_collector.py)의 부모 폴더(루트)에 data 폴더 지정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(root_dir, "data") # 프로젝트 루트/data
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 '{save_dir}' 폴더가 생성되었습니다.")

    # 2. 수집 대상 정의
    target_dict = {
        "SOXL": "SOXL_daily",
        "TQQQ": "TQQQ_daily",
        "UPRO": "UPRO_daily",
        "QQQ": "QQQ_daily",
        "SPY": "SPY_daily",
        "^IXIC": "NASDAQ_daily",
        "KRW=X": "USD_KRW_daily",
        "^TNX": "US10Y_Yield_daily",
        "^VIX": "VIX_daily",
        "DX-Y.NYB": "DXY_daily"
    }

    print(f"🚀 총 {len(target_dict)}개 데이터 수집 시작 (루트/data 폴더 저장)")
    print("-" * 50)

    for ticker, filename in target_dict.items():
        try:
            print(f"📦 {ticker} 수집 중...", end=" ", flush=True)
            df = yf.download(ticker, period="10y", progress=False)
            
            if df.empty:
                print("❌ 실패")
                continue

            df['MA200'] = df['Close'].rolling(window=200).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            
            # 저장 경로 적용
            file_path = os.path.join(save_dir, f"{filename}.csv")
            df.to_csv(file_path)
            print(f"✅ 저장 완료")

        except Exception as e:
            print(f"❌ 에러: {e}")

    print("-" * 50)
    print("✨ 모든 데이터 수집 및 'data' 폴더 저장 완료!")

if __name__ == "__main__":
    collect_all_finance_data()