import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def collect_all_finance_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 📍 폴더 구조 세분화
    base_save_dir = os.path.join(root_dir, "data")
    train_dir = os.path.join(base_save_dir, "train")
    test_dir = os.path.join(base_save_dir, "test")
    
    for d in [train_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"📂 '{d}' 폴더가 생성되었습니다.")

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

    print(f"🚀 데이터 수집 및 Train/Test 분리 시작")
    print("-" * 50)

    for ticker, filename in target_dict.items():
        try:
            print(f"📦 {ticker} 수집 중...", end=" ", flush=True)
            df = yf.download(ticker, period="10y", progress=False)
            
            if df.empty:
                print("❌ 실패")
                continue

            # 기초 지표 계산 (이후 모델에서 쓸 feature)
            df['MA200'] = df['Close'].rolling(window=200).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            df = df.dropna() # 앞부분 빈 데이터 제거

            # 📍 8:2 비율로 Train/Test 분리 (시계열 순서 유지)
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            # 각각 저장
            train_df.to_csv(os.path.join(train_dir, f"{filename}.csv"))
            test_df.to_csv(os.path.join(test_dir, f"{filename}.csv"))
            
            print(f"✅ 완료 (Train: {len(train_df)}일, Test: {len(test_df)}일)")

        except Exception as e:
            print(f"❌ 에러: {e}")

    print("-" * 50)
    print("✨ 모든 데이터 분리 저장 완료!")

if __name__ == "__main__":
    collect_all_finance_data()