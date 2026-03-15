import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def collect_and_split_data():
    # 1. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    base_data_dir = os.path.join(root_dir, "data")
    
    # train, test 하위 폴더 생성
    train_dir = os.path.join(base_data_dir, "train")
    test_dir = os.path.join(base_data_dir, "test")
    
    for d in [train_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"📂 폴더 생성 완료: {d}")

    # 2. 수집 대상 정의
    target_dict = {
        # 매매 종목 (Direct Assets)
        "SOXL": "SOXL_daily",
        "TQQQ": "TQQQ_daily",
        "UPRO": "UPRO_daily",
        "SQQQ": "SQQQ_daily",
        
        # 기준 지수 (Direct Benchmarks)
        "QQQ": "QQQ_daily",
        "SPY": "SPY_daily",
        "^IXIC": "NASDAQ_daily",
        
        # 화폐 및 금리 지표 (Direct Macro)
        "KRW=X": "USD_KRW_daily",
        "DX-Y.NYB": "DXY_daily",
        "^TNX": "US10Y_Yield_daily"
    }

    print(f"🚀 10년치 데이터 수집 및 분할 저장 시작...")
    print("-" * 50)

    for ticker, filename in target_dict.items():
        try:
            print(f"📦 {ticker} 수집 중...", end=" ", flush=True)
            # 10년치 데이터 다운로드
            df = yf.download(ticker, period="10y", progress=False)
            
            if df.empty:
                print("❌ 실패 (데이터 없음)")
                continue

            # 기본 지표 계산 (이평선 등)
            df['MA200'] = df['Close'].rolling(window=200).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            
            # 3. 데이터 분할 (Time-series Split: 80% Train, 20% Test)
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            # 4. 각각 저장
            train_df.to_csv(os.path.join(train_dir, f"{filename}.csv"))
            test_df.to_csv(os.path.join(test_dir, f"{filename}.csv"))
            
            print(f"✅ 저장 완료 (Train: {len(train_df)}행, Test: {len(test_df)}행)")

        except Exception as e:
            print(f"❌ 에러: {e}")

    print("-" * 50)
    print("✨ 데이터 분할 수집 작업이 모두 완료되었습니다!")

if __name__ == "__main__":
    collect_and_split_data()