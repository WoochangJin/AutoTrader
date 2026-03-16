import yfinance as yf
import pandas as pd
import os
import pandas_ta as ta

def collect_advanced_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(root_dir, "data")
    
    for d in ["train", "test"]:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)

    main_tickers = ["TQQQ", "SOXL"]
    macro_tickers = {"^VIX": "VIX", "^TNX": "US10Y"}

    print("🚀 고도화 데이터 수집 시작...")

    macro_dfs = {}
    for ticker, name in macro_tickers.items():
        print(f"📊 {name} 수집 중...")
        m_df = yf.download(ticker, period="10y", progress=False)
        # 📍 Multi-index 방지 및 Close만 추출
        if isinstance(m_df.columns, pd.MultiIndex):
            m_df.columns = m_df.columns.get_level_values(0)
        macro_dfs[name] = m_df['Close']

    for ticker in main_tickers:
        try:
            print(f"📦 {ticker} 및 기술적 지표 계산 중...", end=" ", flush=True)
            df = yf.download(ticker, period="10y", progress=False)

            # 📍 [핵심 수정] Multi-index 컬럼을 단일 레벨로 평탄화
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 데이터가 비어있는지 확인
            if df.empty:
                print("❌ 데이터 없음")
                continue

            # 📍 기술적 지표 계산 (ta 활용 전 데이터 타입 명시)
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            df['Vol_ROC'] = df['Volume'].pct_change() * 100

            # 매크로 지표 병합
            for name, m_series in macro_dfs.items():
                df = df.join(m_series.rename(name), how='left')

            df = df.dropna()
            
            # Train/Test 분리 (8:2)
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            train_df.to_csv(os.path.join(save_dir, "train", f"{ticker}_daily.csv"))
            test_df.to_csv(os.path.join(save_dir, "test", f"{ticker}_daily.csv"))
            print(f"✅ 완료")

        except Exception as e:
            print(f"❌ 에러 ({ticker}): {e}")

    print("-" * 50)
    print("✨ 데이터 정제 완료!")

if __name__ == "__main__":
    collect_advanced_data()