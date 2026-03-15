import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def run_honest_ma_test(ticker="TQQQ"):
    # 1. 경로 설정 (train 폴더 데이터가 더 길어서 분석하기 좋습니다)
    csv_path = f"data/train/{ticker}_daily.csv"
    if not os.path.exists(csv_path):
        csv_path = f"data/test/{ticker}_daily.csv"
        
    if not os.path.exists(csv_path):
        print("❌ 데이터를 찾을 수 없습니다.")
        return

    # 2. 데이터 로드
    df = pd.read_csv(csv_path, index_col=0)
    
    # [수정 포인트] 인덱스를 강제로 Datetime으로 변환하고 정렬
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.dropna(subset=['Close']) # Close가 없는 행 제거
    df = df[df.index.notnull()]      # 날짜 변환 실패한 행 제거
    df = df.sort_index()

    # Close 컬럼 숫자 변환
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    # 3. 200일 이평선 계산
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 4. 전략 실행 구간 (MA200 데이터가 있는 200번째 행부터 시작)
    test_df = df.iloc[200:].copy() 
    
    if len(test_df) < 1:
        print(f"⚠️ 데이터 부족: 현재 유효 데이터 {len(df)}개 (200개 이상 필요)")
        return

    # 5. 매매 로직
    test_df['Signal'] = (test_df['Close'] > test_df['MA200']).astype(int)
    test_df['Market_Return'] = test_df['Close'].pct_change()
    test_df['Strategy_Return'] = test_df['Signal'].shift(1) * test_df['Market_Return']

    # 6. 누적 수익률
    test_df['Market_Cum'] = (1 + test_df['Market_Return'].fillna(0)).cumprod()
    test_df['Strategy_Cum'] = (1 + test_df['Strategy_Return'].fillna(0)).cumprod()

    # 7. 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # 상단: 가격과 이평선
    ax1.plot(test_df.index, test_df['Close'], label='Price', color='black', alpha=0.4)
    ax1.plot(test_df.index, test_df['MA200'], label='MA200', color='red', linewidth=2)
    
    # 날짜 출력을 위해 .date() 사용 (이제 인덱스가 Datetime이라 작동함)
    start_date = test_df.index[0].date()
    end_date = test_df.index[-1].date()
    ax1.set_title(f"[{ticker}] MA200 Strategy Analysis ({start_date} ~ {end_date})")
    ax1.legend()
    ax1.grid(True)

    # 하단: 수익률
    ax2.plot(test_df.index, test_df['Market_Cum'], label='Market (Buy & Hold)', color='gray', linestyle='--')
    ax2.plot(test_df.index, test_df['Strategy_Cum'], label='AI Strategy (MA200)', color='blue', linewidth=2)
    ax2.axhline(1, color='red', linestyle='-', alpha=0.3)
    ax2.legend()
    ax2.grid(True)

# ... (중략: 시각화 코드 이후) ...
    
    plt.tight_layout()
    plt.show()

    # --- 수익률 상세 출력 ---
    final_market_cum = test_df['Market_Cum'].iloc[-1]
    final_strat_cum = test_df['Strategy_Cum'].iloc[-1]
    
    market_pct = (final_market_cum - 1) * 100
    strat_pct = (final_strat_cum - 1) * 100
    outperformance = strat_pct - market_pct

    print("\n" + "="*50)
    print(f"📊 백테스팅 상세 결과 ({start_date} ~ {end_date})")
    print("-"*50)
    print(f"🔹 시장(Buy & Hold) 누적 수익률 : {market_pct:>10.2f}%")
    print(f"🔹 AI(MA200) 전략 누적 수익률  : {strat_pct:>10.2f}%")
    print("-"*50)
    print(f"💡 시장 대비 초과 수익률        : {outperformance:>10.2f}%p")
    
    # 승률 계산 (보유한 날 중 수익이 난 날의 비율)
    trade_days = test_df[test_df['Signal'] == 1]
    win_days = trade_days[trade_days['Market_Return'] > 0]
    win_rate = (len(win_days) / len(trade_days) * 100) if len(trade_days) > 0 else 0
    
    print(f"🔹 전략 가동 중 승률            : {win_rate:>10.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_honest_ma_test("TQQQ")