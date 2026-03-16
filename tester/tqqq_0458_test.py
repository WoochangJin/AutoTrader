import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def run_final_strategy():
    # 1. 데이터 로드 및 이평선 계산
    df = pd.read_csv("data/test/TQQQ_daily.csv", skiprows=3, 
                     names=['Date','Close','High','Low','Open','Volume','MA200','MA60'], 
                     index_col='Date', parse_dates=True).sort_index()
    
    ma_windows = [5, 20, 60, 200]
    weights = [0.19, 0.20, 0.33, 0.28] # 1단계 최적화 가중치
    
    for w in ma_windows:
        df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
    
    # AI_MA(가중합 이평선) 및 기울기(Slope) 계산
    df['AI_MA'] = sum(df[f'MA{w}'] * weights[i] for i, w in enumerate(ma_windows))
    df['Slope'] = df['AI_MA'].pct_change()
    df = df.dropna()

    # --- 전략 파라미터 ---
    LONG_TH = 0.001    # 이평선 기울기가 이 이상이면 '상승장' (풀매수)
    DIP_UNIT = -0.025  # 전일 대비 -2.5% 이상 빠지면 '떨사' (분할추매)
    RIP_GOAL = 0.03    # 내 평단가 대비 +3% 수익 시 '오팔' (탈출)
    BUY_FRACTION = 0.25 # 하락장에서 떨사 한 번에 현금 25%씩 투입
    
    cash = 1000000
    shares = 0
    total_invested = 0
    history = []

    for i in range(1, len(df)):
        curr_close = df.iloc[i]['Close']
        slope = df.iloc[i]['Slope']
        daily_ret = (curr_close / df.iloc[i-1]['Close']) - 1
        
        # 📍 1단계: 이평선 AI 판단 (상승장인가?)
        if slope >= LONG_TH:
            # [상승 추세] 이미 물량이 있다면 홀딩, 없다면 풀매수
            if cash > 0:
                shares += (cash * 0.999) / curr_close
                total_invested += cash
                cash = 0
            status = "TREND_LONG"
        
        # 📍 2단계: 하락장 또는 관망장일 때 '떨사오팔' 가동
        else:
            # 오팔(Rip) 체크: 평단 대비 수익권이면 전량 매도 후 현금화
            if shares > 0:
                avg_price = total_invested / shares
                if (curr_close / avg_price) - 1 >= RIP_GOAL:
                    cash = shares * curr_close * 0.999
                    shares = 0
                    total_invested = 0
                    status = "RIP_SELL"
                else:
                    status = "HOLD_WAIT"
            else:
                status = "CASH_WAIT"

            # 떨사(Dip) 체크: 현금이 있고 폭락(-2.5%↑)하면 분할 매수
            if cash > 0 and daily_ret <= DIP_UNIT:
                buy_amt = cash * BUY_FRACTION
                shares += (buy_amt * 0.999) / curr_close
                total_invested += buy_amt
                cash -= buy_amt
                status = "DIP_BUY"

        total_val = cash + (shares * curr_close)
        history.append({'date': df.index[i], 'total': total_val, 'price': curr_close, 'status': status})

    # 결과 분석 및 그래프
    res_df = pd.DataFrame(history).set_index('date')
    res_df['TQQQ_Hold'] = (res_df['price'] / res_df['price'].iloc[0]) * 1000000
    
    plt.figure(figsize=(14, 7))
    plt.plot(res_df['total'], label='AI Trend + Tteolsaopul', color='red', lw=2)
    plt.plot(res_df['TQQQ_Hold'], label='TQQQ Buy & Hold', color='gray', alpha=0.4)
    plt.yscale('log')
    plt.title("Final Hybrid Strategy: AI MA Trend + Scale-in Dip/Rip")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    print(f"💰 최종 자산: {res_df['total'].iloc[-1]:,.0f}원")
    print(f"📊 전략 수익률: {(res_df['total'].iloc[-1]/1000000 - 1)*100:.2f}%")

if __name__ == "__main__":
    run_final_strategy()