import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

from data_loader.etf_data_process import SimpleStockDataset
from train import TQQQPredictor

def calculate_metrics(history):
    history_series = pd.Series(history)
    final_return = (history[-1] / history[0]) - 1
    days = len(history)
    cagr = (1 + final_return) ** (252 / days) - 1
    peak = history_series.cummax()
    drawdown = (history_series - peak) / peak
    mdd = drawdown.min()
    return final_return, cagr, mdd

def run_backtest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 📍 2021년부터 2025년까지 데이터 로드
    dataset = SimpleStockDataset(
        data_dir=os.path.join(root_dir, "data"), 
        target_ticker='TQQQ',
        start_date='2021-01-01',
        end_date='2025-12-31'
    )

    model = TQQQPredictor(input_dim=8).to(device)
    model.load_state_dict(torch.load(os.path.join(root_dir, "saved/best_tqqq_model.pth"), map_location=device))
    model.eval()

    # 초기 세팅
    cash = 1000000
    shares = 0
    total_invested = 0
    
    # 최적 파라미터 (0458 버전)
    DIP_BUY_UNIT = 0.2
    PROFIT_TARGET = 0.05
    AI_THRESHOLD = -0.005

    history = []
    
    print(f"��백테스트 시작: {dataset.df.index[0].date()} ~" {dataset.df.index[-1].date()})

    for i in range(len(dataset)):
        features, _ = dataset[i]
        curr_price = dataset.df.iloc[i]['Close']
        daily_ret = dataset.df.iloc[i]['pct_change']
        
        with torch.no_grad():
            pred = model(features.to(device).unsqueeze(0)).item()

        # 1. 오팔 (익절)
        if shares > 0:
            avg_price = total_invested / shares
            if (curr_price / avg_price) - 1 >= PROFIT_TARGET:
                cash += shares * curr_price * 0.999
                shares = 0
                total_invested = 0

        # 2. 떨사 (매수)
        if cash > 1000 and pred > AI_THRESHOLD:
            if daily_ret < -0.02 or pred > 0.01:
                buy_amount = cash * DIP_BUY_UNIT
                shares += (buy_amount * 0.999) / curr_price
                total_invested += buy_amount
                cash -= buy_amount

        history.append(cash + (shares * curr_price))

    # 결과 지표 계산
    f_ret, cagr, mdd = calculate_metrics(history)
    hold_history = (1 + dataset.df['Close'].pct_change().fillna(0)).cumprod() * 1000000
    h_ret, h_cagr, h_mdd = calculate_metrics(hold_history.values)

    # 출력 및 시각화
    print(f"\n[결과 리포트]")
    print(f"AI 전략  - CAGR: {cagr*100:.2f}%, MDD: {mdd*100:.2f}%")
    print(f"단순보유 - CAGR: {h_cagr*100:.2f}%, MDD: {h_mdd*100:.2f}%")

    plt.figure(figsize=(14, 7))
    plt.plot(dataset.df.index, hold_history.values, label='TQQQ Buy & Hold', color='gray', alpha=0.5)
    plt.plot(dataset.df.index, history, label='AI Hybrid Strategy (21-25)', color='red', lw=2)
    plt.title(f"TQQQ AI Strategy Backtest (2021-2025)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_backtest()