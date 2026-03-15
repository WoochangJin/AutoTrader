import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

from data_loader.etf_data_process import SimpleStockDataset
from agent.model import SimpleStrategyModel

def run_full_comparison():
    # 1. 모델 가중치 로드
    checkpoint_path = os.path.join(root_dir, 'saved/checkpoint_best.pth')
    if not os.path.exists(checkpoint_path):
        print("❌ 모델 파일이 없습니다. train.py를 실행하여 가중치를 먼저 만드세요.")
        return

    ma_windows = [5, 20, 60, 200]
    model = SimpleStrategyModel(stock_feat_dim=len(ma_windows))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2. 전체 데이터셋 로드 (mode='all'로 설정하여 데이터 병합)
    dataset = SimpleStockDataset(
        data_dir=os.path.join(root_dir, 'data'), 
        target_ticker="TQQQ", 
        mode='all', 
        windows=ma_windows
    )
    
    results = []
    # 시점 일치를 위한 변수 (어제의 결정이 오늘의 수익을 만든다)
    prev_ai_action = 0
    prev_ma200_action = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            # dataset[i]의 ret은 '오늘 시가 대비 내일 시가' 혹은 '오늘 종가 대비 내일 종가' 수익률
            stock_feat, ret = dataset[i]
            market_daily_ret = ret.item()
            
            # [수익 반영] 어제 정한 포지션(0 or 1)으로 오늘 수익률을 곱함
            ai_strat_ret = prev_ai_action * market_daily_ret
            ma200_strat_ret = prev_ma200_action * market_daily_ret
            
            # [신호 생성] 오늘 종가(stock_feat)를 보고 내일(Next Day) 포지션을 결정
            ai_signal = model(stock_feat.unsqueeze(0)).item()
            current_ai_action = 1 if ai_signal > 0.5 else 0
            
            current_close = dataset.df['Close'].iloc[i]
            current_ma200 = dataset.df['MA200'].iloc[i]
            current_ma200_action = 1 if current_close > current_ma200 else 0
            
            results.append({
                'date': dataset.dates[i],
                'market_ret': market_daily_ret,
                'ai_strat_ret': ai_strat_ret,
                'ma200_strat_ret': ma200_strat_ret,
                'ai_signal': ai_signal
            })
            
            # 포지션 업데이트 (내일의 prev_action이 됨)
            prev_ai_action = current_ai_action
            prev_ma200_action = current_ma200_action

    # 3. 결과 계산
    df = pd.DataFrame(results).set_index('date')
    df['Market_Cum'] = (1 + df['market_ret']).cumprod()
    df['AI_Cum'] = (1 + df['ai_strat_ret']).cumprod()
    df['MA200_Cum'] = (1 + df['ma200_strat_ret']).cumprod()

    # 4. 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    start_date = df.index[0].date()
    end_date = df.index[-1].date()

    # 상단: 누적 수익률 (로그 스케일)
    ax1.plot(df.index, df['Market_Cum'], label='Market (B&H)', color='lightgray', alpha=0.5)
    ax1.plot(df.index, df['MA200_Cum'], label='Simple MA200', color='orange', linewidth=1.5)
    ax1.plot(df.index, df['AI_Cum'], label='AI Optimized Strategy', color='red', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_title(f"TQQQ Full Period Backtest ({start_date} ~ {end_date})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 하단: 전략 신호 비교
    ax2.fill_between(df.index, 0, 1, where=(df['MA200_Cum'].pct_change() != 0), 
                     color='orange', alpha=0.1, label='MA200 Holding')
    ax2.plot(df.index, df['ai_signal'], color='red', alpha=0.6, label='AI Signal')
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    ax2.set_ylabel("Signal / Holding")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 5. 최종 수익률 출력
    print(f"\n📊 전체 기간 통합 리포트 ({start_date} ~ {end_date})")
    print("-" * 50)
    print(f"🔹 시장 누적 수익률   : {(df['Market_Cum'].iloc[-1]-1)*100:12.2f}%")
    print(f"🔹 MA200 누적 수익률  : {(df['MA200_Cum'].iloc[-1]-1)*100:12.2f}%")
    print(f"🔹 AI 전략 누적 수익률 : {(df['AI_Cum'].iloc[-1]-1)*100:12.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    run_full_comparison()