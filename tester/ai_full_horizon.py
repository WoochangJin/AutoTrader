import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

from data_loader.etf_data_process import SimpleStockDataset
from agent.model import FullStrategyModel

def run_full_period_test():
    print("🔎 [전체 기간] AI 모델 vs 리스크관리형 MA200 비교 테스트 시작...")
    
    # 1. 모델 설정 (사용자가 확인한 1단계 가중치)
    step1_weights = torch.tensor([0.2331, 0.2524, 0.2572, 0.2573])
    ma_windows = [5, 20, 60, 200]
    
    checkpoint_path = os.path.join(root_dir, 'saved/checkpoint_step2.pth')
    if not os.path.exists(checkpoint_path):
        print(f"❌ 모델 파일이 없습니다: {checkpoint_path}")
        return

    model = FullStrategyModel(stock_feat_dim=len(ma_windows), fixed_weights=step1_weights)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with torch.no_grad():
        sl_val = model.stop_loss_threshold.item()
        tp_val = model.take_profit_threshold.item()

    # 2. 데이터 로드 (mode='all'로 전체 기간 데이터 확보)
    dataset = SimpleStockDataset(data_dir=os.path.join(root_dir, 'data'), 
                                 target_ticker="TQQQ", mode='all', windows=ma_windows)
    df = dataset.df.copy()
    
    results = []
    # 상태 변수 초기화
    ai_prev_action, ai_entry_price = 0, 0.0
    ma200_prev_action, ma200_entry_price = 0, 0.0

    # 3. 백테스트 루프
    with torch.no_grad():
        for i in range(1, len(dataset)):
            curr_feat, ret = dataset[i]
            prev_feat, _ = dataset[i-1]
            current_date = dataset.dates[i]
            current_close = df.loc[current_date, 'Close']
            
            # --- [AI 전략] ---
            ai_curr_ret = (current_close - ai_entry_price) / ai_entry_price if ai_prev_action == 1 else 0.0
            ai_signal = model(curr_feat.unsqueeze(0), prev_feat.unsqueeze(0), 
                              torch.tensor([ai_curr_ret])).item()
            ai_action = 1 if ai_signal > 0.5 else 0
            
            ai_reason = None
            if ai_prev_action == 1 and ai_action == 0:
                if ai_curr_ret <= sl_val: ai_reason = 'STOP_LOSS'
                elif ai_curr_ret >= tp_val: ai_reason = 'TAKE_PROFIT'
                else: ai_reason = 'SLOPE_OUT'
            elif ai_prev_action == 0 and ai_action == 1:
                ai_entry_price = current_close
                ai_reason = 'BUY'

            # --- [MA200 리스크 관리형] ---
            ma200_curr_ret = (current_close - ma200_entry_price) / ma200_entry_price if ma200_prev_action == 1 else 0.0
            ma200_trend_signal = 1 if df['Close'].iloc[i-1] > df['MA200'].iloc[i-1] else 0
            
            ma200_action = ma200_trend_signal
            if ma200_prev_action == 1:
                if ma200_curr_ret <= sl_val or ma200_curr_ret >= tp_val:
                    ma200_action = 0
            
            if ma200_prev_action == 0 and ma200_action == 1:
                ma200_entry_price = current_close

            results.append({
                'date': current_date,
                'market_ret': ret.item(),
                'ai_strat_ret': ai_prev_action * ret.item(),
                'ma200_risk_ret': ma200_prev_action * ret.item(),
                'ai_action': ai_action,
                'ai_reason': ai_reason,
                'price': current_close
            })
            ai_prev_action, ma200_prev_action = ai_action, ma200_action

    res_df = pd.DataFrame(results).set_index('date')
    res_df['AI_Cum'] = (1 + res_df['ai_strat_ret']).cumprod()
    res_df['MA200_Risk_Cum'] = (1 + res_df['ma200_risk_ret']).cumprod()
    res_df['Market_Cum'] = (1 + res_df['market_ret']).cumprod()

    # 4. 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # 상단: 누적 수익률 (로그 스케일 적용 권장 - TQQQ 변동성 때문)
    ax1.plot(res_df.index, res_df['AI_Cum'], label='AI Optimized', color='red', lw=2)
    ax1.plot(res_df.index, res_df['MA200_Risk_Cum'], label='MA200+Risk', color='orange', alpha=0.7, ls='--')
    ax1.plot(res_df.index, res_df['Market_Cum'], label='Market (TQQQ)', color='gray', alpha=0.3, ls=':')
    ax1.set_yscale('log') # 전체 기간은 수익률 차이가 커서 로그 스케일이 보기 편합니다.
    ax1.set_title("Full Period Cumulative Return (Log Scale)")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.1)

    # 하단: 전체 가격 흐름 및 AI 보유 구간
    ax2.plot(df.index, df['Close'], color='black', alpha=0.1, label='TQQQ Price')
    ax2.fill_between(res_df.index, df.loc[res_df.index, 'Close'].min(), df.loc[res_df.index, 'Close'].max(), 
                     where=(res_df['ai_action'] == 1), color='red', alpha=0.05, label='AI Holding')
    
    ax2.set_title("Price Action & AI Holding Zones")
    ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"\n📅 분석 기간: {res_df.index[0].date()} ~ {res_df.index[-1].date()}")
    print(f"🤖 AI 전략     : {res_df['AI_Cum'].iloc[-1]:.2f}배")
    print(f"🟠 MA200+리스크 : {res_df['MA200_Risk_Cum'].iloc[-1]:.2f}배")
    print(f"📉 시장 존버    : {res_df['Market_Cum'].iloc[-1]:.2f}배")

if __name__ == "__main__":
    run_full_period_test()