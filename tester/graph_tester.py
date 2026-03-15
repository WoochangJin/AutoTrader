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

def run_test_case():
    print("🔎 [테스트케이스] 최근 구간 AI vs 리스크관리형 MA200 비교 시작...")
    
    # 1. 모델 및 가중치 설정 (사용자 결과값)
    step1_weights = torch.tensor([0.2331, 0.2524, 0.2572, 0.2573])
    ma_windows = [5, 20, 60, 200]
    
    checkpoint_path = os.path.join(root_dir, 'saved/checkpoint_step2.pth')
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일이 없습니다. (saved/checkpoint_step2.pth)")
        return

    model = FullStrategyModel(stock_feat_dim=len(ma_windows), fixed_weights=step1_weights)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with torch.no_grad():
        sl_val = model.stop_loss_threshold.item()
        tp_val = model.take_profit_threshold.item()

    # 2. 데이터 로드 (최근 구간 'test' 모드)
    dataset = SimpleStockDataset(data_dir=os.path.join(root_dir, 'data'), 
                                 target_ticker="TQQQ", mode='test', windows=ma_windows)
    
    df = dataset.df.copy()
    # 📍 AI 이평선 계산
    df['AI_MA'] = sum(df[f'MA{w}'] * step1_weights[i].item() for i, w in enumerate(ma_windows))
    
    results = []
    ai_prev_action, ai_entry_price = 0, 0.0
    ma200_prev_action, ma200_entry_price = 0, 0.0

    # 3. 최근 구간 시뮬레이션
    with torch.no_grad():
        for i in range(1, len(dataset)):
            curr_feat, ret = dataset[i]
            prev_feat, _ = dataset[i-1]
            current_date = dataset.dates[i]
            current_close = df.loc[current_date, 'Close']
            
            # AI 로직
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

            # MA200 리스크관리형 로직
            ma200_curr_ret = (current_close - ma200_entry_price) / ma200_entry_price if ma200_prev_action == 1 else 0.0
            ma200_signal = 1 if df['Close'].iloc[i-1] > df['MA200'].iloc[i-1] else 0
            ma200_action = ma200_signal
            if ma200_prev_action == 1 and (ma200_curr_ret <= sl_val or ma200_curr_ret >= tp_val):
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
    
    # 상단: 누적 수익률 비교
    ax1.plot(res_df.index, res_df['AI_Cum'], label='AI Optimized (Test)', color='red', lw=2.5)
    ax1.plot(res_df.index, res_df['MA200_Risk_Cum'], label='MA200 + Risk (Test)', color='orange', alpha=0.8, ls='--')
    ax1.plot(res_df.index, res_df['Market_Cum'], label='Market B&H', color='gray', alpha=0.4, ls=':')
    ax1.set_title(f"Test Case Performance ({res_df.index[0].date()} ~ {res_df.index[-1].date()})")
    ax1.legend(); ax1.grid(True, alpha=0.2)

    # 하단: 가격 및 지표 상세
    plot_df = df.loc[res_df.index]
    ax2.plot(plot_df.index, plot_df['Close'], color='black', alpha=0.2, label='TQQQ Price', zorder=1)
    
    # 📍 AI MA 실선 표시
    ax2.plot(plot_df.index, plot_df['AI_MA'], color='red', lw=1.5, label='AI Optimized MA', zorder=3)
    # 📍 MA200 점선 표시
    ax2.plot(plot_df.index, plot_df['MA200'], color='orange', alpha=0.5, ls='--', label='200D MA', zorder=2)
    
    # 배경 면적 (Holding Zone)
    ax2.fill_between(res_df.index, 0, 1, where=(res_df['ai_action'] == 1), 
                     color='red', alpha=0.08, label='AI Holding Zone', 
                     transform=ax2.get_xaxis_transform(), zorder=0)

    # 마커 표시
    markers = [('BUY', '^', 'green', 130, 'BUY'), ('STOP_LOSS', 'X', 'darkred', 130, 'SL'), 
               ('TAKE_PROFIT', '*', 'blue', 200, 'TP'), ('SLOPE_OUT', 'v', 'red', 130, 'Slope Exit')]
    for reason, m, c, s, l in markers:
        pts = res_df[res_df['ai_reason'] == reason]
        ax2.scatter(pts.index, pts['price'], marker=m, color=c, s=s, label=l, zorder=5)

    ax2.set_title("Test Case Trade Execution Details")
    ax2.set_ylim(plot_df['Close'].min() * 0.95, plot_df['Close'].max() * 1.05)
    ax2.legend(loc='upper left', ncol=3); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"\n📊 최근 테스트케이스 결과 ({len(res_df)} 거래일)")
    print(f"🤖 AI 수익률     : {res_df['AI_Cum'].iloc[-1]:.2f}배")
    print(f"🟠 MA200+리스크 : {res_df['MA200_Risk_Cum'].iloc[-1]:.2f}배")
    print(f"📉 시장 존버    : {res_df['Market_Cum'].iloc[-1]:.2f}배")
    print(f"시장 대비 수익률 : {(res_df['AI_Cum'].iloc[-1] - res_df['Market_Cum'].iloc[-1]) * 100:.2f}%")

if __name__ == "__main__":
    run_test_case()