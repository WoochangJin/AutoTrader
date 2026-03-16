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

def run_real_sqqq_test():
    print("🔎 [실제 SQQQ 데이터 반영] TQQQ/SQQQ 스위칭 백테스트 시작...")
    
    # 1. 설정 및 모델 로드
    # 📍 사용자의 1단계 최적화 가중치
    step1_weights = torch.tensor([0.2331, 0.2524, 0.2572, 0.2573])
    ma_windows = [5, 20, 60, 200]
    fee_rate = 0.001 
    
    # 📍 중립 구간(Dead Zone) 설정: 잦은 스위칭 방지
    LONG_THRESHOLD = 0.60
    SHORT_THRESHOLD = 0.40
    
    checkpoint_path = os.path.join(root_dir, 'saved/checkpoint_step2.pth')
    if not os.path.exists(checkpoint_path):
        print("❌ 체크포인트 파일이 없습니다. train_step2.py를 먼저 실행하세요.")
        return

    model = FullStrategyModel(stock_feat_dim=len(ma_windows), fixed_weights=step1_weights)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2. 데이터 로드 (TQQQ는 데이터셋으로, SQQQ는 단순 DF로 로드)
    dataset = SimpleStockDataset(data_dir=os.path.join(root_dir, 'data'), 
                                 target_ticker="TQQQ", mode='test', windows=ma_windows)
    
    # 📍 SQQQ 실제 가격 데이터 로드
    sqqq_path = os.path.join(root_dir, 'data/SQQQ.csv')
    sqqq_df = pd.read_csv(sqqq_path, index_col='Date', parse_dates=True)
    
    df = dataset.df.copy()
    # 시각화용 AI MA 계산
    df['AI_MA'] = sum(df[f'MA{w}'] * step1_weights[i].item() for i, w in enumerate(ma_windows))
    
    results = []
    prev_pos = 0 # 1: TQQQ, -1: SQQQ, 0: Cash
    
    # 3. 시뮬레이션
    with torch.no_grad():
        for i in range(1, len(dataset)):
            curr_feat, tqqq_ret = dataset[i]
            prev_feat, _ = dataset[i-1]
            date = dataset.dates[i]
            prev_date = dataset.dates[i-1]
            
            # 실제 SQQQ 일간 수익률 계산
            try:
                sqqq_close_curr = sqqq_df.loc[date, 'Close']
                sqqq_close_prev = sqqq_df.loc[prev_date, 'Close']
                actual_sqqq_ret = (sqqq_close_curr - sqqq_close_prev) / sqqq_close_prev
            except KeyError:
                # 데이터가 비어있을 경우에 대한 예외 처리
                actual_sqqq_ret = -tqqq_ret.item() 

            # AI 신호 (TQQQ 데이터 기반)
            ai_val = model(curr_feat.unsqueeze(0), prev_feat.unsqueeze(0), torch.tensor([0.0])).item()
            
            # 포지션 결정 (Long / Short / Cash)
            if ai_val >= LONG_THRESHOLD:
                curr_pos = 1
            elif ai_val <= SHORT_THRESHOLD:
                curr_pos = -1
            else:
                curr_pos = 0
            
            # 수익률 계산 (어제 정해진 포지션으로 오늘 수익을 누적)
            daily_ret = 0.0
            if prev_pos == 1:
                daily_ret = tqqq_ret.item()
            elif prev_pos == -1:
                daily_ret = actual_sqqq_ret
            
            # 포지션 스위칭 시 수수료 발생
            if prev_pos != curr_pos:
                # 포지션이 정반대로 바뀔 때(T<->S)는 왕복, 그 외(현금화 등)는 편도
                cost = fee_rate * 2 if (prev_pos * curr_pos == -1) else fee_rate
                daily_ret -= cost

            results.append({
                'date': date,
                'tqqq_ret': tqqq_ret.item(),
                'sqqq_ret': actual_sqqq_ret,
                'strat_ret': daily_ret,
                'pos': curr_pos,
                'ai_val': ai_val,
                'price': df.loc[date, 'Close']
            })
            prev_pos = curr_pos

    res_df = pd.DataFrame(results).set_index('date')
    res_df['AI_Cum'] = (1 + res_df['strat_ret']).cumprod()
    res_df['TQQQ_Cum'] = (1 + res_df['tqqq_ret']).cumprod()

    # 4. 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # 상단: 실제 SQQQ 데이터 반영 누적 수익률
    ax1.plot(res_df.index, res_df['AI_Cum'], label='AI Switching (Real TQQQ/SQQQ)', color='indigo', lw=2.5)
    ax1.plot(res_df.index, res_df['TQQQ_Cum'], label='TQQQ Buy & Hold', color='gray', alpha=0.4, ls=':')
    ax1.set_title("Performance with Real Market Data (TQQQ & SQQQ)")
    ax1.legend(); ax1.grid(True, alpha=0.2)

    # 하단: 가격 및 포지션 면적
    ax2.plot(res_df.index, res_df['price'], color='black', alpha=0.2, label='TQQQ Price')
    ax2.plot(res_df.index, df.loc[res_df.index, 'AI_MA'], color='red', alpha=0.5, lw=1, label='AI MA')
    
    # 면적 표시 (빨강: TQQQ, 파랑: SQQQ)
    ax2.fill_between(res_df.index, 0, 1, where=(res_df['pos'] == 1), color='red', alpha=0.1, label='Long (TQQQ)', transform=ax2.get_xaxis_transform())
    ax2.fill_between(res_df.index, 0, 1, where=(res_df['pos'] == -1), color='blue', alpha=0.1, label='Short (SQQQ)', transform=ax2.get_xaxis_transform())
    
    ax2.set_title("Trading Zones: Long(Red) / Short(Blue) / Cash(White)")
    ax2.set_ylim(res_df['price'].min() * 0.9, res_df['price'].max() * 1.1)
    ax2.legend(loc='upper left', ncol=4); ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"\n✅ 실데이터 백테스트 완료")
    print(f"🤖 AI 최종 누적 수익률 : {res_df['AI_Cum'].iloc[-1]:.4f}배")
    print(f"📉 시장 존버 대비 성과 : {(res_df['AI_Cum'].iloc[-1] - res_df['TQQQ_Cum'].iloc[-1]) * 100:.2f}%")

if __name__ == "__main__":
    run_real_sqqq_test()