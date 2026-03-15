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
from agent.model import SimpleStrategyModel

def run_test_only_visual():
    # 1. 모델 가중치 로드
    checkpoint_path = os.path.join(root_dir, 'saved/checkpoint_best.pth')
    if not os.path.exists(checkpoint_path):
        print("❌ 모델 파일이 없습니다.")
        return

    ma_windows = [5, 20, 60, 200]
    model = SimpleStrategyModel(stock_feat_dim=len(ma_windows))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # AI 가중치 추출 (시각화용 AI MA 계산을 위해)
    with torch.no_grad():
        weights = torch.softmax(model.stock_weights, dim=-1).numpy()

    # 2. TEST 데이터 로드
    dataset = SimpleStockDataset(
        data_dir=os.path.join(root_dir, 'data'), 
        target_ticker="TQQQ", 
        mode='test', 
        windows=ma_windows
    )
    df = dataset.df.copy()
    
    # AI 가중 평균 이평선 직접 계산
    df['AI_MA'] = sum(df[f'MA{w}'] * weights[i] for i, w in enumerate(ma_windows))
    
    results = []
    prev_ai_action = 0
    prev_ma200_action = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            stock_feat, ret = dataset[i]
            ai_signal = model(stock_feat.unsqueeze(0)).item()
            ai_action = 1 if ai_signal > 0.5 else 0
            
            # MA200 로직
            ma200_action = 1 if df['Close'].iloc[i] > df['MA200'].iloc[i] else 0
            
            results.append({
                'date': dataset.dates[i],
                'market_ret': ret.item(),
                'ai_strat_ret': prev_ai_action * ret.item(),
                'ma200_strat_ret': prev_ma200_action * ret.item(),
                'ai_signal': ai_signal
            })
            prev_ai_action = ai_action
            prev_ma200_action = ma200_action

    res_df = pd.DataFrame(results).set_index('date')
    res_df['AI_Cum'] = (1 + res_df['ai_strat_ret']).cumprod()
    res_df['MA200_Cum'] = (1 + res_df['ma200_strat_ret']).cumprod()

    # 4. 시각화 (2단 구성)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2]})
    
    # 상단: 누적 수익률 비교
    ax1.plot(res_df.index, res_df['AI_Cum'], label='AI Strategy', color='red', linewidth=2)
    ax1.plot(res_df.index, res_df['MA200_Cum'], label='MA200 Strategy', color='orange', alpha=0.7)
    ax1.set_title(f"Test Set Performance ({res_df.index[0].date()} ~ {res_df.index[-1].date()})")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # 하단: 가격 및 이평선 비교 (AI MA vs MA200)
    ax2.plot(df.index, df['Close'], label='TQQQ Close', color='black', alpha=0.2)
    ax2.plot(df.index, df['MA200'], label='MA200 (Traditional)', color='orange', linestyle='--')
    ax2.plot(df.index, df['AI_MA'], label='AI Optimized MA', color='red', linewidth=1.5)
    
    # AI 매수 구간 색칠 (신호가 0.5 이상인 구간)
    ax2.fill_between(df.index, df['Close'].min(), df['Close'].max(), 
                     where=(res_df['ai_signal'] > 0.5), color='red', alpha=0.05, label='AI Buy Zone')

    ax2.set_title("Price vs Moving Averages Comparison")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"🔹 AI 가중치 현황: [5일:{weights[0]:.2f}, 20일:{weights[1]:.2f}, 60일:{weights[2]:.2f}, 200일:{weights[3]:.2f}]")

if __name__ == "__main__":
    run_test_only_visual()