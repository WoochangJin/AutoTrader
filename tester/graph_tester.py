import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

# 모델 구조는 agent/model.py에 정의된 것을 가져온다고 가정합니다.
from agent.model import FullStrategyModel

def run_final_backtest():
    print("🔎 [최종 테스트] 메타데이터 스킵 및 가중치 반영 백테스트 시작...")
    
    # ---------------------------------------------------------
    # 2. 설정 및 파라미터 (ASUS님 최적화 결과 반영)
    # ---------------------------------------------------------
    step1_weights = torch.tensor([0.19, 0.20, 0.33, 0.28]) 
    ma_windows = [5, 20, 60, 200]
    fee_rate = 0.001 # 왕복 수수료 0.2% 가정
    
    # 모델 초기화 및 2단계 체크포인트 로드
    model = FullStrategyModel(stock_feat_dim=len(ma_windows), fixed_weights=step1_weights)
    checkpoint_path = os.path.join(root_dir, 'saved', 'checkpoint_step2.pth')
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'], strict=False)
        print("✅ saved/checkpoint_step2.pth 로드 성공")
    else:
        print("⚠️ 체크포인트가 없어 기본 문턱값(0.6 / 0.4)으로 시뮬레이션합니다.")
    model.eval()

    # ---------------------------------------------------------
    # 3. 데이터 로드 (상단 3줄 스킵 로직 적용)
    # ---------------------------------------------------------
    tqqq_path = os.path.join(root_dir, 'data', 'test', 'TQQQ_daily.csv')
    sqqq_path = os.path.join(root_dir, 'data', 'test', 'SQQQ_daily.csv')
    
    # 파일 내 컬럼 순서에 맞게 강제 지정
    colnames = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'MA200', 'MA60']
    
    try:
        # skiprows=3으로 상단 메타데이터 제거
        df = pd.read_csv(tqqq_path, skiprows=3, names=colnames, index_col='Date', parse_dates=True)
        sqqq_df = pd.read_csv(sqqq_path, skiprows=3, names=colnames, index_col='Date', parse_dates=True)
        print("✅ TQQQ/SQQQ 데이터 로드 및 전처리 완료")
    except Exception as e:
        print(f"❌ 데이터 로드 중 에러 발생: {e}")
        return

    # ---------------------------------------------------------
    # 4. 지표 계산 및 데이터 정렬
    # ---------------------------------------------------------
    # 이평선 다시 계산 (기존 파일에 있는 MA 대신 학습 때 쓴 로직 그대로 적용)
    for w in ma_windows:
        df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
    
    df = df.dropna()
    # AI MA (하단 차트용 빨간 실선)
    df['AI_MA'] = sum(df[f'MA{w}'] * step1_weights[i].item() for i, w in enumerate(ma_windows))
    
    # 날짜 동기화
    common_dates = df.index.intersection(sqqq_df.index)
    df = df.loc[common_dates].sort_index()
    sqqq_df = sqqq_df.loc[common_dates].sort_index()

    # ---------------------------------------------------------
    # 5. 백테스트 시뮬레이션
    # ---------------------------------------------------------
    results = []
    prev_pos = 0 # 1: TQQQ, -1: SQQQ, 0: Cash
    dates = df.index.tolist()

    for i in range(1, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i-1]
        
        # 모델 입력값 생성
        curr_feat = torch.tensor([df.loc[curr_date, f'MA{w}'] for w in ma_windows]).float()
        prev_feat = torch.tensor([df.loc[prev_date, f'MA{w}'] for w in ma_windows]).float()
        
        tqqq_ret = (df.loc[curr_date, 'Close'] - df.loc[prev_date, 'Close']) / df.loc[prev_date, 'Close']
        sqqq_ret = (sqqq_df.loc[curr_date, 'Close'] - sqqq_df.loc[prev_date, 'Close']) / sqqq_df.loc[prev_date, 'Close']

        with torch.no_grad():
            ai_val = model(curr_feat.unsqueeze(0), prev_feat.unsqueeze(0), torch.tensor([0.0])).item()
        
        # 문턱값 추출
        l_th = model.long_threshold.item()
        s_th = model.short_threshold.item()
        
        # 포지션 결정
        if ai_val >= l_th: curr_pos = 1
        elif ai_val <= s_th: curr_pos = -1
        else: curr_pos = 0
        
        # 수익률 계산
        daily_ret = 0.0
        if prev_pos == 1: daily_ret = tqqq_ret
        elif prev_pos == -1: daily_ret = sqqq_ret
        
        # 스위칭 수수료 적용
        if prev_pos != curr_pos:
            daily_ret -= (fee_rate * 2 if (prev_pos * curr_pos == -1) else fee_rate)

        results.append({
            'date': curr_date,
            'strat_ret': daily_ret,
            'market_ret': tqqq_ret,
            'pos': curr_pos,
            'price': df.loc[curr_date, 'Close']
        })
        prev_pos = curr_pos

    res_df = pd.DataFrame(results).set_index('date')
    res_df['AI_Cum'] = (1 + res_df['strat_ret']).cumprod()
    res_df['Market_Cum'] = (1 + res_df['market_ret']).cumprod()

    # ---------------------------------------------------------
    # 6. 결과 시각화
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # 수익률 차트
    ax1.plot(res_df.index, res_df['AI_Cum'], label='AI Strategy (TQQQ/SQQQ)', color='purple', lw=2.5)
    ax1.plot(res_df.index, res_df['Market_Cum'], label='TQQQ Buy & Hold', color='gray', alpha=0.3, ls='--')
    ax1.set_yscale('log')
    ax1.set_title("Strategy Performance (Log Scale)")
    ax1.legend(); ax1.grid(True, alpha=0.2)

    # 가격 및 포지션 차트
    ax2.plot(res_df.index, res_df['price'], color='black', alpha=0.2, label='TQQQ Price')
    ax2.plot(res_df.index, df.loc[res_df.index, 'AI_MA'], color='red', lw=1.5, label='AI Optimized MA')
    
    # 포지션 영역 표시
    ax2.fill_between(res_df.index, 0, 1, where=(res_df['pos'] == 1), color='red', alpha=0.15, label='Long (TQQQ)', transform=ax2.get_xaxis_transform())
    ax2.fill_between(res_df.index, 0, 1, where=(res_df['pos'] == -1), color='blue', alpha=0.15, label='Short (SQQQ)', transform=ax2.get_xaxis_transform())
    
    ax2.set_title("Price & AI Signal Visualization")
    ax2.legend(loc='upper left', ncol=4)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"\n📊 최종 백테스트 결과")
    print(f"💰 전략 수익률 : {res_df['AI_Cum'].iloc[-1]:.4f}배")
    print(f"📉 시장 수익률 : {res_df['Market_Cum'].iloc[-1]:.4f}배")

if __name__ == "__main__":
    run_final_backtest()