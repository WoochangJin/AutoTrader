import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import json

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

from data_loader.etf_data_process import SimpleStockDataset
from agent.model import SimpleStrategyModel

def run_pure_test():
    # 1. 모델 가중치 로드
    checkpoint_path = os.path.join(root_dir, 'saved/checkpoint_best.pth')
    if not os.path.exists(checkpoint_path):
        print("❌ 모델 파일이 없습니다. 먼저 학습을 완료해주세요.")
        return

    ma_windows = [5, 20, 60, 200]
    model = SimpleStrategyModel(stock_feat_dim=len(ma_windows))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2. 오직 'test' 데이터만 로드
    # (학습 과정에서 한 번도 보지 못한 데이터셋입니다)
    dataset = SimpleStockDataset(
        data_dir=os.path.join(root_dir, 'data'), 
        target_ticker="TQQQ", 
        mode='test', 
        windows=ma_windows
    )
    
   # ... (앞부분 로드 및 모델 준비 로직 동일)

    results = []
    # 이전 날의 신호를 저장할 변수 (초기값은 현금 보유 0)
    prev_ai_action = 0
    prev_ma200_action = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            stock_feat, ret = dataset[i]
            
            # 1. '오늘' 발생한 수익률 (ret)
            # 이 수익률은 '어제' 결정한 포지션(prev_action)에 의해 내 계좌에 반영됨
            market_daily_ret = ret.item()
            ai_strat_ret = prev_ai_action * market_daily_ret
            ma200_strat_ret = prev_ma200_action * market_daily_ret
            
            # 2. '오늘' 종가 데이터를 보고 '내일' 취할 행동을 결정 (Signal Generation)
            ai_signal = model(stock_feat.unsqueeze(0)).item()
            current_ai_action = 1 if ai_signal > 0.5 else 0
            
            current_close = dataset.df['Close'].iloc[i]
            current_ma200 = dataset.df['MA200'].iloc[i]
            current_ma200_action = 1 if current_close > current_ma200 else 0
            
            # 결과 저장
            results.append({
                'date': dataset.dates[i],
                'market_ret': market_daily_ret,
                'ai_strat_ret': ai_strat_ret,
                'ma200_strat_ret': ma200_strat_ret
            })
            
            # 3. 오늘의 결정을 다음 날로 넘김
            prev_ai_action = current_ai_action
            prev_ma200_action = current_ma200_action

# ... (이후 누적 수익률 계산 및 시각화 동일) 