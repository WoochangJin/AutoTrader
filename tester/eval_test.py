import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 📍 1. 경로 문제 해결: 현재 파일(tester/eval_model.py)의 부모 폴더(루트)를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

# 이제 상위 폴더의 모듈들을 정상적으로 가져올 수 있습니다.
from data_loader.etf_data_process import SimpleStockDataset
from train import TQQQPredictor 

def evaluate_and_plot(mode='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 📍 2. 데이터 및 모델 경로를 루트 기준으로 재설정
    data_path = os.path.join(root_dir, "data")
    model_path = os.path.join(root_dir, "saved", "best_tqqq_model.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    # 데이터 로드
    dataset = SimpleStockDataset(data_dir=data_path, target_ticker='TQQQ', mode=mode)
    
    # 모델 로드
    model = TQQQPredictor(input_dim=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 예측값 추출
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            features, target = dataset[i]
            features = features.to(device).unsqueeze(0)
            
            pred = model(features)
            all_preds.append(pred.item())
            all_targets.append(target.item())

    # 결과 시각화
    plt.figure(figsize=(15, 8))
    
    # 상단: 최근 100일 수익률 방향성 비교
    plt.subplot(2, 1, 1)
    plt.plot(all_targets[-100:], label='Actual Return', color='gray', alpha=0.4)
    plt.plot(all_preds[-100:], label='AI Prediction', color='red', lw=1.5)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title(f"Return Direction Check ({mode.upper()} Set - Last 100 days)")
    plt.legend()

    # 하단: 누적 수익률 비교
    plt.subplot(2, 1, 2)
    acc_actual = (1 + np.array(all_targets)).cumprod()
    # 전략: AI가 0보다 큰 수익률을 예측할 때만 매수
    strat_returns = [all_targets[i] if all_preds[i] > 0 else 0 for i in range(len(all_preds))]
    acc_strat = (1 + np.array(strat_returns)).cumprod()
    
    plt.plot(acc_actual, label='TQQQ Buy & Hold', color='gray', alpha=0.4)
    plt.plot(acc_strat, label='AI Logic Strategy', color='blue', lw=2)
    plt.title(f"Cumulative Wealth Comparison ({mode.upper()} Set)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("📈 [1/2] TRAIN 데이터셋 평가 중...")
    evaluate_and_plot(mode='train')
    
    print("\n🚀 [2/2] TEST 데이터셋 평가 중...")
    evaluate_and_plot(mode='test')