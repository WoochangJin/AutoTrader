import torch
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader.etf_data_process import SimpleStockDataset
from agent.model import FullStrategyModel
from trainer.trainer_step2 import StrategyTrainerStep2 # 📍 경로 수정

def main():
    # 1단계 가중치 고정값
    step1_weights = torch.tensor([0.19, 0.20, 0.33, 0.28]) 
    ma_windows = [5, 20, 60, 200]

    dataset = SimpleStockDataset(data_dir='data', target_ticker="TQQQ", mode='train', windows=ma_windows)
    model = FullStrategyModel(stock_feat_dim=4, fixed_weights=step1_weights)
    trainer = StrategyTrainerStep2(model)

    print("🚀 2단계: 스위칭 및 리스크 파라미터 최적화...")
    for epoch in range(800):
        loss = trainer.train_epoch(dataset)
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1} | Sharpe Loss: {loss:.4f} | L_Th: {model.long_threshold.item():.2f} | S_Th: {model.short_threshold.item():.2f}")

    os.makedirs('saved', exist_ok=True)
    torch.save({'state_dict': model.state_dict()}, 'saved/checkpoint_step2.pth')
    print("✅ 2단계 완료 및 저장!")

if __name__ == "__main__":
    main()