import torch
import os
from data_loader.etf_data_process import SimpleStockDataset
from agent.model import HybridStrategyModel
from trainer.dip_rip_trainer import DipRipTrainer

def main():
    # 1. 환경 설정
    fixed_weights = torch.tensor([0.19, 0.20, 0.33, 0.28]) # ASUS님 1단계 결과
    ma_windows = [5, 20, 60, 200]
    
    # 2. 모델 및 데이터 로드
    model = HybridStrategyModel(stock_feat_dim=4, fixed_weights=fixed_weights)
    dataset = SimpleStockDataset(data_dir='data', target_ticker="TQQQ", mode='train', windows=ma_windows)
    
    trainer = DipRipTrainer(model, lr=0.005)
    
    print(f"✅ 데이터 로드 완료 ({len(dataset)}일)")
    print("🚀 2단계: 떨사오팔(Dip/Rip) 최적화 시작...")

    for epoch in range(100):
        loss, count = trainer.train_epoch(dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.6f} | 하락장 횟수: {count} | "
                  f"Dip: {model.dip_threshold.item()*100:.2f}% | Rip: {model.rip_threshold.item()*100:.2f}%")

    # 3. 저장
    if not os.path.exists('saved'): os.makedirs('saved')
    torch.save(model.state_dict(), 'saved/hybrid_model_step2.pth')
    print("✨ 최적화 완료! saved/hybrid_model_step2.pth에 저장되었습니다.")

if __name__ == "__main__":
    main()