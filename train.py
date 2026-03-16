import torch
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader.etf_data_process import SimpleStockDataset
from agent.model import FullStrategyModel
from trainer.trainer import StrategyTrainer

def main():
    ma_windows = [5, 20, 60, 200]
    dataset = SimpleStockDataset(data_dir='data', target_ticker="TQQQ", mode='train', windows=ma_windows)
    model = FullStrategyModel(stock_feat_dim=4)
    trainer = StrategyTrainer(model, lr=0.001)

    print(f"🚀 1단계: 200일선 억제 및 정밀 최적화 (1000 Epochs / Data: {len(dataset)})")
    
    for epoch in range(800):
        loss = trainer.train_step(dataset)
        if (epoch+1) % 100 == 0:
            weights = torch.softmax(model.stock_weights / 2.0, dim=-1)
            w_np = weights.detach().numpy()
            print(f"Epoch {epoch+1:4d} | Loss: {loss:.6f} | 5d:{w_np[0]:.2f} 20d:{w_np[1]:.2f} 60d:{w_np[2]:.2f} 200d:{w_np[3]:.2f}")

    os.makedirs('saved', exist_ok=True)
    torch.save({'state_dict': model.state_dict()}, 'saved/checkpoint_step1.pth')
    print("✅ 200일선 독식 방지 학습 완료!")

if __name__ == "__main__":
    main()