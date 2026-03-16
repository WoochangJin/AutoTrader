import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader.etf_data_process import SimpleStockDataset
import os

# 1. 간단하지만 강력한 회귀 모델 정의
class TQQQPredictor(nn.Module):
    def __init__(self, input_dim):
        super(TQQQPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1) # 내일 예상 수익률 1개 출력
        )

    def forward(self, x):
        return self.net(x)

def train_model():
    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치: {device}")

    # 데이터 로드
    dataset = SimpleStockDataset(data_dir='data', target_ticker='TQQQ', mode='train')
    # 📍 전체 데이터를 셔플하여 학습 효율 극대화
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델 초기화 (입력 피처 8개)
    model = TQQQPredictor(input_dim=8).to(device)
    criterion = nn.MSELoss() # 수익률 차이를 줄이는 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"✅ 학습 시작 (데이터셋 크기: {len(dataset)}일)")
    print("-" * 50)

    # 2. 학습 루프
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 예측 및 오차 계산
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 가중치 업데이트 (최적화)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")

    # 3. 최적화된 모델 저장
    os.makedirs('saved', exist_ok=True)
    torch.save(model.state_dict(), 'saved/best_tqqq_model.pth')
    print("-" * 50)
    print("✨ 최적화 완료! 'saved/best_tqqq_model.pth'에 가중치가 저장되었습니다.")

if __name__ == "__main__":
    train_model()