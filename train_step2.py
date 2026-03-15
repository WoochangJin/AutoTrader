import torch
from data_loader.etf_data_process import SimpleStockDataset
from agent.model import FullStrategyModel
from trainer.trainer_step2 import train_risk_parameters

def main():
    # 📍 보내주신 1단계 결과 가중치 (Softmax가 이미 적용된 값으로 입력)
    step1_weights = torch.tensor([0.2331, 0.2524, 0.2572, 0.2573])

    ma_windows = [5, 20, 60, 200]
    config = {"train": {"learning_rate": 0.0001, "epochs": 50}}

    train_dataset = SimpleStockDataset(data_dir='data', target_ticker="TQQQ", mode='train', windows=ma_windows)
    model = FullStrategyModel(stock_feat_dim=4, fixed_weights=step1_weights)

    train_risk_parameters(model, train_dataset, config)

    with torch.no_grad():
        print("\n" + "="*50)
        print("🏆 2단계: 리스크 파라미터 최적화 완료")
        print(f"📍 기울기 문턱값: {model.slope_threshold.item():.6f}")
        print(f"🚨 최적 손절: {model.stop_loss_threshold.item()*100:.2f}%")
        print(f"💰 최적 익절: {model.take_profit_threshold.item()*100:.2f}%")
        print("="*50)

if __name__ == "__main__":
    main()