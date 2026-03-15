import torch
import os
from data_loader.etf_data_process import SimpleStockDataset
from agent.model import FullStrategyModel
from trainer.trainer import train_full_strategy

def main():
    target_ticker = "TQQQ"
    ma_windows = [5, 20, 60, 200]
    
    # 1단계 학습 설정
    # 이평선 가중치라는 '핵심 뼈대'를 잡는 단계이므로 학습률을 너무 높이지 않습니다.
    config = {
        "train": {
            "batch_size": 1, 
            "learning_rate": 0.0005, 
            "epochs": 100 
        }
    }

    # 1. 데이터 로드 (Train 모드)
    print(f"📦 [{target_ticker}] 1단계: 이평선 최적화 데이터 로딩...")
    train_dataset = SimpleStockDataset(
        data_dir='data',
        target_ticker=target_ticker,
        mode='train',
        windows=ma_windows
    )

    if len(train_dataset) == 0:
        print("❌ 데이터를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 모델 초기화 (기울기, 익절/손절은 내부적으로 고정됨)
    model = FullStrategyModel(stock_feat_dim=len(ma_windows))

    # 3. 학습 실행 (trainer.py의 수정된 함수 호출)
    train_full_strategy(model, train_dataset, config)

    # 4. 1단계 결과 리포트
    print("\n" + "="*60)
    print("🎯 [1단계 완료] 이평선 조합 최적화 결과")
    print("-" * 60)
    
    with torch.no_grad():
        # Softmax를 통해 최종 결정된 가중치(합계 100%) 추출
        weights = torch.softmax(model.stock_weights, dim=-1).cpu().numpy()
        
        for i, window in enumerate(ma_windows):
            # 가중치가 높을수록 해당 이평선이 방향성 예측에 중요하다는 뜻입니다.
            bar = '█' * int(weights[i] * 20)
            print(f"🔹 {window:>3}일 이평선 비중: {weights[i]*100:6.2f}%  {bar}")
            
    print("-" * 60)
    print(f"📍 고정된 기울기 문턱값: {model.slope_threshold}")
    print(f"🚨 고정된 리스크 관리  : 손절 {model.stop_loss_threshold*100}% / 익절 {model.take_profit_threshold*100}%")
    print("="*60)
    print("💡 이 가중치가 마음에 든다면 2단계(문턱값/리스크 최적화)로 넘어갑니다.")

if __name__ == "__main__":
    main()