import json
import torch
import os
from torch.utils.data import DataLoader
from etf_data_process import UniversalStockDataset
from agent.model import UniversalStrategyModel
from trainer.trainer import StrategyTrainer

def main():
    # 1. Config 로드 (합쳐진 새로운 구조 대응)
    with open('config.json', 'r') as f:
        config = json.load(f)

    # 파라미터 경로 변수화
    strat_params = config['strategy']['params']
    train_params = config['model_training']

    # 2. 데이터 로더 설정
    # config의 ma_windows를 데이터셋 생성 시 전달
    train_ds = UniversalStockDataset(
        data_dir='data', 
        target_ticker=train_params['target_ticker'], 
        mode='train',
        windows=strat_params['ma_windows']
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_params['batch_size'], 
        shuffle=False
    )

    # 3. 모델 초기화
    # 입력 차원을 config의 이평선 개수에 따라 동적으로 설정
    model = UniversalStrategyModel(
        stock_feat_dim=len(strat_params['ma_windows']), 
        macro_feat_dim=2  # DXY, TNX
    )

    # 4. 트레이너 실행 (config 전체 전달)
    # trainer 내부에서 lr, epochs 등을 가져다 씁니다.
    trainer = StrategyTrainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=train_loader, 
        config=config, 
        device="cpu"
    )

    print(f"🚀 {train_params['target_ticker']} 학습 시작...")
    best_loss = float('inf')
    
    for epoch in range(train_params['epochs']):
        loss = trainer.train_epoch()
        
        # 성능 개선 시 저장
        if loss < best_loss:
            best_loss = loss
            trainer.save_checkpoint(epoch, loss)
            
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{train_params['epochs']}] - Loss: {loss:.4f}")

    print("✨ 학습 완료! 최적의 파라미터가 saved/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()