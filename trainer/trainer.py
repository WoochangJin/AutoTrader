import torch
import os
import json

class StrategyTrainer:
    def __init__(self, model, train_loader, val_loader, config, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config  # config.json 내용을 담은 딕셔너리
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['trainer']['lr'])
        self.save_dir = config['trainer']['save_dir'] # "saved/"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_checkpoint(self, epoch, loss):
        """최적의 모델 파라미터와 설정을 저장"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'loss': loss
        }
        filename = os.path.join(self.save_dir, 'checkpoint_best.pth')
        torch.save(state, filename)
        
        # 가중치값만 따로 JSON으로 저장 (가독성용)
        weights_info = {
            "stock_weights": torch.softmax(self.model.stock_weights, dim=0).tolist(),
            "macro_weights": torch.softmax(self.model.macro_weights, dim=0).tolist(),
            "threshold": self.model.threshold.item()
        }
        with open(os.path.join(self.save_dir, 'best_weights.json'), 'w') as f:
            json.dump(weights_info, f, indent=4)
        
        print(f"✅ 모델 및 가중치 저장 완료: {filename}")

    # ... (기존 custom_loss 및 train_epoch 로직) ...

    def custom_loss(self, signals, returns):
        """
        개월 대비 수익률(시간 효율)을 최적화하는 Loss 함수
        - signals: 모델이 내뱉은 매수 강도 (0~1)
        - returns: 내일의 수익률
        """
        # 1. 전략의 일일 수익률 계산 (매수 강도 * 실제 수익률)
        strategy_returns = signals * returns
        
        # 2. 총 누적 수익 (Log return의 합으로 계산하여 기하급수적 성장 반영)
        # 단순히 더하는 게 아니라 기하 평균적인 성장을 돕기 위해 log를 활용하거나 
        # 직접적으로 평균 수익률을 높이는 방향으로 설정
        portfolio_return = torch.mean(strategy_returns)
        
        # 3. 변동성(위험) 계산 - 너무 들쭉날쭉한 수익은 감점
        portfolio_std = torch.std(strategy_returns) + 1e-6
        
        # 4. '시간 대비 수익률' 최적화 (Sharpe Ratio와 유사)
        # 분모에 변동성을 넣어 휩쏘에 의한 손실을 방지하고, 
        # 분자에는 평균 수익을 두어 기간 대비 효율을 극대화
        loss = -(portfolio_return / portfolio_std) 
        
        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for stock, macro, ret in self.train_loader:
            stock, macro, ret = stock.to(self.device), macro.to(self.device), ret.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 모델의 매수 신호 출력
            signals = self.model(stock, macro)
            
            # 커스텀 손실 함수 적용
            loss = self.custom_loss(signals, ret)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_profit = 0
        with torch.no_grad():
            for stock, macro, ret in self.val_loader:
                stock, macro, ret = stock.to(self.device), macro.to(self.device), ret.to(self.device)
                signals = self.model(stock, macro)
                # 단순 누적 수익률 계산 (검증용)
                val_profit += torch.sum(signals * ret).item()
        return val_profit