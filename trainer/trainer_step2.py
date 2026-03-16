import torch
import torch.optim as optim

class StrategyTrainerStep2:
    def __init__(self, model, lr=0.005):
        self.model = model
        # 2단계는 가중치 빼고 나머지 파라미터(Thresh 등)만 학습
        params = [p for n, p in model.named_parameters() if "stock_weights" not in n]
        self.optimizer = optim.Adam(params, lr=lr)

    def train_epoch(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        
        total_returns = []
        for i in range(1, len(dataset)):
            curr_feat, ret = dataset[i]
            prev_feat, _ = dataset[i-1]
            
            signal = self.model(curr_feat.unsqueeze(0), prev_feat.unsqueeze(0), torch.tensor([0.0]))
            
            # 미분 가능한 포지션 결정 (Long-Short 스위칭 유도)
            pos = torch.tanh(10.0 * (signal - 0.5)) 
            total_returns.append(pos * ret)
            
        returns_tensor = torch.stack(total_returns)
        sharpe = returns_tensor.mean() / (returns_tensor.std() + 1e-6)
        
        loss = -sharpe
        loss.backward()
        self.optimizer.step()
        return loss.item()