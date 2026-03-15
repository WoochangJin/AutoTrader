import torch
import torch.optim as optim
import torch.nn as nn

class StrategyTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam([model.stock_weights], lr=lr)
        self.criterion = nn.BCELoss()

    def train_step(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for i in range(1, len(dataset)):
            curr_feat, ret = dataset[i]
            prev_feat, _ = dataset[i-1]
            target = 1.0 if ret > 0 else 0.0
            output = self.model(curr_feat.unsqueeze(0), prev_feat.unsqueeze(0), torch.tensor([0.0]))
            total_loss += self.criterion(output, torch.tensor([target]))
            
        avg_loss = total_loss / (len(dataset)-1)
        
        # 📍 200일선 고착화 방지 페널티
        weights = torch.softmax(self.model.stock_weights / 2.0, dim=-1)
        # 마지막 가중치(200일선)가 0.3을 넘어가면 Loss를 대폭 증가시킴
        long_term_penalty = torch.pow(torch.clamp(weights[-1] - 0.3, min=0), 2) * 10.0
        
        loss = avg_loss + long_term_penalty
        loss.backward()
        self.optimizer.step()
        return loss.item()