import torch
import torch.optim as optim

class DipRipTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam([model.dip_threshold, model.rip_threshold], lr=lr)
        self.criterion = torch.nn.MSELoss()

    def train_epoch(self, dataset):
        self.model.train()
        total_loss = 0
        match_count = 0
        
        for i in range(len(dataset)):
            curr_feat, next_ret = dataset[i]
            signal = self.model.get_ai_signal(curr_feat.unsqueeze(0))
            
            # 📍 수정: 하락장 신호 + 실제로 오늘 주가가 조금이라도 떨어진 날만 학습
            if signal.item() <= self.model.short_threshold and next_ret.item() < -0.005:
                match_count += 1
                
                # UserWarning 방지를 위해 view(-1) 추가
                loss = self.criterion(self.model.dip_threshold.view(-1), next_ret.view(-1))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                # 모델 내부 clamp 실행
                self.model(curr_feat.unsqueeze(0))

        return total_loss / (match_count + 1e-6), match_count