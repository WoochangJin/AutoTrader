import torch
import torch.nn as nn

class FullStrategyModel(nn.Module):
    def __init__(self, stock_feat_dim=4, fixed_weights=None):
        super(FullStrategyModel, self).__init__()
        
        if fixed_weights is not None:
            self.register_buffer('stock_weights', fixed_weights)
        else:
            # 초기값: [5일, 20일, 60일, 200일]
            # 200일선(마지막) 로짓을 일부러 낮게 시작해서 고착화를 방어합니다.
            self.stock_weights = nn.Parameter(torch.tensor([1.2, 1.2, 0.8, 0.4]))
        
        self.slope_threshold = nn.Parameter(torch.tensor(0.0))
        self.stop_loss_threshold = nn.Parameter(torch.tensor(-0.07))
        self.take_profit_threshold = nn.Parameter(torch.tensor(0.25))
        self.short_threshold = nn.Parameter(torch.tensor(0.40)) 
        self.long_threshold = nn.Parameter(torch.tensor(0.60))  
        self.steepness = 100.0 

    def forward(self, current_feat, prev_feat, current_return):
        weights = self.stock_weights
        if isinstance(weights, nn.Parameter):
            # T=2.0으로 더 높여서 가중치가 아주 완만하게 섞이도록 합니다.
            weights = torch.softmax(weights / 2.0, dim=-1) 
            
        curr_ma = torch.sum(current_feat * weights, dim=-1)
        prev_ma = torch.sum(prev_feat * weights, dim=-1)
        slope = curr_ma - prev_ma
        
        prob = torch.sigmoid(self.steepness * (slope - self.slope_threshold))
        stop_signal = torch.sigmoid(100.0 * (current_return - self.stop_loss_threshold))
        profit_signal = torch.sigmoid(100.0 * (self.take_profit_threshold - current_return))
        
        return prob * stop_signal * profit_signal