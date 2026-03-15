import torch
import torch.nn as nn

class FullStrategyModel(nn.Module):
    def __init__(self, stock_feat_dim=4, fixed_weights=None):
        super(FullStrategyModel, self).__init__()
        
        if fixed_weights is not None:
            # 📍 2단계: 가중치 고정 (학습 제외)
            self.register_buffer('stock_weights', fixed_weights)
            # 나머지 리스크 파라미터만 학습 대상으로 설정
            self.slope_threshold = nn.Parameter(torch.tensor(0.0001))
            self.stop_loss_threshold = nn.Parameter(torch.tensor(-0.07))
            self.take_profit_threshold = nn.Parameter(torch.tensor(0.25))
        else:
            # 📍 1단계: 이평선 가중치만 학습
            self.stock_weights = nn.Parameter(torch.ones(stock_feat_dim))
            # 나머지는 상수로 고정
            self.slope_threshold = 0.0
            self.stop_loss_threshold = -0.07
            self.take_profit_threshold = 0.25
        
        self.steepness = 20.0

    def forward(self, current_feat, prev_feat, current_return):
        # stock_weights가 Parameter든 Buffer든 동일하게 작동
        weights = torch.softmax(self.stock_weights, dim=-1) if isinstance(self.stock_weights, nn.Parameter) else self.stock_weights
        
        curr_ma = torch.sum(current_feat * weights, dim=-1)
        prev_ma = torch.sum(prev_feat * weights, dim=-1)
        slope = curr_ma - prev_ma
        
        direction_prob = torch.sigmoid(self.steepness * (slope - self.slope_threshold))
        stop_signal = torch.sigmoid(100.0 * (current_return - self.stop_loss_threshold))
        profit_signal = torch.sigmoid(100.0 * (self.take_profit_threshold - current_return))
        
        return direction_prob * stop_signal * profit_signal