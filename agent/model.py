import torch
import torch.nn as nn

class HybridStrategyModel(nn.Module):
    def __init__(self, stock_feat_dim=4, fixed_weights=None):
        super(HybridStrategyModel, self).__init__()
        self.register_buffer('stock_weights', fixed_weights if fixed_weights is not None else torch.tensor([0.25]*4))
        
        # 📍 초기값을 -0.03(-3%)으로 설정
        self.dip_threshold = nn.Parameter(torch.tensor(-0.03))
        self.rip_threshold = nn.Parameter(torch.tensor(0.02))

        self.long_threshold = 0.6 
        self.short_threshold = 0.4 
        self.steepness = 10.0

    def get_ai_signal(self, current_feat):
        weighted_ma = torch.sum(current_feat * self.stock_weights, dim=-1)
        return torch.sigmoid(self.steepness * weighted_ma)

    def forward(self, current_feat):
        # 📍 Clamp 범위를 -10%에서 -0.1%까지 넓힘
        self.dip_threshold.data.clamp_(-0.10, -0.001)
        self.rip_threshold.data.clamp_(0.001, 0.10)
        return self.get_ai_signal(current_feat)