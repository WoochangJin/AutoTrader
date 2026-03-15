import torch
import torch.nn as nn

class FullStrategyModel(nn.Module):
    # 📍 fixed_weights 인자를 받을 수 있도록 수정되었습니다.
    def __init__(self, stock_feat_dim=4, fixed_weights=None):
        super(FullStrategyModel, self).__init__()
        
        # 1. 이평선 가중치 설정
        if fixed_weights is not None:
            # [2단계] 가중치를 고정값(Buffer)으로 등록 (학습 제외)
            self.register_buffer('stock_weights', fixed_weights)
            
            # 리스크 파라미터들을 학습 대상으로 설정
            self.slope_threshold = nn.Parameter(torch.tensor(0.0001))
            self.stop_loss_threshold = nn.Parameter(torch.tensor(-0.07))
            self.take_profit_threshold = nn.Parameter(torch.tensor(0.25))
        else:
            # [1단계] 가중치 자체를 학습 대상으로 설정
            self.stock_weights = nn.Parameter(torch.ones(stock_feat_dim) * 2.0)
            
            # 리스크 파라미터들은 고정 (1단계에서는 방향성만 측정)
            self.register_buffer('slope_threshold', torch.tensor(0.0))
            self.register_buffer('stop_loss_threshold', torch.tensor(-0.07))
            self.register_buffer('take_profit_threshold', torch.tensor(0.25))
        
        self.steepness = 100.0

    def forward(self, current_feat, prev_feat, current_return):
        # stock_weights가 Parameter든 Buffer든 동일하게 weights로 취급
        # Parameter(1단계)일 때는 확률 밀도로 변환하기 위해 softmax 적용
        if isinstance(self.stock_weights, nn.Parameter):
            weights = torch.softmax(self.stock_weights, dim=-1)
        else:
            weights = self.stock_weights
        
        # AI 최적화 이평선(MA) 계산
        curr_ma = torch.sum(current_feat * weights, dim=-1)
        prev_ma = torch.sum(prev_feat * weights, dim=-1)
        slope = curr_ma - prev_ma
        
        # 1. 방향성 확률 (Sigmoid)
        direction_prob = torch.sigmoid(self.steepness * (slope - self.slope_threshold))
        
        # 2. 리스크 필터 (손절/익절)
        stop_signal = torch.sigmoid(100.0 * (current_return - self.stop_loss_threshold))
        profit_signal = torch.sigmoid(100.0 * (self.take_profit_threshold - current_return))
        
        # 최종 신호: (내일 오를 확률) * (손절선 내부여부) * (익절선 내부여부)
        return direction_prob * stop_signal * profit_signal