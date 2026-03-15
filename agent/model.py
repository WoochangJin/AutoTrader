import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalStrategyModel(nn.Module):
    def __init__(self, stock_feat_dim, macro_feat_dim):
        """
        stock_feat_dim: 이평선 개수 (예: 5, 20, 60, 200일선이면 4)
        macro_feat_dim: 매크로 지표 개수 (DXY, TNX면 2)
        """
        super(UniversalStrategyModel, self).__init__()
        
        # 1. 주식 그룹 가중치 (이평선 간의 중요도)
        self.stock_weights = nn.Parameter(torch.randn(stock_feat_dim))
        
        # 2. 매크로 그룹 가중치 (매크로 지표 간의 중요도)
        self.macro_weights = nn.Parameter(torch.randn(macro_feat_dim))
        
        # 3. 비교 임계치 (Threshold)
        # 초기값은 0.05(5%) 정도로 설정
        self.threshold = nn.Parameter(torch.tensor([0.05]))
        
        # 4. 시그널 민감도 (Steepness)
        # 시그모이드 함수의 경사도를 조절하여 0과 1 사이의 출력을 명확하게 만듭니다.
        self.steepness = nn.Parameter(torch.tensor([10.0]))

    def forward(self, stock_features, macro_features):
        """
        stock_features: (batch, stock_feat_dim)
        macro_features: (batch, macro_feat_dim)
        """
        # Softmax를 통해 각 그룹 내 가중치 합을 1로 유지 (해석 용이성)
        sw = F.softmax(self.stock_weights, dim=0)
        mw = F.softmax(self.macro_weights, dim=0)
        
        # 가중합 계산 (각 그룹의 점수 산출)
        stock_score = torch.sum(stock_features * sw, dim=1)
        macro_score = torch.sum(macro_features * mw, dim=1)
        
        # [주식 점수] vs [매크로 점수 + 임계치] 비교
        # 차이가 클수록 1(매수)에 가까워지고, 작으면 0(관망)에 가까워짐
        diff = stock_score - (macro_score + self.threshold)
        signal = torch.sigmoid(diff * self.steepness)
        
        return signal