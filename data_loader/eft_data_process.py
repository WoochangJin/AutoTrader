import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os

class UniversalStockDataset(Dataset):
    def __init__(self, data_dir, target_ticker="TQQQ", mode='train', windows=[5, 20, 60, 200]):
        self.data_dir = data_dir
        self.target_ticker = target_ticker
        self.mode = mode
        self.windows = windows
        
        # 1. 파일 경로 확인
        self.base_path = os.path.join(data_dir, mode)
        
        # 2. 데이터 병합 및 전처리
        df = self._prepare_data()
        
        # 3. 컬럼 이름 정의 (범용)
        self.stock_cols = [f'MA{w}' for w in windows]
        self.macro_cols = ['DXY_Close', 'TNX_Close']
        
        # 4. 정규화
        self.scaler = StandardScaler()
        all_features = self.stock_cols + self.macro_cols
        df[all_features] = self.scaler.fit_transform(df[all_features])
        
        # 5. Tensor 변환
        self.stock_data = torch.tensor(df[self.stock_cols].values, dtype=torch.float32)
        self.macro_data = torch.tensor(df[self.macro_cols].values, dtype=torch.float32)
        self.returns = torch.tensor(df['Target_Return'].values, dtype=torch.float32)

    def _prepare_data(self):
        # 대상 종목과 매크로 지표 로드
        target_df = pd.read_csv(f"{self.base_path}/{self.target_ticker}_daily.csv", index_col=0, parse_dates=True)
        dxy_df = pd.read_csv(f"{self.base_path}/DXY_daily.csv", index_col=0, parse_dates=True)
        tnx_df = pd.read_csv(f"{self.base_path}/US10Y_Yield_daily.csv", index_col=0, parse_dates=True)

        # 데이터 병합 (Inner join으로 휴장일 등 일치시킴)
        combined = pd.DataFrame(index=target_df.index)
        combined['Close'] = target_df['Close']
        combined['DXY_Close'] = dxy_df['Close']
        combined['TNX_Close'] = tnx_df['Close']
        combined = combined.dropna()

        # 이평선 생성
        for w in self.windows:
            combined[f'MA{w}'] = combined['Close'].rolling(window=w).mean()
        
        # 타겟 수익률 (내일의 등락률)
        combined['Target_Return'] = combined['Close'].pct_change().shift(-1)
        
        return combined.dropna()

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx):
        return self.stock_data[idx], self.macro_data[idx], self.returns[idx]