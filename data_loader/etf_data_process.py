import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch

class SimpleStockDataset:
    def __init__(self, data_dir, target_ticker, mode='train', windows=[5, 20, 60, 200]):
        self.data_dir = data_dir
        self.target_ticker = target_ticker
        self.windows = windows
        
        # 경로 설정 (all 모드 지원)
        paths = [os.path.join(data_dir, 'train'), os.path.join(data_dir, 'test')] if mode == 'all' else [os.path.join(data_dir, mode)]
        
        # 데이터 병합 및 정제
        dfs = []
        for p in paths:
            file_p = f"{p}/{target_ticker}_daily.csv"
            if os.path.exists(file_p):
                temp = pd.read_csv(file_p, index_col=0)
                temp.index = pd.to_datetime(temp.index, errors='coerce')
                dfs.append(temp)
        
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        # 이평선 계산
        for w in self.windows:
            df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
        
        self.df = df.dropna()
        self.dates = self.df.index
        
        # 정규화 (이평선들만)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.df[[f'MA{w}' for w in self.windows]])
        self.target_returns = self.df['Close'].pct_change().shift(-1).fillna(0).values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.target_returns[idx]])