import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class SimpleStockDataset(Dataset):
    def __init__(self, data_dir, target_ticker, mode='train', windows=[5, 20, 60, 200]):
        self.windows = windows
        paths = [os.path.join(data_dir, 'train'), os.path.join(data_dir, 'test')] if mode == 'all' else [os.path.join(data_dir, mode)]
        
        colnames = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'MA200', 'MA60']
        dfs = []
        for p in paths:
            file_p = os.path.join(p, f"{target_ticker}_daily.csv")
            if os.path.exists(file_p):
                # 📍 ASUS님 특수 파일 형식 대응 (3줄 스킵)
                temp = pd.read_csv(file_p, skiprows=3, names=colnames, index_col='Date', parse_dates=True)
                dfs.append(temp)
        
        if not dfs: raise ValueError(f"파일을 찾을 수 없습니다: {target_ticker}")

        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        for w in self.windows:
            df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
        
        self.df = df.dropna()
        self.dates = self.df.index
        
        # 정규화
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.df[[f'MA{w}' for w in self.windows]])
        # 수익률: 내일 종가 / 오늘 종가 - 1 (학습 정답지)
        self.target_returns = self.df['Close'].pct_change().shift(-1).fillna(0).values

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.target_returns[idx]])