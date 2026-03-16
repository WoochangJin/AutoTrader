import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class SimpleStockDataset(Dataset):
    def __init__(self, data_dir, target_ticker, mode='train', start_date=None, end_date=None):
        self.file_path = os.path.join(data_dir, mode, f"{target_ticker}_daily.csv")
        
        # 파일 로드
        df = pd.read_csv(self.file_path, index_col='Date', parse_dates=True)
        df = df.sort_index()

        # 📍 날짜 필터링 추가
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        # MA 기울기 및 피처 처리 (기존 로직 동일)
        ma_cols = ['MA5', 'MA20', 'MA200']
        for col in ma_cols:
            df[f'{col}_Slope'] = df[col].pct_change()

        self.feature_cols = ['MA5_Slope', 'MA20_Slope', 'MA200_Slope', 'RSI', 'CCI', 'Vol_ROC', 'VIX', 'US10Y']
        df = df.dropna(subset=self.feature_cols)
        
        self.df = df
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.df[self.feature_cols])
        self.target_returns = self.df['Close'].pct_change().shift(-1).fillna(0).values

    # ... (__len__, __getitem__ 등은 동일)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.features[idx]), 
                torch.FloatTensor([self.target_returns[idx]]))

# 📍 검증 코드
if __name__ == "__main__":
    dataset = SimpleStockDataset(data_dir='data', target_ticker='TQQQ', mode='train')
    print(f"✅ 피처 개수: {len(dataset.feature_cols)}개")
    print(f"📊 첫 날 피처(정규화됨): \n{dataset[0][0]}")