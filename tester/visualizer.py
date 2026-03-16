import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_combined_with_tqqq_ma():
    tqqq_path = "data/TQQQ_daily.csv"
    dxy_path = "data/DXY_daily.csv"
    
    if not os.path.exists(tqqq_path) or not os.path.exists(dxy_path):
        print("❌ 데이터 파일이 없습니다.")
        return

    try:
        # 1. 데이터 로드
        df_tqqq = pd.read_csv(tqqq_path, header=[0, 1, 2], index_col=0, parse_dates=True)
        df_dxy = pd.read_csv(dxy_path, header=[0, 1, 2], index_col=0, parse_dates=True)
        
        # 2. 종가 및 TQQQ 자체 이평선 추출
        tqqq_close = df_tqqq.xs('Close', axis=1, level=0).iloc[:, 0]
        tqqq_ma200 = df_tqqq.xs('MA200', axis=1, level=0).iloc[:, 0]
        dxy_close = df_dxy.xs('Close', axis=1, level=0).iloc[:, 0]
        
        # 3. 데이터 병합 및 결측치 제거
        df = pd.DataFrame({
            'TQQQ': tqqq_close, 
            'TQQQ_MA200': tqqq_ma200, 
            'DXY': dxy_close
        }).dropna()
        
        # 4. 정규화 (시작점 100 기준)
        # 중요: MA200도 TQQQ 종가와 같은 비율로 스케일링해야 합니다.
        base_tqqq = df['TQQQ'].iloc[0]
        base_dxy = df['DXY'].iloc[0]
        
        df['TQQQ_norm'] = (df['TQQQ'] / base_tqqq) * 100
        df['TQQQ_MA200_norm'] = (df['TQQQ_MA200'] / base_tqqq) * 100
        df['DXY_norm'] = (df['DXY'] / base_dxy) * 100
        
        # 5. 복합 지수 생성 (TQQQ 70% : DXY 30%)
        w_tqqq, w_dxy = 0.7, 0.3
        df['Combined_Index'] = (df['TQQQ_norm'] * w_tqqq) + (df['DXY_norm'] * w_dxy)
        
        # 6. 그래프 시각화
        plt.figure(figsize=(14, 8))
        
        # 복합 지수 (보라색)
        plt.plot(df.index, df['Combined_Index'], label='Combined Index (TQQQ+DXY)', color='purple', alpha=0.5)
        
        # 정규화된 TQQQ 종가 (하늘색 - 참고용)
        plt.plot(df.index, df['TQQQ_norm'], label='TQQQ (Normalized)', color='dodgerblue', alpha=0.3, linestyle=':')
        
        # TQQQ의 200일 이평선 (빨간색 - 실제 기준선)
        plt.plot(df.index, df['TQQQ_MA200_norm'], label='TQQQ 200-day MA (Normalized)', color='red', linewidth=2.5)
        
        # 7. 스타일링
        plt.title('Combined Index vs TQQQ 200-Day Moving Average', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Normalized Scale')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        print("📊 차트 팝업을 띄웁니다. 복합 지수가 TQQQ 이평선을 뚫는지 확인해보세요!")
        plt.show()

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    plot_combined_with_tqqq_ma()