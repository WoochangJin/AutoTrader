import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class LocalStockGui:
    def __init__(self, root):
        self.root = root
        self.root.title("QuantTrader 로컬 데이터 조회기")
        self.root.geometry("1000x800")

        # --- 경로 설정 ---
        # 실행 위치(tester/) 기준으로 데이터 폴더 경로 설정
        self.base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

        # --- 입력부 ---
        input_frame = tk.Frame(self.root)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 모드 선택 (train / test)
        tk.Label(input_frame, text="데이터 모드:").grid(row=0, column=0, padx=5)
        self.mode_var = tk.StringVar(value="test")
        tk.Radiobutton(input_frame, text="Test", variable=self.mode_var, value="test").grid(row=0, column=1)
        tk.Radiobutton(input_frame, text="Train", variable=self.mode_var, value="train").grid(row=0, column=2)

        # 종목 입력
        tk.Label(input_frame, text="종목명(예: TQQQ, SOXL):").grid(row=0, column=3, padx=5)
        self.ticker_entry = tk.Entry(input_frame, width=20)
        self.ticker_entry.insert(0, "TQQQ, SOXL")
        self.ticker_entry.grid(row=0, column=4, padx=5)

        self.search_btn = tk.Button(input_frame, text="로컬 파일 조회", command=self.update_chart, bg="#3498db", fg="white", font=('Arial', 10, 'bold'))
        self.search_btn.grid(row=0, column=5, padx=15)

        # --- 그래프 출력부 ---
        plt.style.use('dark_background') # 코딩 감성용 다크 모드
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_chart(self):
        mode = self.mode_var.get()
        tickers = [t.strip().upper() for t in self.ticker_entry.get().split(",")]
        
        try:
            self.ax.clear()
            found_any = False
            colnames = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'MA200', 'MA60']

            for ticker in tickers:
                # 📍 ASUS님 파일명 규칙 적용 (_daily.csv)
                file_name = f"{ticker}_daily.csv"
                file_path = os.path.join(self.base_path, mode, file_name)

                if os.path.exists(file_path):
                    # 상단 3줄 메타데이터 스킵 로직 적용
                    df = pd.read_csv(file_path, skiprows=3, names=colnames, index_col='Date', parse_dates=True)
                    
                    if not df.empty:
                        self.ax.plot(df.index, df['Close'], label=f"{ticker} ({mode})", lw=1.5)
                        found_any = True
                else:
                    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")

            if found_any:
                self.ax.set_title(f"Local Data View: {', '.join(tickers)}", color='white', fontsize=12)
                self.ax.set_ylabel("Price (USD)")
                self.ax.legend()
                self.ax.grid(True, alpha=0.2)
                self.fig.tight_layout()
                self.canvas.draw()
            else:
                messagebox.showwarning("파일 없음", f"선택한 {mode} 폴더에 해당 csv 파일이 없습니다.")
            
        except Exception as e:
            messagebox.showerror("에러", f"파일을 읽는 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LocalStockGui(root)
    root.mainloop()