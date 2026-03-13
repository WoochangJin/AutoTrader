import sys
import os
from datetime import datetime

# 프로젝트 루트 경로 추가 (api 폴더를 찾기 위함)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from api.kis_auth import KISAuth
from api.kis_overseas import KISOverseas

def test_overseas_price():
    print("--- [SOXL] 미국 주식 데이터 수집 테스트 시작 ---")
    
    # 1. 인증 객체 생성 및 토큰 발급
    auth = KISAuth()
    
    # 2. 해외 주식 객체 생성
    overseas = KISOverseas(auth)
    
    # 3. 조회 기준일 설정 (오늘 날짜 YYYYMMDD)
    target_date = datetime.now().strftime("%Y%m%d")
    symbol = "SOXL"
    
    print(f"조회 종목: {symbol}")
    print(f"조회 기준일: {target_date}")
    
    # 4. 데이터 호출
    result = overseas.get_overseas_daily_ohlcv(symbol, target_date)
    
    if result:
        # 한투 API의 output2는 일자별 데이터 리스트입니다.
        # 최근 5일치만 샘플로 출력해봅니다.
        print(f"\n✅ 최근 5거래일 시세 데이터:")
        print(f"{'날짜':<10} | {'종가':<8} | {'고가':<8} | {'저가':<8} | {'거래량':<12}")
        print("-" * 60)
        
        for day in result[:5]:
            date = day['xymd']
            close = day['clos']
            high = day['high']
            low = day['low']
            vol = day['tvol']
            print(f"{date:<10} | {close:<8} | {high:<8} | {low:<8} | {vol:<12}")
            
        print("\n--- 테스트 성공! 이 데이터를 활용해 이평선을 계산할 수 있습니다. ---")
    else:
        print("❌ 데이터 수집 실패. API 응답을 확인하세요.")

if __name__ == "__main__":
    test_overseas_price()
