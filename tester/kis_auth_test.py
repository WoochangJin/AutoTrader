import sys
import os

# 현재 파일 위치 기준으로 프로젝트 루트(QuantTrader)를 path에 추가
# 그래야 'from api import KISAuth'를 인식합니다.
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from api.kis_auth import KISAuth
from dotenv import load_dotenv

def test_connection():
    print("--- 한국투자증권 연결 테스트 시작 ---")
    
    # 1. KISAuth 인스턴스 생성
    auth = KISAuth()
    
    # 2. .env 데이터 로드 확인
    if not auth.app_key or not auth.app_secret:
        print("❌ 에러: .env 파일에서 API Key를 읽어오지 못했습니다.")
        print("파일 위치와 변수명을 다시 확인해주세요.")
        return

    print(f"✔️ API Key 로드 완료: {auth.app_key[:5]}*****")

    # 3. 토큰 발급 요청
    token = auth.get_token()
    
    if token:
        print(f"✔️ 토큰 발급 성공!")
        print(f"✔️ 발급된 토큰(일부): {token[:20]}...")
        
        # 4. 헤더 구성 확인
        headers = auth.auth_headers
        if "Authorization" in headers:
            print("✔️ 인증 헤더 구성 완료.")
            print("--- 테스트 성공! 이제 주가 조회를 시도해도 좋습니다. ---")
    else:
        print("❌ 테스트 실패: 토큰을 받아오지 못했습니다.")

if __name__ == "__main__":
    test_connection()
