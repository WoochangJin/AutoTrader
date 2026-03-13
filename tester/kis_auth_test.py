# tester/kis_auth_test.py
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from api.kis_auth import KISAuth

def test_connection():
    print("--- 한국투자증권 연결 테스트 시작 ---")
    try:
        # 객체를 생성하는 순간 토큰 발급 로직이 실행됩니다.
        auth = KISAuth()
        
        if auth.access_token:
            print(f"✅ 테스트 성공! 발급된 토큰: {auth.access_token[:10]}...")
        else:
            print("❌ 테스트 실패: 토큰이 비어있습니다.")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    test_connection()
