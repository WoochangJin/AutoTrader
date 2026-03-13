import os
import json
import time
import requests
from dotenv import load_dotenv

# 1. 절대 경로 설정을 통해 프로젝트 어디서든 .env를 찾을 수 있게 합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # api 폴더의 부모인 QuantTrader 폴더
env_path = os.path.join(root_dir, '.env')

# .env 로드
load_dotenv(env_path)

class KISAuth:
    def __init__(self):
        # 환경 변수 로드
        self.app_key = os.getenv("KIS_APP_KEY")
        self.app_secret = os.getenv("KIS_APP_SECRET")
        self.base_url = os.getenv("KIS_BASE_URL")

# --- 여기서부터 추가 ---
        print(f"DEBUG: APP_KEY 길이 = {len(self.app_key) if self.app_key else 0}")
        print(f"DEBUG: APP_SECRET 길이 = {len(self.app_secret) if self.app_secret else 0}")
        if self.app_secret:
            print(f"DEBUG: SECRET 마지막 문자 = '{self.app_secret[-1]}'") # 마지막이 따옴표인지 확인
# --- 여기까지 추가 ---
        
        # .token.json 저장 경로 설정
        self.token_file = os.path.join(root_dir, ".token.json")
        
        # 기본 URL이 제대로 로드되지 않았을 경우를 대비한 안전장치
        if not self.base_url:
            raise ValueError(f"❌ 설정 오류: {env_path}에서 KIS_BASE_URL을 찾을 수 없습니다.")
            
        self.access_token = self._get_valid_token()

    def _get_valid_token(self):
        """기존 토큰이 유효하면 재사용하고, 아니면 새로 발급합니다."""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r") as f:
                    token_data = json.load(f)
                
                # 만료 시간 체크 (현재 시간보다 1시간 정도 여유를 둡니다)
                if time.time() < token_data.get("expire_time", 0):
                    print("✅ 기존 토큰 재사용 중...")
                    return token_data["access_token"]
            except (json.JSONDecodeError, KeyError):
                # 파일이 깨졌거나 형식이 다르면 새로 발급 시도
                pass

        return self._issue_new_token()

    def _issue_new_token(self):
        """한국투자증권 서버에 새 토큰 발급을 요청합니다."""
        print("🌐 새 Access Token 발급 시도 중...")
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        
        # 수정 전: "secretkey": self.app_secret
        # 수정 후: "appsecret": self.app_secret
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret  # <--- 이 부분!!
        }
        
        res = requests.post(url, headers=headers, json=data)
        
        if res.status_code == 200:
            res_data = res.json()
            token = res_data["access_token"]
            
            # 만료 시간 계산 (현재 시간 + expires_in 초 - 3600초 여유)
            expire_time = time.time() + int(res_data["expires_in"]) - 3600
            
            # 파일에 저장
            with open(self.token_file, "w") as f:
                json.dump({
                    "access_token": token, 
                    "expire_time": expire_time
                }, f)
            
            print("✅ 새 Access Token 발급 및 저장 완료!")
            return token
        else:
            # 호출 제한(1분 1회) 등에 걸렸을 때 구체적인 에러 메시지 출력
            print(f"❌ 토큰 발급 실패: {res.text}")
            return None

    @property
    def auth_headers(self):
        """다른 API 호출 시 사용할 공통 헤더"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
