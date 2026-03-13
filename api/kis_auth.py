import os
import requests
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class KISAuth:
    def __init__(self):
        # 보안 정보 로드
        self.app_key = os.getenv("KIS_APP_KEY")
        self.app_secret = os.getenv("KIS_APP_SECRET")
        self.base_url = "https://openapi.koreainvestment.com:9443"  # 실전매매용
        # 모의투자용일 경우 주소: https://openapivts.koreainvestment.com:29443
        
        self._token = None

    def get_token(self):
        """한국투자증권 Access Token 발급"""
        path = "/oauth2/tokenP"
        url = f"{self.base_url}{path}"
        
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        res = requests.post(url, headers=headers, data=json.dumps(data))
        
        if res.status_code == 200:
            self._token = res.json()["access_token"]
            print("✅ Access Token 발급 성공!")
            return self._token
        else:
            print(f"❌ Token 발급 실패: {res.text}")
            return None

    @property
    def auth_headers(self):
        """모든 API 요청에 공통으로 들어갈 헤더 구성"""
        if not self._token:
            self.get_token()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
