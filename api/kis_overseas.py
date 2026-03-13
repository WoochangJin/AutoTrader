import requests
import json
import pandas as pd

class KISOverseas:
    def __init__(self, auth):
        self.auth = auth
        self.base_url = auth.base_url

    def get_overseas_daily_ohlcv(self, symbol, end_date):
        """
        미국 주식 일봉 데이터 조회
        """
        path = "/uapi/overseas-stock/v1/quotations/dailyprice"
        url = f"{self.base_url}{path}"
        
        # auth 객체에서 미리 만들어둔 헤더를 가져옵니다.
        headers = self.auth.auth_headers
        headers.update({
            "tr_id": "HHDFS00000300" 
        })
        
        params = {
            "AUTH": "",
            "EXCD": "NAS",  # SOXL은 나스닥(NAS) 종목입니다.
            "SYMB": symbol,
            "GUBN": "0",    # 0: 일봉
            "BYMD": end_date,
            "MODP": "1"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
