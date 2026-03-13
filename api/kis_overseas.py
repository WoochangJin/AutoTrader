import requests
import json

class KISOverseas:
    def __init__(self, auth):
        self.auth = auth
        self.base_url = auth.base_url

    def get_overseas_daily_ohlcv(self, symbol, end_date, excd="NAS"):
        """
        미국 주식 일봉 데이터 조회 (HHDFS00000300)
        """
        path = "/uapi/overseas-price/v1/quotations/dailyprice" # 하이픈 확인
        url = f"{self.base_url}{path}"
        
        headers = self.auth.auth_headers
        headers.update({
            "tr_id": "HHDFS76240000" 
        })
        
        params = {
            "AUTH": "",
            "EXCD": excd, 
            "SYMB": symbol,
            "GUBN": "0", 
            "BYMD": end_date,
            "MODP": "1"
        }
        
        res = requests.get(url, headers=headers, params=params)
        
        print(f"DEBUG: 상태 코드 = {res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            # 서버가 보낸 전체 데이터를 일단 다 찍어봅니다.
            print(f"DEBUG: 서버 응답 전체 데이터 -> {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # 데이터가 output2에 있는지, 아니면 다른 곳에 있는지 확인 루프
            return data.get('output2', [])
        else:
            print(f"❌ 시세 조회 실패 상세: {res.status_code} - {res.text}")
            return None
