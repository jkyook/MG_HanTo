import websockets
import asyncio
import json
import requests

# 실시간 데이터 수신 및 주문 실행에 필요한 기본 설정
g_appkey = "여기에_앱키를_입력하세요"
g_appsecret = "여기에_앱시크릿을_입력하세요"
account = "여기에_계좌번호를_입력하세요"
access_token = "여기에_발급받은_액세스_토큰을_입력하세요"


# 액세스 토큰 발급
def get_access_token(appkey, appsecret):
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    data = {
        "grant_type": "client_credentials",
        'appkey': appkey,
        'appsecret': appsecret
    }
    headers = {
        "content-type": "application/json"
    }
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print("Failed to get access token")
        return None


# 주문 실행 함수
def execute_order(order_type, code, qty, prc, OrdNo_org=None):
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order"
    if order_type == "reorder" or order_type == "cancel":
        url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order-rvsecncl"

    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {access_token}'
    }

    payload = json.dumps({
        "ORD_PRCS_DVSN_CD": "02",
        "CANO": account,
        "ACNT_PRDT_CD": "03",
        "SLL_BUY_DVSN_CD": "02",  # 매수 예시
        "SHTN_PDNO": code,
        "ORD_QTY": qty,
        "UNIT_PRICE": prc,
        "NMPR_TYPE_CD": "01",
        "KRX_NMPR_CNDT_CD": "0",
        "ORD_DVSN_CD": "01"
    })

    response = requests.post(url, headers=headers, data=payload)
    print(response.text)


# 실시간 데이터 수신 및 주문 실행
async def connect_and_order():
    url = 'ws://ops.koreainvestment.com:31000'  # 모의투자계좌 WebSocket 주소
    async with websockets.connect(url, ping_interval=None) as websocket:
        # 여기에 웹소켓을 통해 데이터를 수신하고 처리하는 코드 작성
        pass

    # 예시: 실시간 데이터 처리 후 주문 실행
    # execute_order("new", "105V03", "1", "362.70")


# 액세스 토큰 발급 및 WebSocket 연결 시작
if __name__ == "__main__":
    access_token = get_access_token(g_appkey, g_appsecret)
    if access_token:
        asyncio.get_event_loop().run_until_complete(connect_and_order())
    else:
        print("Access token is not available.")
