import asyncio
import websockets
import concurrent.futures
import json
import requests
import pandas as pd
import telegram
from datetime import datetime, timedelta
import time


# 토큰 관련 변수
access_token = ""
token_update_time = None
token_valid_time = timedelta(hours=24)
token_refresh_interval = timedelta(hours=6)

your_chat_id = "322233222"
if 1 == 0:  # real
    g_appkey = "PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC"
    g_appsceret = "KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM="

if 1 == 1:  # demo
    g_appkey = "PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF"
    g_appsceret = "rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s="


# 토큰 발급 함수
def get_access_token():
    global access_token, token_update_time, g_appkey, g_appsceret

    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    data = {
        "grant_type": "client_credentials",
        'appkey': g_appkey,
        'appsecret': g_appsceret,
    }
    headers = {"content-type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        access_token = response.json()["access_token"]
        token_update_time = datetime.now()
        print("액세스 토큰 발급 완료")
    else:
        print("액세스 토큰 발급 실패")
        print("응답 코드:", response.status_code)
        print("응답 내용:", response.text)

# 웹소켓 접속키 발급
def get_approval():
    global approval_key, g_appkey, g_appsceret

    url = 'https://openapivts.koreainvestment.com:29443' # 모의투자계좌
    # url = 'https://openapi.koreainvestment.com:9443'  # 실전투자계좌
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials",
            "appkey": g_appkey,
            "secretkey": g_appsceret}
    PATH = "oauth2/Approval"
    URL = f"{url}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    approval_key = res.json()["approval_key"]

    return approval_key


# 토큰 갱신 함수
async def refresh_token():
    while True:
        if access_token == "" or datetime.now() >= token_update_time + token_refresh_interval:
            get_access_token()
        await asyncio.sleep(60)  # 60초마다 토큰 갱신 체크


# 접속 정보
account = "60025978"
code = "105V03"  # 종목코드
qty = 1  # 주문수량
bns = "02"  # 매수

# 텔레그램 봇 정보
chat_token = "5030631557:AAGFTf-C0XDWCViU3pOtLea5qSdiNSxDL7g"
bot = telegram.Bot(token=chat_token)

# 주문 관련 딕셔너리
orders = {}
orders_che = {}

# 실시간 데이터 변수
now_price = 0
bid_price_1 = 0  # 매수호가1
ask_price_1 = 0  # 매도호가1

# 미체결 주문 확인 및 정정 주문 함수
async def check_unexecuted_orders():
    global your_chat_id
    global g_appkey, g_appsceret

    while True:
        try:
            # 주문 조회
            url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/inquire-ccnl"
            payload = {
                "CANO": account,
                "ACNT_PRDT_CD": "03",
                "STRT_ORD_DT": "20240312",#datetime.now().strftime("%Y%m%d"),
                "END_ORD_DT": "20240312", #datetime.now().strftime("%Y%m%d"),
                "SLL_BUY_DVSN_CD": "00",
                "CCLD_NCCS_DVSN": "00",
                "SORT_SQN": "DS",
                "STRT_ODNO": "",
                "PDNO": code,
                "MKET_ID_CD": "00",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": ""
            }
            headers = {
                'content-type': 'application/json',
                'authorization': 'Bearer ' + str(access_token),
                'appkey': g_appkey,
                'appsecret': g_appsceret,
                'tr_id': 'VTTO5201R'
            }

            response = requests.request("GET", url, headers=headers, params=payload)

            response_data = response.json()

            if response_data['rt_cd'] == '0':  # 정상 응답인 경우에만 처리

                # output1 필드의 데이터를 DataFrame으로 변환
                df = pd.DataFrame(response_data['output1'])
                # 원하는 열만 선택하여 표시
                columns_to_display = ['odno', 'ord_tmd', 'trad_dvsn_name', 'ord_qty', 'ord_idx', 'tot_ccld_qty']
                df_selected = df.loc[:, columns_to_display]
                # 열 이름을 한글로 변경
                df_selected.columns = ['주문번호', '주문시각', '거래구분', '주문수량', '주문가격', '체결수량']
                # 주문가격을 소수점 둘째 자리까지 표시
                pd.options.display.float_format = '{:.2f}'.format
                df_selected.loc[:, '주문가격'] = df_selected['주문가격'].astype(float)
                # 열의 너비 조정
                pd.set_option('display.max_colwidth', None)
                pd.set_option('display.width', None)

                if 1== 0:
                    for order in response_data['output']:
                        ord_no = order['odno']  # 주문번호
                        ord_qty = int(order['ord_qty'])  # 주문수량
                        ccld_qty = int(order['tot_ccld_qty'])  # 체결수량

                        if ord_no not in orders_che and ord_qty > ccld_qty:  # 미체결 수량이 있는 경우
                            print(f"미체결 주문 발견 - 주문번호: {ord_no}, 미체결수량: {ord_qty - ccld_qty}")

                            # 정정 주문
                            ord_qty = ord_qty - ccld_qty
                            prc = ask_price
                            OrdNo_org = ord_no

                            payload = json.dumps({
                                "ORD_PRCS_DVSN_CD": "02",
                                "CANO": account,
                                "ACNT_PRDT_CD": "03",
                                "RVSE_CNCL_DVSN_CD": "01",
                                "ORGN_ODNO": OrdNo_org,
                                "ORD_QTY": str(ord_qty),
                                "UNIT_PRICE": str(prc),
                                "NMPR_TYPE_CD": "01",
                                "KRX_NMPR_CNDT_CD": "0",
                                "RMN_QTY_YN": "Y",
                                "ORD_DVSN_CD": "01"
                            })
                            url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order-rvsecncl"
                            response = requests.post(url, headers=headers, data=payload)
                            print(f"정정 주문 요청 - 원주문번호: {OrdNo_org}, 정정수량: {ord_qty}, 정정가격: {prc}")
                            bot.sendMessage(chat_id=your_chat_id, text=f"정정 주문 요청 - 원주문번호: {OrdNo_org}, 정정수량: {ord_qty}, 정정가격: {prc}")
            else:  # 에러 응답인 경우
                print(f"주문 조회 실패: {response_data}")
                bot.sendMessage(chat_id=your_chat_id, text=f"주문 조회 실패: {response_data}")

            await asyncio.sleep(1)  # 1초 간격으로 미체결 확인

        except Exception as e:
            print(f"에러 발생: {e}")
            bot.sendMessage(chat_id=your_chat_id, text=f"에러 발생: {e}")

async def connect_websocket():
    uri = "ws://ops.koreainvestment.com:21000"

    async def connect_websocket():
        uri = "ws://ops.koreainvestment.com:21000"

        async with websockets.connect(uri) as websocket:
            code_list = [['1', 'H0IFASP0', code], ['1', 'H0IFCNT0', code], ['1', 'H0IFCNI0', 'jika79']]

            for c in code_list:
                tr_id = c[1]
                tr_key = c[2]
                input_data = json.dumps({"header": {"authorization": f"Bearer {approval_key}", "appkey": g_appkey, "tr_type": "1", "custtype": "P", "appsecret": g_appsceret},
                                         "body": {"input": {"tr_id": tr_id, "tr_key": tr_key}}})
                await websocket.send(input_data)
                print(f"Sent: {input_data}")

        while True:
            data = await websocket.recv()
            print(f"Received: {data}")
            on_data(data)  # 데이터 처리 함수 호출

# 체결 데이터 처리 함수
def on_data(data):
    global now_price, volume, bid_price, ask_price, your_chat_id

    try:
        if data[0] == '0':
            recvstr = data.split('|')
            trid0 = recvstr[1]

            if trid0 == "H0IFASP0":  # 지수선물호가
                print("#### 지수선물호가 ####")
                stockhoka_futs(recvstr[3])
                time.sleep(0.2)  # asyncio.sleep 대신 time.sleep 사용

            elif trid0 == "H0IFCNT0":  # 지수선물체결
                print("#### 지수선물체결 ####")
                data_cnt = int(recvstr[2])  # 체결데이터 개수
                stockspurchase_futs(data_cnt, recvstr[3])
                time.sleep(0.2)  # asyncio.sleep 대신 time.sleep 사용

        elif data[0] == '1':
            recvstr = data.split('|')
            trid0 = recvstr[1]

            if trid0 == "H0IFCNI0":  # 지수선물 체결통보
                aes_dec_str = aes_cbc_base64_dec(aes_key, aes_iv, recvstr[3])
                pValue = aes_dec_str.split('^')

                ord_no = pValue[2]  # 주문번호
                ord_qty = int(pValue[8])  # 체결수량
                ord_prc = float(pValue[9])  # 체결가격

                print(f"체결통보 - 주문번호: {ord_no}, 체결수량: {ord_qty}, 체결가격: {ord_prc}")

                if ord_no not in orders_che:
                    orders_che[ord_no] = (ord_qty, ord_prc, datetime.now())

                if ord_no in orders and ord_qty == orders[ord_no][0]:
                    print(f"{ord_no} 주문이 모두 체결되었습니다.")
                    del orders[ord_no]  # 전량 체결시 딕셔너리에서 삭제

            bot.sendMessage(chat_id=your_chat_id, text=f"체결통보 - 주문번호: {ord_no}, 체결수량: {ord_qty}, 체결가격: {ord_prc}")

    except Exception as e:
        print(f"에러 발생: {e}")
        bot.sendMessage(chat_id=your_chat_id, text=f"에러 발생: {e}")


# 지수선물호가 출력라이브러리
def stockhoka_futs(data):
    # print(data)
    recvvalue = data.split('^')  # 수신데이터를 split '^'

    # 호가 데이터 파싱 및 변수 업데이트
    ask_price = float(recvvalue[2])  # 선물매도호가1
    bid_price = float(recvvalue[7])  # 선물매수호가1

    print("지수선물  [" + recvvalue[0] + "]")
    print("영업시간  [" + recvvalue[1] + "]")
    print("====================================")
    print("선물매도호가1 [" + recvvalue[2] + "]" + ",    매도호가건수1    [" + recvvalue[12] + "]" + ",    매도호가잔량1   [" + recvvalue[22] + "]")
    print("선물매도호가2 [" + recvvalue[3] + "]" + ",    매도호가건수2    [" + recvvalue[13] + "]" + ",    매도호가잔량2   [" + recvvalue[23] + "]")
    print("선물매도호가3 [" + recvvalue[4] + "]" + ",    매도호가건수3    [" + recvvalue[14] + "]" + ",    매도호가잔량3   [" + recvvalue[24] + "]")
    print("선물매도호가4 [" + recvvalue[5] + "]" + ",    매도호가건수4    [" + recvvalue[15] + "]" + ",    매도호가잔량4   [" + recvvalue[25] + "]")
    print("선물매도호가5 [" + recvvalue[6] + "]" + ",    매도호가건수5    [" + recvvalue[16] + "]" + ",    매도호가잔량5   [" + recvvalue[26] + "]")
    print("선물매수호가1 [" + recvvalue[7] + "]" + ",    매수호가건수1    [" + recvvalue[17] + "]" + ",    매수호가잔량1   [" + recvvalue[27] + "]")
    print("선물매수호가2 [" + recvvalue[8] + "]" + ",    매수호가건수2    [" + recvvalue[18] + "]" + ",    매수호가잔량2   [" + recvvalue[28] + "]")
    print("선물매수호가3 [" + recvvalue[9] + "]" + ",    매수호가건수3    [" + recvvalue[19] + "]" + ",    매수호가잔량3   [" + recvvalue[29] + "]")
    print("선물매수호가4 [" + recvvalue[10] + "]" + ",   매수호가건수4    [" + recvvalue[20] + "]" + ",    매수호가잔량4   [" + recvvalue[30] + "]")
    print("선물매수호가5 [" + recvvalue[11] + "]" + ",    매수호가건수5   [" + recvvalue[21] + "]" + ",    매수호가잔량5   [" + recvvalue[31] + "]")
    print("====================================")
    print("총매도호가건수 [" + recvvalue[32] + "]" + ",    총매도호가잔량   [" + recvvalue[34] + "]" + ",    총매도호가잔량증감 [" + recvvalue[36] + "]")
    print("총매수호가건수 [" + recvvalue[33] + "]" + ",    총매수호가잔량   [" + recvvalue[35] + "]" + ",    총매수호가잔량증감 [" + recvvalue[37] + "]")


# 지수선물체결처리 출력라이브러리
def stockspurchase_futs(data_cnt, data):
    print("============================================")
    print(data)
    menulist = "선물단축종목코드|영업시간|선물전일대비|전일대비부호|선물전일대비율|선물현재가|선물시가|선물최고가|선물최저가|최종거래량|누적거래량|누적거래대금|HTS이론가|시장베이시스|괴리율|근월물약정가|원월물약정가|스프레드|미결제약정수량|미결제약정수량증감|시가시간|시가대비현재가부호|시가대비지수현재가|최고가시간|최고가대비현재가부호|최고가대비지수현재가|최저가시간|최저가대비현재가부호|최저가대비지수현재가|매수비율|체결강도|괴리도|미결제약정직전수량증감|이론베이시스|선물매도호가|선물매수호가|매도호가잔량|매수호가잔량|매도체결건수|매수체결건수|순매수체결건수|총매도수량|총매수수량|총매도호가잔량|총매수호가잔량|전일거래량대비등락율|협의대량거래량|실시간상한가|실시간하한가|실시간가격제한구분"
    menustr = menulist.split('|')
    pValue = data.split('^')
    print(pValue)
    i = 0
    for cnt in range(data_cnt):  # 넘겨받은 체결데이터 개수만큼 print 한다
        print("### [%d / %d]" % (cnt + 1, data_cnt))
        for menu in menustr:
            print("%-13s[%s]" % (menu, pValue[i]))
            i += 1

# 주문 함수
def order():
    global now_price, bid_price_1, ask_price_1, your_chat_id, orders

    tick = 0.05
    prc = now_price - tick * 5  # 현재가 기준 5틱 낮은 가격에 매수 주문

    payload = json.dumps({
        "ORD_PRCS_DVSN_CD": "02",
        "CANO": account,
        "ACNT_PRDT_CD": "03",
        "SLL_BUY_DVSN_CD": bns,
        "SHTN_PDNO": code,
        "ORD_QTY": str(qty),
        "UNIT_PRICE": str(prc),
        "NMPR_TYPE_CD": "01",
        "KRX_NMPR_CNDT_CD": "0",
        "ORD_DVSN_CD": "01"
    })
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order"
    headers = {
        'content-type': 'application/json',
        'authorization': f'Bearer {access_token}',
        'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
        'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
        'tr_id': 'VTTO1101U'
    }
    response = requests.post(url, headers=headers, data=payload)
    response_data = response.json()

    if response_data['rt_cd'] == '0':
        ord_no = response_data['output']['ORD_NO']
        print(f"신규 주문 요청 - 주문번호: {ord_no}, 주문수량: {qty}, 주문가격: {prc}")
        orders[ord_no] = (qty, prc, datetime.now())
        bot.sendMessage(chat_id=your_chat_id, text=f"신규 주문 요청 - 주문번호: {ord_no}, 주문수량: {qty}, 주문가격: {prc}")
    else:
        print(f"주문 실패: {response_data}")
        bot.sendMessage(chat_id=your_chat_id, text=f"주문 실패: {response_data}")

# 비동기 태스크 실행
async def main():
    # get_access_token()  # 초기 토큰 발급
    # get_approval()

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await asyncio.gather(
            connect_websocket(),
            check_unexecuted_orders(),
            loop.run_in_executor(pool, order),  # 주문 함수를 별도 쓰레드에서 실행
            refresh_token()  # 토큰 갱신 태스크 추가
        )


# 이벤트 루프 실행
get_access_token()  # 초기 토큰 발급
get_approval()
asyncio.run(main())

# get_access_token()