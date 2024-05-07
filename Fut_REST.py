import requests
import json
import pandas as pd


# account = "60025978"
account = "64154012" # real
code = "105V05"
qty = "1"
prc = "371.2"
OrdNo_org = "1401"
bns = "02" # 01:sell, 02:buy
mode = 1 # 1:new_ord, 2:reord/cancel, 3:che, 4:unexed

# access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6Ijc1N2NjNDZkLWJlNTktNGU1OS05MGFlLTFmMTBmNDU1NTY4MCIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEyOTY3MjEzLCJpYXQiOjE3MTI4ODA4MTMsImp0aSI6IlBTcFJWc0tTTllqZE9UbXZjclBOMEMwTXl1cUVaQmFleTJBQyJ9.IKmyzWgbcAlefsraDfKnSUMl7fIG0oWYcmgPgp5D6cbDPllomPwCCYJhrNadD1qXpF3dvudkh-_te1JEU69dYw"
access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6IjJmNGFmYTM2LTJkYTktNDM5NC1hYWNlLThjMGQyNzY0MGUxMCIsInByZHRfY2QiOiIiLCJpc3MiOiJ1bm9ndyIsImV4cCI6MTcxNDM2NzIwNSwiaWF0IjoxNzE0MjgwODA1LCJqdGkiOiJQU3BSVnNLU05ZamRPVG12Y3JQTjBDME15dXFFWkJhZXkyQUMifQ.u9NOR0WlwnH7jPa3ySOjmsT4I-hUYbQfk8uGAbu9-1grQrQAi8sJhsscbea1rgzJnKfqmg8dymqOm9lLS6bo9g"
#########################################################
# 액세스 토큰 발급 요청 URL #
#########################################################

if 1==1:

  # global access_token

  url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"

  # # 요청 바디 설정 (demo)
  # data = {
  #     "grant_type": "client_credentials",
  #   'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
  #   'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s='
  # }

  # 요청 바디 설정 (real)
  data = {
      "grant_type": "client_credentials",
  'appkey': 'PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC',
  'appsecret': 'KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=',
  }

  # 요청 헤더 설정
  headers = {
      "content-type": "application/json"
  }

  # POST 요청 보내기
  response = requests.post(url, data=json.dumps(data), headers=headers)

  # 응답 확인
  if response.status_code == 200:
      access_token = response.json()["access_token"]
      print("액세스 토큰:", access_token)
  else:
      print("액세스 토큰 발급 실패")
      print("응답 코드:", response.status_code)
      print("응답 내용:", response.text)

#########################################################
# 신규주문
#########################################################

if mode == 1:

  # url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order"
  url = "https://openapi.koreainvestment.com:9443/uapi/domestic-futureoption/v1/trading/order"

  # ORD_PRCS_DVSN_CD: 주문처리구분코드(02: 실전투자)
  # CANO: 계좌번호
  # ACNT_PRDT_CD: 계좌상품코드(03: 선물옵션)
  # SLL_BUY_DVSN_CD: 매도매수구분코드(02: 매수)
  # SHTN_PDNO: 단축상품번호(선물옵션
  # 종목번호)
  # ORD_QTY: 주문수량
  # UNIT_PRICE: 주문가격
  # NMPR_TYPE_CD: 호가유형코드(01: 지정가)
  # KRX_NMPR_CNDT_CD: KRX호가조건코드(0: 없음)
  # ORD_DVSN_CD: 주문구분코드(01: 일반주문)

  payload = json.dumps({
    "ORD_PRCS_DVSN_CD": "02",
    "CANO": account,
    "ACNT_PRDT_CD": "03",
    "SLL_BUY_DVSN_CD": bns,
    "SHTN_PDNO": code,
    "ORD_QTY": qty,
    "UNIT_PRICE": prc,
    "NMPR_TYPE_CD": "01",
    "KRX_NMPR_CNDT_CD": "0",
    "ORD_DVSN_CD": "01"
  })

  # # demo
  # headers = {
  #   'content-type': 'application/json',
  #   'authorization': 'Bearer ' + str(access_token),
  #   'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
  #   'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
  #   'tr_id': 'VTTO1101U',
  #   'hashkey': ''
  # }

  # real
  headers = {
    'content-type': 'application/json',
    'authorization': 'Bearer ' + str(access_token),
    'appkey': 'PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC',
    'appsecret': 'KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=',
    'tr_id': 'TTTO1101U', #'TTTO1101U',
    'hashkey': ''
  }

  response = requests.request("POST", url, headers=headers, data=payload)

  print(response.text)


#########################################################
# 정정/취소
#########################################################

if mode == 2:

  # url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order-rvsecncl" # demo
  url = "https://openapi.koreainvestment.com:9443/uapi/domestic-futureoption/v1/trading/order-rvsecncl" # real

  # ORD_PRCS_DVSN_CD: 주문처리구분코드(02: 실전투자)
  # CANO: 계좌번호
  # ACNT_PRDT_CD: 계좌상품코드(03: 선물옵션)
  # RVSE_CNCL_DVSN_CD: 정정취소구분코드(02: 취소)
  # ORGN_ODNO: 원주문번호
  # ORD_QTY: 주문수량
  # UNIT_PRICE: 주문가격
  # NMPR_TYPE_CD: 호가유형코드(01: 지정가)
  # KRX_NMPR_CNDT_CD: KRX호가조건코드(0: 없음)
  # RMN_QTY_YN: 잔량처리여부(Y: 잔량전부)
  # ORD_DVSN_CD: 주문구분코드(01: 일반주문)

  payload = json.dumps({
    "ORD_PRCS_DVSN_CD": "02",
    "CANO": account,
    "ACNT_PRDT_CD": "03",
    "RVSE_CNCL_DVSN_CD": "01",
    "ORGN_ODNO": OrdNo_org,
    "ORD_QTY": qty,
    "UNIT_PRICE": prc,
    "NMPR_TYPE_CD": "01",
    "KRX_NMPR_CNDT_CD": "0",
    "RMN_QTY_YN": "Y",
    "ORD_DVSN_CD": "01"
  })

  # if real_demo == 0:
  #     tr_id = 'VTTO1103U'
  # if real_demo == 1:
  #   tr_id = 'TTTO1103U'

  headers = {
    'Content-Type': 'application/json',
    'authorization': 'Bearer '+ str(access_token),
    # 'appKey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
    # 'appSecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
    'appKey': "PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC",
    'appSecret': "KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=",
    'tr_id': 'TTTO1103U',
    'hashkey': ''
  }

  response = requests.request("POST", url, headers=headers, data=payload)

  print(response.text)


#########################################################
# 정정/취소
#########################################################

if mode == 3:

  url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/inquire-ccnl"

  payload = {
    "CANO": account,
    "ACNT_PRDT_CD": "03",
    "STRT_ORD_DT": "20240312",
    "END_ORD_DT": "20240312",
    "SLL_BUY_DVSN_CD": "00",
    "CCLD_NCCS_DVSN": "00",
    "SORT_SQN": "DS",
    "STRT_ODNO": "",
    "PDNO": code,
    "MKET_ID_CD": "00",
    "CTX_AREA_FK200": "",
    "CTX_AREA_NK200": ""
  }

    # CANO=60025978&
    # ACNT_PRDT_CD=03
    # STRT_ORD_DT=20220730
    # END_ORD_DT=20220830
    # SLL_BUY_DVSN_CD=00
    # CCLD_NCCS_DVSN=00
    # SORT_SQN=DS
    # STRT_ODNO=
    # PDNO=
    # MKET_ID_CD=00
    # CTX_AREA_FK200=
    # CTX_AREA_NK200=

  headers = {
      'content-type': 'application/json',
      'authorization': 'Bearer ' + str(access_token),
      'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
      'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
      'tr_id': 'VTTO5201R'
  }

  response = requests.request("GET", url, headers=headers, params=payload)

  # 응답 데이터
  response_data = response.json()

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

  # 표 출력
  print(df_selected)


#########################################################
# 미체결
#########################################################

if mode == 4:

  # url = "https://openapivts.koreainvestment.com:9443/uapi/domestic-futureoption/v1/trading/inquire-ccnl"
  url = "https://openapi.koreainvestment.com:9443/uapi/domestic-futureoption/v1/trading/inquire-ccnl"

  payload = {
    "CANO": account,
    "ACNT_PRDT_CD": "03",
    "STRT_ORD_DT": "20240412",
    "END_ORD_DT": "20240412",
    "SLL_BUY_DVSN_CD": "00",
    "CCLD_NCCS_DVSN": "00",
    "SORT_SQN": "DS",
    "STRT_ODNO": "",
    "PDNO": "",
    "MKET_ID_CD": "00",
    "CTX_AREA_FK200": "",
    "CTX_AREA_NK200": ""
  }
  # headers = {
  #   'content-type': 'application/json',
  #   'authorization': 'Bearer '+ str(access_token),
  #   'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
  #   'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
  #   'tr_id': 'VTTO5201R'
  # }

  headers = {
    'content-type': 'application/json',
    'authorization': 'Bearer ' + access_token,  # 'Bearer ' + str(access_token),
    'appkey': "PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC",
    'appsecret': "KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=",
    'tr_id': 'TTTO5201R'  # VTTO5201R
  }

  # response = requests.request("GET", url, headers=headers, data=payload)
  response = requests.request("GET", url, headers=headers, params=payload)
  # response = requests.request("GET", url, headers=headers, params=payload)

  # print("response: ", response)
  # print(response.json())

  data = response.json()
  print("data ", data)
  # 데이터 프레임 생성
  df = pd.DataFrame(data["output1"])

  print(df)

  # # 'qty' 열의 데이터 타입을 정수형으로 변환하여 1 이상인 행만 필터링
  # filtered_df = df[df['ord_qty'].astype(int) >= 1]
  #
  # # 필터링된 데이터 프레임 출력
  # print(filtered_df)

  # print(response.text)
