import requests
import json

# access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6ImQ5MGI0YTg4LTFiNDktNDFhOC05YWMwLTJhY2MxOTA2MWJjOSIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEwMjk1Mzk5LCJpYXQiOjE3MTAyMDg5OTksImp0aSI6IlBTTUlENk1vbHpTY25YMHNjUjlXQjdnWlVLM2N4cnVhNEZ3RiJ9.z2UewVeaLluJ9kTL8NGRuvkQwLynvUHxl3KGbdgoQum9gPs0aH5rs853bFRfE5xaCljyu4nMdKUEiEOt5d5FsQ"


############################################
# token delete

if 1==0:
    url = "https://openapivts.koreainvestment.com:29443/oauth2/revokeP"

    payload = json.dumps({
      "appkey": "PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF",
      "appsecret": "rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=",
      # "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6ImMwYTYzZjQzLTk2MjgtNDRlOC1hYmI4LTY4YmE5OGVlODU4NiIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEwMjE0MzA1LCJpYXQiOjE3MTAxMjc5MDUsImp0aSI6IlBTTUlENk1vbHpTY25YMHNjUjlXQjdnWlVLM2N4cnVhNEZ3RiJ9.JXUnikyjI_JsZp51ZPuerQvgUsGpIPLMPiZJ6ZKIEcAqd132Hz5PRUCBfVVwHBX0U_sUTvrEHPyTu3Ad-3Bm2g"
      "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6ImQ5MGI0YTg4LTFiNDktNDFhOC05YWMwLTJhY2MxOTA2MWJjOSIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEwMjk1Mzk5LCJpYXQiOjE3MTAyMDg5OTksImp0aSI6IlBTTUlENk1vbHpTY25YMHNjUjlXQjdnWlVLM2N4cnVhNEZ3RiJ9.z2UewVeaLluJ9kTL8NGRuvkQwLynvUHxl3KGbdgoQum9gPs0aH5rs853bFRfE5xaCljyu4nMdKUEiEOt5d5FsQ"

    })
    headers = {
      'content-type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


############################################
# token issue

if 1==0:

    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"

    payload = json.dumps({
        "grant_type": "client_credentials",
        "appkey": "PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF",
        "appsecret": "rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s="
    })
    headers = {
        'content-type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # 응답 확인
    if response.status_code == 200:
      access_token = response.json()["access_token"]
      print("액세스 토큰:", access_token)
    else:
      print("액세스 토큰 발급 실패")
      print("응답 코드:", response.status_code)
      print("응답 내용:", response.text)

############################################

# # CANO: 모의계좌번호(주식)
# # ACNT_PRDT_CD: 계좌상품코드 (01: 주식)
# # PDNO: 상품번호 (주식 종목코드)
# # ORD_DVSN: 주문구분 (00: 지정가)
# # ORD_QTY: 주문수량
# # ORD_UNPR: 주문단가
#
#

url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/trading/order-cash"

payload = json.dumps({
  "CANO": "50105720",
  "ACNT_PRDT_CD": "01",
  "PDNO": "005930",
  "ORD_DVSN": "00",
  "ORD_QTY": "1",
  "ORD_UNPR": "73500"
})
headers = {
  'content-type': 'application/json',
  'authorization': 'Bearer '+ str(access_token),
  'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
  'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
  'tr_id': 'VTTC0802U',
  'hashkey': ''
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
