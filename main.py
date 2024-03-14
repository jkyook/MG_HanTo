# -*- coding: utf-8 -*-
import asyncio
import aiohttp
import websockets
import json
import logging
import requests
from datetime import datetime, timedelta
import telegram
from config import api_key, secret_key, telegram_token, chat_id, account, code, qty

import sys
import os
import time
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
import tkinter as tk
# import tkFileDialog  # //Python 2.7
from tkinter import filedialog  # //Python 3
# # python 2.7py
# import Tkinter as Tk
# from tkFileDialog import askopenfilename

import matplotlib.pyplot as plt
import MyWindow2

import sqlite3
import scipy.stats as stat
import copy
import NProb

# from twilio.rest import Client


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 토큰 관련 변수
access_token = ""
token_update_time = None
token_valid_time = timedelta(hours=24)
token_refresh_interval = timedelta(hours=6)

# 주문 관련 딕셔너리
orders = {}
orders_che = {}
ord_sent = 0

# 텔레그램 봇 정보
bot = telegram.Bot(token=telegram_token)

# start
# start_ox = 1
# ex.chkForb.setChecked(0)
# Console_Log("Data_Receive..Start", 1)
text1 = " *** (MG7) 시스템 가동 시작 ***, Start == 0"
try:
    if bot_alive == 1:
        bot.sendMessage(chat_id="322233222", text=text1)
except:
    pass
msg_last = "ok"
# print("Dome_ab", dome_ab)
last_time = time.time()
cvolume_mid = 0
count_mid = 1
OrgMain = "n"
nfset = 0
inp = 0
inp_o = 0
stock = 0
AvePrc = 0
circulation = 0
print("jump to NP")
NP = NProb.Nprob()
print("NP..Laoded")
# Console_Log("Advising Data", 1)

# 변수 설정
nf = 0
last_volume = 0
ExedQty = 0

# 비동기 HTTP 요청 클라이언트 생성
async def create_session():
    return aiohttp.ClientSession()

access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6IjgwZGVjMjlmLTFkYzQtNGYyOS04YzM4LTA5ZTI0NTdkOTBkZCIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEwMzc0OTc2LCJpYXQiOjE3MTAyODg1NzYsImp0aSI6IlBTTUlENk1vbHpTY25YMHNjUjlXQjdnWlVLM2N4cnVhNEZ3RiJ9.8X-2fr43AAWbrYApF_7Og6ETqeAt_1EbU4SI0XSqpfKNRhy_m8nLBUi1FLaruwxwSh83LudS9Hj4QugAKAW08g"

# 토큰 발급 함수
async def get_access_token(session):
    global access_token, token_update_time

    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    data = {
        "grant_type": "client_credentials",
        "appkey": api_key,
        "appsecret": secret_key
    }

    try:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                access_token = result["access_token"]
                print("access_token: ", access_token)
                token_update_time = datetime.now()
                logger.info("액세스 토큰 발급 완료")
            else:
                logger.error("액세스 토큰 발급 실패")
    except Exception as e:
        logger.error(f"액세스 토큰 발급 중 오류 발생: {e}")

# 웹소켓 접속키 발급
async def get_approval():
    global approval_key

    url = 'https://openapivts.koreainvestment.com:29443' # 모의투자계좌
    # url = 'https://openapi.koreainvestment.com:9443'  # 실전투자계좌
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials",
            "appkey": api_key,
            "secretkey": secret_key}
    PATH = "oauth2/Approval"
    URL = f"{url}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    approval_key = res.json()["approval_key"]

    return approval_key

# 웹소켓 접속 및 데이터 수신
async def connect_websocket(session):
    global ord_sent

    await get_approval()

    url = 'ws://ops.koreainvestment.com:31000' # 모의투자계좌
    # url = 'ws://ops.koreainvestment.com:21000'  # 실전투자계좌

    code_list = [['1','H0IFASP0','101V03'],['1','H0IFCNT0','101V03'], # 지수선물호가, 체결가
                 ['1', 'H0IFASP0', '105V03'], ['1', 'H0IFCNT0', '105V03'],
    #              ['1','H0IOASP0','201T11317'],['1','H0IOCNT0','201T11317'], # 지수옵션호가, 체결가
                 ['1','H0IFCNI0','jika79']] # 선물옵션체결통보

    senddata_list = []

    for i, j, k in code_list:
        temp = '{"header":{"approval_key": "%s","custtype":"P","tr_type":"%s","content-type":"utf-8"},"body":{"input":{"tr_id":"%s","tr_key":"%s"}}}' % (approval_key, i, j, k)
        senddata_list.append(temp)

    async with websockets.connect(url, ping_interval=None) as websocket:

        for senddata in senddata_list:
            await websocket.send(senddata)
            await asyncio.sleep(0.5)
            print(f"Input Command is :{senddata}")

        while True:
            try:
                data = await websocket.recv()
                # logger.info(f"Received data: {data}")

                if data[0] == '0':
                    recvstr = data.split('|')  # 수신데이터가 실데이터 이전은 '|'로 나뉘어져있어 split
                    trid0 = recvstr[1]

                    if trid0 == "H0IFASP0":  # 지수선물호가 tr 일경우의 처리 단계
                        # print("#### 지수선물호가 ####")
                        stockhoka_futs(recvstr[3])
                        # await asyncio.sleep(0.2)
                        pass

                    elif trid0 == "H0IFCNT0":  # 지수선물체결 데이터 처리
                        # print("#### 지수선물체결 ####")
                        data_cnt = int(recvstr[2])  # 체결데이터 개수

                        # npp 계산
                        stockspurchase_futs(data_cnt, recvstr[3]) # price 출력

                        # 주문처리
                        # place_order()

                        # 테스트
                        # if ord_sent == 0:
                        #     print("process_order")
                        #     ord_sent = 1
                        # await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Error while receiving data from websocket: {e}")

#####################################################################

# # 주문 처리 함수
# async def place_order():
#     global npp
#
#     if npp ==  4:
#         await send_order(bns = "02")
#     if npp ==  -4:
#         await send_order(bns = "01")


#####################################################################

# 주문 처리 함수
async def send_order(bns):
    global orders, ord_sent, api_key, secret_key

    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order"

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
    "ORD_QTY": str(qty),
    "UNIT_PRICE": str(price),
    "NMPR_TYPE_CD": "01",
    "KRX_NMPR_CNDT_CD": "0",
    "ORD_DVSN_CD": "01"
    })
    headers = {
    'content-type': 'application/json',
    'authorization': 'Bearer ' + str(access_token),
    'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
    'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
    # "appkey": api_key,
    # "secretkey": secret_key,
    'tr_id': 'VTTO1101U',
    'hashkey': ''
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    result = response.json()
    if result["rt_cd"] == "0":
        ord_no = result["output"]["ODNO"]
        logger.info(f"주문 요청 완료 - 주문번호: {ord_no}")
        orders[ord_no] = (qty, price, datetime.now())
        bot.sendMessage(chat_id=chat_id, text=f"신규 주문 요청 - 주문번호: {ord_no}, 주문수량: {qty}, 주문가격: {str(price)}")
        ord_sent = 1
    else:
        logger.error("주문 요청 실패")

# 미체결 주문 확인 및 처리
async def check_unexecuted_orders(session):
    global access_token

    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/inquire-ccnl"

    payload = {
        "CANO": account,
        "ACNT_PRDT_CD": "03",
        "STRT_ORD_DT": "20240313",
        "END_ORD_DT": "20240313",
        "SLL_BUY_DVSN_CD": "00",
        "CCLD_NCCS_DVSN": "00",
        "SORT_SQN": "DS",
        "STRT_ODNO": "",
        "PDNO": "",
        "MKET_ID_CD": "00",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }
    headers = {
        'content-type': 'application/json',
        'authorization': 'Bearer ' + str(access_token),
        'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
        'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
        'tr_id': 'VTTO5201R'
    }

    while True:
        try:
            # async with session.get(url, headers=headers, params=payload) as response:
            response = requests.request("GET", url, headers=headers, params=payload)

            data = response.json()
            # print(data)

            df = pd.DataFrame(data["output1"])
            # filtered_df = df[df['ord_qty'].astype(int) >= 1]
            filtered_df = df[df['tot_ccld_qty'] != df['ord_qty']][['odno', 'ord_tmd', 'trad_dvsn_name', 'ord_qty', 'ord_idx']]
            print(filtered_df)

        except Exception as e:
            logger.error(f"미체결 주문 확인 중 오류 발생: {e}")

        await asyncio.sleep(1)  # 1초 간격으로 미체결 주문 확인

#####################################################################

# 토큰 갱신 함수
async def refresh_token(session):
    while True:
        try:
            if token_update_time is None or datetime.now() >= token_update_time + token_refresh_interval:
                await get_access_token(session)
        except Exception as e:
            logger.error(f"토큰 갱신 중 오류 발생: {e}")

        await asyncio.sleep(60)  # 60초마다 토큰 갱신 체크


# npp read
async def check():
    global price

    now = datetime.now()

    # NP에서 데이터 입수 기록
    f1 = open("npp.txt", 'r')
    f1_r = f1.readline()
    np1, prf1 = f1_r.strip().split(',')
    print("NP1 (-):", np1, prf1)
    f1.close()

    # (cover_b) -> cover_ordered = 1/0
    f2 = open("npp_.txt", 'r')
    f2_r = f2.readline()
    np2, prf2, chkForb = f2_r.strip().split(',')
    print("NP2 (+):", np2, prf2, chkForb)
    f2.close()

    if np1 == "":
        np1 = 0
    if np2 == "":
        np2 = 0
    np_qty = int(np1) + int(np2)

    tick = 0.02
    b_prc = str(float(now_prc) + tick * 1)
    s_prc = str(float(now_prc) - tick * 1)
    ord_type = "1"

    if cover_ordered1 == 0:
        if np1 == "1":
            send_order(bns = "02")
            cover_ordered1 = 1
    elif cover_ordered1 == 1:
        if np1 == "0":
            send_order(bns = "01")
            cover_ordered1 = 0

    if cover_ordered2 == 0:
        if np2 == "-1":
            send_order(bns = "01")
            cover_ordered2 = -1
    elif cover_ordered2 == 1:
        if np2 == "0":
            send_order(bns = "02")
            cover_ordered1 = 0

#####################################################################
# 지수선물호가 출력라이브러리
def stockhoka_futs(data):
    global lblSqty2v, lblSqty1v,lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v

    # print(data)
    recvvalue = data.split('^')  # 수신데이터를 split '^'
    if recvvalue[0] == "101V03":

        lblSqty2v = float(recvvalue[23])
        lblSqty1v = float(recvvalue[22])
        lblShoga1v =float(recvvalue[2])

        lblBqty2v = float(recvvalue[28])
        lblBqty1v = float(recvvalue[27])
        lblBhoga1v = float(recvvalue[7])

        # if 1==1:
        # print("지수선물  [" + recvvalue[0] + "]")
        # print("영업시간  [" + recvvalue[1] + "]")
        # print("====================================")
        # print("선물매도호가1	[" + recvvalue[2] + "]" + ",    매도호가건수1	[" + recvvalue[12] + "]" + ",    매도호가잔량1	[" + recvvalue[22] + "]")
        # print("선물매도호가2	[" + recvvalue[3] + "]" + ",    매도호가건수2	[" + recvvalue[13] + "]" + ",    매도호가잔량2	[" + recvvalue[23] + "]")
        # print("선물매도호가3	[" + recvvalue[4] + "]" + ",    매도호가건수3	[" + recvvalue[14] + "]" + ",    매도호가잔량3	[" + recvvalue[24] + "]")
        # print("선물매도호가4	[" + recvvalue[5] + "]" + ",    매도호가건수4	[" + recvvalue[15] + "]" + ",    매도호가잔량4	[" + recvvalue[25] + "]")
        # print("선물매도호가5	[" + recvvalue[6] + "]" + ",    매도호가건수5	[" + recvvalue[16] + "]" + ",    매도호가잔량5	[" + recvvalue[26] + "]")
        # print("선물매수호가1	[" + recvvalue[7] + "]" + ",    매수호가건수1	[" + recvvalue[17] + "]" + ",    매수호가잔량1	[" + recvvalue[27] + "]")
        # print("선물매수호가2	[" + recvvalue[8] + "]" + ",    매수호가건수2	[" + recvvalue[18] + "]" + ",    매수호가잔량2	[" + recvvalue[28] + "]")
        # print("선물매수호가3	[" + recvvalue[9] + "]" + ",    매수호가건수3	[" + recvvalue[19] + "]" + ",    매수호가잔량3	[" + recvvalue[29] + "]")
        # print("선물매수호가4	[" + recvvalue[10] + "]" + ",   매수호가건수4	[" + recvvalue[20] + "]" + ",    매수호가잔량4	[" + recvvalue[30] + "]")
        # print("선물매수호가5	[" + recvvalue[11] + "]" + ",    매수호가건수5	[" + recvvalue[21] + "]" + ",    매수호가잔량5	[" + recvvalue[31] + "]")
        # print("====================================")
        # print("총매도호가건수	[" + recvvalue[32] + "]" + ",    총매도호가잔량	[" + recvvalue[34] + "]" + ",    총매도호가잔량증감	[" + recvvalue[36] + "]")
        # print("총매수호가건수	[" + recvvalue[33] + "]" + ",    총매수호가잔량	[" + recvvalue[35] + "]" + ",    총매수호가잔량증감	[" + recvvalue[37] + "]")


# 지수선물체결처리 출력라이브러리
def stockspurchase_futs(data_cnt, data):
    global price, volume, cvolume, cgubun, count, npp
    global last_time, last_volume, cgubun_sum, cvolume_mid, cvolume_sum, count_mid, nf, ExedQty
    global lblSqty2v, lblSqty1v, lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1

    # print("============================================")
    menulist = "선물단축종목코드|영업시간|선물전일대비|전일대비부호|선물전일대비율|선물현재가|선물시가|선물최고가|선물최저가|최종거래량|누적거래량|누적거래대금|HTS이론가|시장베이시스|괴리율|근월물약정가|원월물약정가|스프레드|미결제약정수량|미결제약정수량증감|시가시간|시가대비현재가부호|시가대비지수현재가|최고가시간|최고가대비현재가부호|최고가대비지수현재가|최저가시간|최저가대비현재가부호|최저가대비지수현재가|매수비율|체결강도|괴리도|미결제약정직전수량증감|이론베이시스|선물매도호가|선물매수호가|매도호가잔량|매수호가잔량|매도체결건수|매수체결건수|순매수체결건수|총매도수량|총매수수량|총매도호가잔량|총매수호가잔량|전일거래량대비등락율|협의대량거래량|실시간상한가|실시간하한가|실시간가격제한구분"
    menustr = menulist.split('|')
    pValue = data.split('^')
    print("pValue[0]: ", pValue[0])

    if pValue[0] == "105V03":
        # 현재가
        prc_o1 = float(pValue[5])

    elif pValue[0] == "101V03":
        # 현재가
        price = float(pValue[5])
        volume = pValue[10]
        if last_volume == 0:
            last_volume = int(volume) + 1
        cvolume = int(volume) - int(last_volume)  #pValue[9]
        last_volume = volume
        cgubun = pValue[5]
        if pValue[5] == pValue[35]:
            cgubun = "Buy"
        else:
            cgubun = "Sell"

        # print(price, cgubun, cvolume, pValue[9], volume)

    if 1==1:
        #####################################
        # NProb
        #####################################

        t1 = time.time()
        mt = t1 - last_time
        timestamp = int(t1 * 1000)

        # nprob at under 0.5
        if mt < 0.5:
            if cgubun == "Buy":
                cvolume_mid += cvolume
            else:
                cvolume_mid += cvolume * -1
            count_mid += 1
        else:
            print("                                                   ")
            print("                                                   ")
            # if cgubun == "Buy":
            #     cvolume = cvolume
            if cgubun == "Sell":
                cvolume = cvolume * -1
            cvolume_sum = cvolume_mid + cvolume
            if cvolume_sum > 0:
                cgubun_sum = "Buy"
            else:
                cgubun_sum = "Sell"
            count = count_mid + 1
            mt = mt / count

        print(price, timestamp, mt, count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v,
                       lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1)

            ##########
            # nProb()
            ##########
        nf += 1
        npp = NP.nprob(price, timestamp, mt, count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v,
                       lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1)

        if NP.auto_cover == 1:
            f = open("npp_.txt", 'w')
            # s = ex.chkForb.isChecked()
            f.write(str(NP.cover_ordered) + "," + str(NP.profit_opt))# + "," + str(s))
            f.close()
        if NP.auto_cover == 2:
            f = open("npp.txt", 'w')
            f.write(str(NP.cover_ordered) + "," + str(NP.profit_opt))
            f.close()

        print('NP: ', npp)
        print("[sys] ****** np.exed : ", NP.exed_qty)
        print("[sys] Exed: ", ExedQty)

        cvolume_mid = 0
        cvolume_sum = 0
        count_mid = 0
        last_time = t1
        elap = (time.time() - t1) * 1000
        print("elap: ", elap)
        # ex.txtElap.setText(str("%0.1f" % elap))



    # # 시장 체결데이터 출력
    # if 1==0:
    #     i = 0
    #     for cnt in range(data_cnt):  # 넘겨받은 체결데이터 개수만큼 print 한다
    #         print("### [%d / %d]" % (cnt + 1, data_cnt))
    #         for menu in menustr:
    #             print("%-13s[%s]" % (menu, pValue[i]))
    #             i += 1
#####################################################################


async def main():
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            connect_websocket(session),
            # check_unexecuted_orders(session),
            # refresh_token(session)
        )


# 프로그램 실행
if __name__ == "__main__":
    asyncio.run(main())