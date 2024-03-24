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
import NProb2

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
unexecuted_orders = {}
df_npp_m = pd.DataFrame()
ord_sent = 0

# 텔레그램 봇 정보
bot = telegram.Bot(token=telegram_token)
text1 = " *** (MG7) 시스템 가동 시작 ***, Start == 0"
try:
    if bot_alive == 1:
        bot.sendMessage(chat_id="322233222", text=text1)
except:
    pass

nf = 0
last_volume = 0
ExedQty = 0
msg_out = ""
bot_alive = 1
sub = 0
auto_time = 1
msg_last = "ok"
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
chkForb = 0
np_count = 0
cum_qty = 0

print("jump to NP")
NP = NProb.Nprob()
NP2 = NProb2.Nprob()
print("NP..Laoded")

# code = "105V04"
# account = "60025978"
# qty =1

# 비동기 HTTP 요청 클라이언트 생성
async def create_session():
    return aiohttp.ClientSession()

access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6IjQ1MmE4MWRjLTg3NTYtNDU2OC1hMGI4LTM5NGVmYjI5ODBmZSIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEwODk4NzE0LCJpYXQiOjE3MTA4MTIzMTQsImp0aSI6IlBTTUlENk1vbHpTY25YMHNjUjlXQjdnWlVLM2N4cnVhNEZ3RiJ9.hk35jZCGQTnjNDhEXdm_vA59TdF-Rqhj8jwQHdoNA1YMep23g9l7eAmzlOB9nX0O7eYM9tYL34NaR_0fZKxxyg"


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
                with open('token_update_time.txt', 'w') as f:
                    f.write(token_update_time.strftime('%Y-%m-%d %H:%M:%S'))
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
    global ord_sent, npp

    await get_approval()

    url = 'ws://ops.koreainvestment.com:31000' # 모의투자계좌
    # url = 'ws://ops.koreainvestment.com:21000'  # 실전투자계좌

    code_list = [['1','H0IFASP0','101V06'],['1','H0IFCNT0','101V06'], # 지수선물호가, 체결가
                 ['1', 'H0IFASP0', '105V04'], ['1', 'H0IFCNT0', '105V04'],
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

                        # # npp 계산
                        # stockspurchase_futs(data_cnt, recvstr[3]) # price 출력
                        # npp 계산
                        asyncio.create_task(stockspurchase_futs(data_cnt, recvstr[3]))  # price 출력

                        # 테스트
                        # if ord_sent == 0:
                        #     print("process_order")
                        #     ord_sent = 1
                        # await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Error while receiving data from websocket: {e}")


# 토큰 갱신 함수
async def refresh_token(session):
    global token_update_time

    try:
        # 파일에서 읽어들이기
        with open('token_update_time.txt', 'r') as f:
            token_update_time_str = f.read().strip()

        # 문자열을 datetime 객체로 변환
        token_update_time = datetime.strptime(token_update_time_str, '%Y-%m-%d %H:%M:%S')
    except:
        pass

    while True:
        try:
            if token_update_time is None or datetime.now() >= token_update_time + token_refresh_interval:
                await get_access_token(session)
        except Exception as e:
            logger.error(f"토큰 갱신 중 오류 발생: {e}")

        await asyncio.sleep(60)  # 60초마다 토큰 갱신 체크


# order from file
async def file_check():
    global price, now_prc

    now = datetime.now()

    # NP에서 데이터 입수 기록
    file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp_.txt'
    f1 = open(file_path, 'r')
    f1_r = f1.readline()
    np1, prf1 = f1_r.strip().split(',')
    print("NP1 (-):", np1, prf1)
    f1.close()

    # (cover_b) -> cover_ordered = 1/0
    # file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo_2/npp.txt'
    file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp.txt'
    f2 = open(file_path, 'r')
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

    # if cover_ordered1 == 0:
    #     if np1 == "1":
    #         send_order(bns = "02")
    #         cover_ordered1 = 1
    # elif cover_ordered1 == 1:
    #     if np1 == "0":
    #         send_order(bns = "01")
    #         cover_ordered1 = 0
    #
    # if cover_ordered2 == 0:
    #     if np2 == "-1":
    #         send_order(bns = "01")
    #         cover_ordered2 = -1
    # elif cover_ordered2 == 1:
    #     if np2 == "0":
    #         send_order(bns = "02")
    #         cover_ordered1 = 0

#####################################################################
# 지수선물호가 출력라이브러리
def stockhoka_futs(data):
    global lblSqty2v, lblSqty1v,lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v

    # print(data)
    recvvalue = data.split('^')  # 수신데이터를 split '^'
    if recvvalue[0] == "101V06":

        lblSqty2v = float(recvvalue[23])
        lblSqty1v = float(recvvalue[22])
        lblShoga1v =float(recvvalue[2])

        lblBqty2v = float(recvvalue[28])
        lblBqty1v = float(recvvalue[27])
        lblBhoga1v = float(recvvalue[7])

#####################################################################

# 지수선물체결처리 출력라이브러리
async def stockspurchase_futs(data_cnt, data):
    global price, volume, cvolume, cgubun, count, npp, npp2
    global last_time, last_volume, cgubun_sum, cvolume_mid, cvolume_sum, count_mid, nf, ExedQty
    global lblSqty2v, lblSqty1v, lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1

    # print("============================================")
    menulist = "선물단축종목코드|영업시간|선물전일대비|전일대비부호|선물전일대비율|선물현재가|선물시가|선물최고가|선물최저가|최종거래량|누적거래량|누적거래대금|HTS이론가|시장베이시스|괴리율|근월물약정가|원월물약정가|스프레드|미결제약정수량|미결제약정수량증감|시가시간|시가대비현재가부호|시가대비지수현재가|최고가시간|최고가대비현재가부호|최고가대비지수현재가|최저가시간|최저가대비현재가부호|최저가대비지수현재가|매수비율|체결강도|괴리도|미결제약정직전수량증감|이론베이시스|선물매도호가|선물매수호가|매도호가잔량|매수호가잔량|매도체결건수|매수체결건수|순매수체결건수|총매도수량|총매수수량|총매도호가잔량|총매수호가잔량|전일거래량대비등락율|협의대량거래량|실시간상한가|실시간하한가|실시간가격제한구분"
    menustr = menulist.split('|')
    pValue = data.split('^')
    # print("code: ", pValue[0])

    if pValue[0] == "105V04":
        # 현재가
        prc_o1 = float(pValue[5])

    elif pValue[0] == "101V06":
        # 현재가
        price = float(pValue[5])
        volume = pValue[10]
        if last_volume == 0:
            last_volume = int(volume) + 1
        cvolume = int(volume) - int(last_volume)  #pValue[9]
        last_volume = volume
        cgubun = pValue[5]
        if pValue[5] == pValue[35]:
            cgubun = "Sell"
        else:
            cgubun = "Buy"

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
            # print(price, f"{mt:.2f}", count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v,
            #                lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1)

            ##########
            # npp
            ##########

            nf += 1
            npp = NP.nprob(price, timestamp, mt, count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v,
                           lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1)
            npp2 = NP2.nprob(price, timestamp, mt, count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v,
                           lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1)



            # 기록
            if NP.auto_cover == 1:
                file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp_.txt'
                f = open(file_path, 'w')
                s = chkForb #ex.chkForb.isChecked()
                f.write(str(NP.cover_ordered) + "," + str(NP.profit_opt) + "," + str(s))
                f.close()
            # if NP.auto_cover == 2:
            #     file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo_2/npp.txt'
            #     f = open(file_path, 'w')
            #     f.write(str(NP.cover_ordered) + "," + str(NP.profit_opt))
            #     f.close()

            if NP2.auto_cover == 2:
                file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp.txt'
                f = open(file_path, 'w')
                f.write(str(NP2.cover_ordered) + "," + str(NP2.profit_opt))
                f.close()

            ##########
            # 주문처리
            ##########
            asyncio.create_task(place_order())

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

# 주문 처리 함수
async def place_order():
    global npp, npp2, nf, np_count, cum_qty

    if nf == 10:
        await send_order(bns = "02")

    if npp ==  4:
        await send_order(bns = "02")
    elif npp ==  -4:
        await send_order(bns = "01")

    if npp2 ==  4:
        await send_order(bns = "02")
    elif npp2 ==  -4:
        await send_order(bns = "01")

    # 기록
    try:
        file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp_.txt'
        f1 = open(file_path, 'r')
        f1_r = f1.readline()
        np1, prf1, chkForb = f1_r.strip().split(',')
        NP.np1 = int(np1)
        print("NP1 (-):", np1, prf1)
        f1.close()

        # (cover_b) -> cover_ordered = 1/0
        # file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo_2/npp.txt'
        file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp.txt'
        f2 = open(file_path, 'r')
        f2_r = f2.readline()
        np2, prf2 = f2_r.strip().split(',')
        NP.np2 = int(np2)
        print("NP2 (+):", np2, prf2)
        f2.close()

        if np1 == "":
            np1 = 0
        if np2 == "":
            np2 = 0
        np_qty = int(np1) + int(np2)

        now = datetime.now()

        print("********")
        print(np_count, np_qty, int(cum_qty))
        print("********")
        ##
        np_count += 1
        df_npp_m.at[np_count, 'np1'] = int(np1)
        df_npp_m.at[np_count, 'np2'] = int(np2)
        df_npp_m.at[np_count, 'np_sum'] = np_qty
        df_npp_m.at[np_count, 'real_sum'] = int(cum_qty)
        df_npp_m.at[np_count, 'now_prc'] = price
        df_npp_m.at[np_count, 'prf1'] = float(prf1)
        df_npp_m.at[np_count, 'prf2'] = float(prf2)
        df_npp_m.at[np_count, 'prf'] = float(prf1) + float(prf2)
        df_npp_m.at[np_count, 'time'] = str(now.hour) + str(now.minute) + str(now.second)
        # df_npp_m에 차이값 계산 및 저장
        df_npp_m['difference'] = (df_npp_m['np_sum'] - df_npp_m['real_sum']).abs()

        if np_count % 1000 == 0 and NP.auto_cover == 1:
            ts = datetime.now().strftime("%m-%d-%H-%M")
            filename = "(e)df_npp_%s.csv" % (ts)
            if NP.which_market == 4:
                filename = "(e4)df_npp_%s.csv" % (ts)
            df_npp_m.to_csv('%s' % filename)  # + time.strftime("%m-%d") + '.csv')
    except:
        pass

    print('NP: ', npp)
    print("[sys] ****** np.exed : ", NP.exed_qty)
    print("[sys] Exed: ", ExedQty)

#####################################################################

async def send_order(bns):
    global orders, ord_sent, api_key, secret_key, price, qty, code, account, chkForb, auto_time, prc_o1

    if auto_time == 1:
        url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order"

        payload = json.dumps({
            "ORD_PRCS_DVSN_CD": "02",
            "CANO": account,
            "ACNT_PRDT_CD": "03",
            "SLL_BUY_DVSN_CD": bns,
            "SHTN_PDNO": code,
            "ORD_QTY": str(qty),
            "UNIT_PRICE": str(prc_o1),
            "NMPR_TYPE_CD": "01",
            "KRX_NMPR_CNDT_CD": "0",
            "ORD_DVSN_CD": "01"
        })

        headers = {
            'content-type': 'application/json',
            'authorization': 'Bearer ' + str(access_token),
            'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
            'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
            'tr_id': 'VTTO1101U',
            'hashkey': ''
        }

        if chkForb != 1:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload) as response:
                    result = await response.json()
                    if result["rt_cd"] == "0":
                        ord_no = result["output"]["ODNO"]
                        logger.info(f"주문 요청 완료 - 주문번호: {ord_no}")
                        orders[ord_no] = (bns, qty, price, prc_o1, datetime.now().strftime("%H:%M"))
                        bot.sendMessage(chat_id=chat_id, text=f"신규 주문 요청 - 주문번호: {ord_no}, 구분: {bns}, 주문가격: {str(prc_o1)}, 주문수량: {qty}")
                        ord_sent = 1
                    else:
                        logger.error("주문 요청 실패")

#####################################################################

# 미체결 주문 확인 및 처리
async def check_unexecuted_orders(session):
    global access_token
    global unexecuted_orders, orders, orders_che

    # 시스템상
    for ord_no, order_info in orders.items():
        if ord_no not in orders_che:
            unexecuted_orders[ord_no] = order_info

    print(unexecuted_orders)

    # 조회
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

# 선물옵션 체결통보 출력라이브러리
def stocksigningnotice_futsoptn(data, key, iv):
    global orders, orders_che, price, qty, code, account, cum_qty

    # AES256 처리 단계
    aes_dec_str = aes_cbc_base64_dec(key, iv, data)
    # print(aes_dec_str)
    pValue = aes_dec_str.split('^')
    # print(pValue)

    if pValue[6] == '0':  # 체결통보
        print("#### 지수선물옵션 체결 통보 ####")
        menulist_sign = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|체결수량|체결단가|체결시간|거부여부|체결여부|접수여부|지점번호|주문수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
        menustr = menulist_sign.split('|')
        bns_che = pValue[4]  # 매도매수구분

        qty_che = pValue[8]  # 체결수량
        if bns_che == "매도":
            qty_che = int(qty_che) * -1
        cum_qty += qty_che

        ord_no_che = pValue[3]  # 원주문번호
        prc_o1_che = pValue[9]  # 체결단가
        time_che = pValue[10]  # 체결시간
        orders_che[ord_no_che] = (bns_che, qty_che, price, prc_o1_che, time_che)
        i = 0
        for menu in menustr:
            print("%s  [%s]" % (menu, pValue[i]))
            i += 1

        print(orders_che)


    else:  # pValue[6] == 'L', 주문·정정·취소·거부 접수 통보

        if pValue[5] == '1':  # 정정 접수 통보 (정정구분이 1일 경우)
            print("#### 지수선물옵션 정정 접수 통보 ####")
            menulist_revise = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|정정수량|정정단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
            menustr = menulist_revise.split('|')
            i = 0
            for menu in menustr:
                print("%s  [%s]" % (menu, pValue[i]))
                i += 1

        elif pValue[5] == '2':  # 취소 접수 통보 (정정구분이 2일 경우)
            print("#### 지수선물옵션 취소 접수 통보 ####")
            menulist_cancel = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|취소수량|주문단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
            menustr = menulist_cancel.split('|')
            i = 0
            for menu in menustr:
                print("%s  [%s]" % (menu, pValue[i]))
                i += 1

        elif pValue[11] == '1':  # 거부 접수 통보 (거부여부가 1일 경우)
            print("#### 지수선물옵션 거부 접수 통보 ####")
            menulist_refuse = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|주문수량|주문단가|주문시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
            menustr = menulist_refuse.split('|')
            i = 0
            for menu in menustr:
                print("%s  [%s]" % (menu, pValue[i]))
                i += 1

        else:  # 주문 접수 통보
            print("#### 지수선물옵션 주문 접수 통보 ####")
            menulist_order = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|주문수량|체결단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
            menustr = menulist_order.split('|')
            i = 0
            for menu in menustr:
                print("%s  [%s]" % (menu, pValue[i]))
                i += 1


#####################################################################

async def msg():
    global bot, df, nf, df1, msg_now, msg_last, msg_last_sent, bot_alive, OrgOrdNo
    global isblocked_msg, isreleased_msg, istimeblocked_msg, stat_out_org, stat_in_org, OrgOrdNo_Cov
    global started, sub, orders, NP, msg_out, auto_time, chkForb

    while True:
        if NP.auto_cover != 0:

            # chat_token = "5269168004:AAEVzu9b7QBc3EBGDlQXum6abOCFxKEVmbg"
            chat_token = "5030631557:AAGFTf-C0XDWCViU3pOtLea5qSdiNSxDL7g" # bot2

            url = "https://api.telegram.org/bot{}/getUpdates".format(chat_token)
            response = requests.get(url)
            updates = response.json()
            updates = updates["result"][-5:]
            messages = [str(update['message']['text']) for update in updates if 'message' in update]

            try:
                msg_now = messages[-1]
            except:
                msg_now = msg_last
            # print("msg_now: ", msg_now)

            a = ["sym", "qty", "Amt", "entPrc", "nowPrc", "prcDif", "prf"]
            c = ["coin", "nowPrc", "prf"]

            now = datetime.now()

            msgs = ["block", "stop", "x", "start", "now", "last", "out", "mout", "time", "qty", "cover", "orders", "orders_che", "tr", "plot", "stat", "statset", "ord", "cordnum", "cexed", "plotset", "reord", "list", "delord", "deleted", "auto"]

            try:

                if msg_now != msg_last and msg_now != msg_out and (msg_now in msgs or msg_now[:3] in msgs) :# and msg_last[0] == "1":# and msg_now != "shut":  # and msg_now != "last":
                    if bot_alive == 1 or bot_alive == 2:
                        bot.sendMessage(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) + " (now): " + msg_now + ", (last) : " + msg_last)

                    msg_last = msg_now

                    if nf > 100 and (msg_now == "stop" or msg_now == "x" or msg_now == "000"):
                        if NP.cover_ordered != 0:
                            text = str(NP.auto_cover) + " = (한투) (MG7) 커버 진입상태임..확인필요 ="
                            chkForb = 1
                            if bot_alive == 1 or bot_alive == 2:
                                bot.sendMessage(chat_id="322233222", text=text)
                        if NP.cover_ordered == 0:
                            chkForb = 1
                            text = str(NP.auto_cover) + " = (한투) (MG7) 매매를 중단합니다. ="
                            if bot_alive == 1 or bot_alive == 2:
                                bot.sendMessage(chat_id="322233222", text=text)

                        if chkForb == 1 and isblocked_msg == 0:
                            text = str(NP.auto_cover) + " = (한투) (MG7) 매매가 중단된 상태입니다. ="
                            if bot_alive == 1 or bot_alive == 2:
                                bot.sendMessage(chat_id="322233222", text=text)
                            isblocked_msg = 1
                            isreleased_msg = 0

                    if msg_now == "auto":
                        if auto_time == 0:
                            auto_time = 1
                        elif auto_time == 1:
                            auto_time = 0
                        bot.sendMessage(chat_id="322233222", text="(한투)" + str(auto_time) + ", 0:release, 1:set")

                    if (msg_now == "start" or msg_now == "111") and msg_out != "start":
                        chkForb = 0
                        text = "(한투)" + str(NP.auto_cover) + " = (한투) (MG7) 매매 중단을 해제합니다. ="
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                        if chkForb == 0 and isreleased_msg == 0:
                            text = "(한투)" + str(NP.auto_cover) + " = (한투) (MG7) 매매가 시작되었습니다. ="
                            if bot_alive == 1 or bot_alive == 2:
                                bot.sendMessage(chat_id="322233222", text=text)
                            isreleased_msg = 1
                            isblocked_msg = 0
                            istimeblocked_msg = 0

                    if msg_now[:3] == "out":
                        msg_out = msg_now[4:]
                        bot.sendMessage(chat_id="322233222", text= "(한투)" + "아래 명령어 제외 :" + msg_now[4:])

                    if msg_now == "mout":
                        bot.sendMessage(chat_id="322233222", text= "(한투)" + "제외된 명령어 :" + msg_out)

                    if msg_now == "cover":
                        text = "(한투)" + str(NP.auto_cover) + " (한투) cover_ordered: " + str(NP.cover_ordered) + ",  exed: " + str(NP.cover_order_exed)
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "qty":
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text="(한투)" + str(cum_qty))

                    if msg_now == "orders":
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) + " " + str(orders))

                    if msg_now == "orders_che":
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) + " " + str(orders_che))

                    if msg_now == "now":
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) +" " + msg_now)

                    if msg_now == "last":
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) +" " + msg_last)

                    if msg_now == "stat":
                        text = "(한투)" + str(NP.auto_cover) + "stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + "(sub) stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "statset":
                        stat_in_org = [0,0,0]
                        stat_out_org = [1,1,1]
                        text = "(한투)" + str(NP.auto_cover) + "stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + "(sub) stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "ord":
                        text = "(한투)" + str(NP.auto_cover) + "Ord: " + str(OrgOrdNo)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) Ord: " + str(OrgOrdNo)
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "cordnum":
                        text = "(한투)" + str(NP.auto_cover) + "Ord_Cov: " + str(OrgOrdNo_Cov)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) Ord_Cov: " + str(OrgOrdNo_Cov)
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "corded":
                        text = "(한투)" + str(NP.auto_cover) + "Ord_Ordered: " + str(NP.cover_ordered)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) Ord_Ordered: " + str(NP.cover_ordered)
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "cexed":
                        text = "(한투)" + str(NP.auto_cover) + "cover_order_exed: " + str(NP.cover_order_exed)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) cover_order_exed: " + str(NP.cover_order_exed)
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    # if msg_now == "reord":
                    #     text = "(한투)" + str(NP.auto_cover) + "reOrdered: " + str(reordered) + "  reOrdExed: " + str(reordered_exed)
                    #     if sub == 1:
                    #         text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) reOrdered: " + str(reordered) + "  reOrdExed: " + str(reordered_exed)
                    #     if bot_alive == 1 or bot_alive == 2:
                    #         bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "block":
                        if chkForb == 0:
                            text = "(한투)" + str(NP.auto_cover) + " Block: " + str("Released")
                            if sub == 1:
                                text = "(한투)" + str(NP.auto_cover) + " (sub) Block: " + str("Released")
                        if chkForb == 1:
                            text = "(한투)" + str(NP.auto_cover) + " Block: " + str("Blocked")
                            if sub == 1:
                                text = "(한투)" + str(NP.auto_cover) + " (sub) Block: " + str("Released")
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text=text)

                    if msg_now == "list":
                        # text = " [block, stop, start, time, cover, plot, stat, statset, ord, cordnum, cexed, plotset, reord] "
                        if bot_alive == 1 or bot_alive == 2:
                            bot.sendMessage(chat_id="322233222", text="(한투)" + ", ".join(msgs))
            except:
                if msg_now != "no var in list":
                    bot.sendMessage(chat_id="322233222", text="no var in list")


            if msg_now != msg_last:
                if msg_now[:2] == "np":
                    # want_see = msg_now[2]
                    if bot_alive == 1 or bot_alive == 2:
                        bot.sendMessage(chat_id="322233222", text="(한투)" + "np :" + str(NP.msg_now[3:]))

                if msg_now[:6] == "delord":
                    del orders[int(msg_now[6:])]
                    bot.sendMessage(chat_id="322233222", text="(한투)" + "deleted : " + str(int(msg_now[6:])))

                if msg_now[:7] == "deleted":
                    orders = []
                    bot.sendMessage(chat_id="322233222", text="(한투)" + "dumped : " + str(NP.auto_cover))


            if nf % 500 == 0 and nf != 0:
                if chkForb == 0:
                    text = str(NP.auto_cover) + " = (한투) (MG7) In Released Status=" + str(nf)
                    if sub == 1:
                        text = str(NP.auto_cover) + "(한투, sub) = (한투) (MG7) In Released Status=" + str(nf)
                    if bot_alive == 1 or bot_alive == 2:
                        bot.sendMessage(chat_id="322233222", text=text)
                if chkForb == 1:
                    text = str(NP.auto_cover) + " = (한투) (MG7) In Blocked Status =" + str(nf)
                    if sub == 1:
                        text = str(NP.auto_cover) + " =(한투, sub) (MG7) In Blocked Status =" + str(nf)
                    if bot_alive == 1 or bot_alive == 2:
                        bot.sendMessage(chat_id="322233222", text=text)


        await asyncio.sleep(5)

def aes_cbc_base64_dec(key, iv, cipher_text):
    """
    :param key:  str type AES256 secret key value
    :param iv: str type AES256 Initialize Vector
    :param cipher_text: Base64 encoded AES256 str
    :return: Base64-AES256 decodec str
    """
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    return bytes.decode(unpad(cipher.decrypt(b64decode(cipher_text)), AES.block_size))


#####################################################################

async def main():
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            connect_websocket(session),
            asyncio.create_task(msg()),
            # check_unexecuted_orders(session),
            # file_check(),
            refresh_token(session)
        )

#####################################################################

# 프로그램 실행
if __name__ == "__main__":
    asyncio.run(main())