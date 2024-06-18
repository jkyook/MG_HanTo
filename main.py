# -*- coding: utf-8 -*-

import sys
import os
import time
import pandas as pd
import numpy as np
import tkinter as tk
import asyncio
import aiohttp
import websockets
import json
import logging
import requests
import sqlite3
import scipy.stats as stat
import copy
from datetime import datetime, timedelta
import telegram
from config import api_key, secret_key, telegram_token, chat_id, account, code, code_ovs, qty
from tkinter import filedialog  # //Python 3
import matplotlib.pyplot as plt
import subprocess  # subprocess 모듈 import 추가
import MyWindow2

from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from qasync import QEventLoop
# from qasync import asyncSlot, QThreadExecutor
# from quamash import QEventLoop

import NProb
import NProb2

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from base64 import b64decode

real_demo = 1  # 0:demo, 1: real
which_market = 1 # 1:kospi, 2:s&p
temp_id = ""
manu_reorder = 0

#####################################################################
def calculate_kospi_mini_futures_code():
    today = datetime.today()

    # 두 번째 주 목요일 계산
    first_day = datetime(today.year, today.month, 1)
    first_thursday = first_day + timedelta(days=(3 - first_day.weekday()) % 7)
    second_thursday = first_thursday + timedelta(days=7)

    if today <= second_thursday:
        target_month = today.month
    else:
        target_month = today.month + 1 if today.month != 12 else 1

    target_year = today.year if today.month != 12 else today.year + 1

    year_map = {2024: 'V', 2025: 'W'}  # Update and expand as necessary
    month_map = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C'}

    year_code = year_map.get(target_year, 'Unknown year')
    month_code = month_map.get(target_month, 'Unknown month')

    product_code = f"105{year_code}0{month_code}"
    return product_code

code = calculate_kospi_mini_futures_code()

print(code)
#####################################################################
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QBrush, QColor

class UnexecutedOrderListText(QTextEdit):
    global gui

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def mouseReleaseEvent(self, event):
        global temp_id

        cursor = self.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            print("선택되었습니다.")
            order_info = self.extract_order_info(selected_text)
            if order_info:
                self.highlight_order(cursor)
                print("선택된 주문 내역:")
                print(f"주문 ID: {order_info['order_id'].zfill(10)}")
                print(f"주문 시간: {order_info['order_time'].replace('@ ', '')}")
                print(f"주문 구분: {order_info['order_type']}")
                print(f"주문 수량: {order_info['order_quantity']}")
                print(f"주문 가격: {order_info['order_price']}")

                temp_id = order_info['order_id'].zfill(10)
                gui.modify_order_button.setStyleSheet("background-color: yellow")

        else:
            super().mouseReleaseEvent(event)

    def extract_order_info(self, selected_text):
        try:
            order_details = selected_text.split(',')
            order_info = {
                'order_id': order_details[0].strip(),
                'order_time': order_details[3].strip(),
                'order_type': order_details[1].strip(),
                'order_quantity': order_details[2].strip(),
                'order_price': order_details[2].strip()
            }
            return order_info
        except:
            return None

    def highlight_order(self, cursor):
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QBrush(QColor("green")))
        cursor.mergeCharFormat(highlight_format)

#####################################################################
class AutoTradeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(500)  # 1초마다 update_gui 호출
        self.buy_sell_selection = "BUY"  # 초기값은 매수
        self.resize(1000, 800)  # 초기 창 크기 설정

    def initUI(self):
        central_widget = QWidget()
        self.setWindowTitle('[KOSPI] Auto Trade System')

        # 계좌 정보 구획
        self.account_group = QGroupBox('계좌 정보 등')
        self.account_layout = QVBoxLayout()
        self.account_num_label = QLabel('계좌번호: ')
        self.login_status_label = QLabel('로그인 상태: ')
        self.elap_label = QLabel('nf, elap: ')
        self.account_layout.addWidget(self.account_num_label)
        self.account_layout.addWidget(self.login_status_label)
        self.account_layout.addWidget(self.elap_label)

        # 주문금지 버튼 추가
        self.order_ban_button = QPushButton('주문가능')
        self.order_ban_button.setStyleSheet("background-color: green;")  # 초기 배경색은 녹색
        self.order_ban_button.clicked.connect(self.toggle_order_ban)  # 버튼 클릭 이벤트 연결
        self.account_layout.addWidget(self.order_ban_button)

        self.account_group.setLayout(self.account_layout)

        # 시세 내역 구획
        self.market_group = QGroupBox('시세 정보')
        self.market_layout = QVBoxLayout()
        self.stock_code_label = QLabel('종목코드: ')
        self.stock_open_label = QLabel('시가: ')
        self.stock_bid_label = QLabel('매수호가 잔량: ')
        self.stock_ask_label = QLabel('매도호가 잔량: ')
        self.market_layout.addWidget(self.stock_code_label)
        self.market_layout.addWidget(self.stock_open_label)
        self.market_layout.addWidget(self.stock_bid_label)
        self.market_layout.addWidget(self.stock_ask_label)
        self.market_group.setLayout(self.market_layout)

        # 그래프 구획
        self.grpah_group = QGroupBox('그래프')
        self.grpah_layout = QHBoxLayout()
        self.open_gui_button = QPushButton('Files')  # GUI 열기 버튼 추가
        self.open_mer_df_button = QPushButton('merge_df')  # GUI 열기 버튼 추가
        self.open_df_button = QPushButton('df')  # GUI 열기 버튼 추가
        self.grpah_layout.addWidget(self.open_gui_button)  # GUI 열기 버튼 추가
        self.grpah_layout.addWidget(self.open_mer_df_button)  # GUI 열기 버튼 추가
        self.grpah_layout.addWidget(self.open_df_button)  # GUI 열기 버튼 추가
        self.grpah_group.setLayout(self.grpah_layout)


        # 주문 입력 구획
        self.order_group = QGroupBox('주문 입력')
        self.order_layout = QVBoxLayout()
        self.order_qty_edit = QLineEdit()
        self.order_price_edit = QLineEdit()
        self.order_button = QPushButton('주문')
        self.buy_button = QPushButton('매수')
        self.sell_button = QPushButton('매도')
        self.modify_order_button = QPushButton('정정주문')  # 정정주문 버튼 추가
        self.order_layout.addWidget(QLabel('주문수량: '))
        self.order_layout.addWidget(self.order_qty_edit)
        self.order_layout.addWidget(QLabel('주문가격: '))
        self.order_layout.addWidget(self.order_price_edit)
        self.order_layout.addWidget(QLabel('구분: '))
        order_selection_layout = QHBoxLayout()
        order_selection_layout.addWidget(self.buy_button)
        order_selection_layout.addWidget(self.sell_button)
        order_selection_layout.addWidget(self.modify_order_button)  # 정정주문 버튼 추가
        self.order_layout.addLayout(order_selection_layout)
        self.order_layout.addWidget(self.order_button)
        self.order_group.setLayout(self.order_layout)

        # 누적수익 구획
        self.profit_group = QGroupBox('수익 등')
        self.profit_layout = QHBoxLayout()
        self.npp_label = QLabel('NPPs: ')
        self.profit_layout.addWidget(self.npp_label)
        self.profit_label = QLabel('누적수익: ')
        self.profit_layout.addWidget(self.profit_label)
        self.profit_group.setLayout(self.profit_layout)
        self.profit_group.setFixedSize(420, 70)  # 누적수익 구획 크기 지정

        # 주문 상태 구획
        self.order_status_group = QGroupBox('주문 상태')
        self.order_status_layout = QVBoxLayout()
        self.cover_ordered_label = QLabel('커버주문: ')
        self.cover_price_label = QLabel('주문가격: ')
        self.cover_time_label = QLabel('주문시각: ')
        self.cover_type_label = QLabel('유형: ')
        self.order_status_layout.addWidget(self.cover_ordered_label)
        self.order_status_layout.addWidget(self.cover_price_label)
        self.order_status_layout.addWidget(self.cover_time_label)
        self.order_status_layout.addWidget(self.cover_type_label)
        self.order_status_group.setLayout(self.order_status_layout)
        self.order_status_group.setFixedSize(420, 180)  # 주문 상태 구획 크기 지정

        # 주문 내역 구획
        self.order_list_group = QGroupBox('주문 내역')
        self.order_list_layout = QVBoxLayout()
        self.order_list_text = QTextEdit()
        self.order_list_layout.addWidget(self.order_list_text)
        self.order_list_group.setLayout(self.order_list_layout)

        # 미체결 주문 내역 구획
        self.unexecuted_order_list_group = QGroupBox('미체결 주문 내역')
        self.unexecuted_order_list_layout = QVBoxLayout()
        # self.unexecuted_order_list_text = QTextEdit()
        # self.unexecuted_order_list_text.setReadOnly(True)

        self.unexecuted_order_list_text = UnexecutedOrderListText()  # UnexecutedOrderListText 클래스 사용

        self.unexecuted_order_list_layout.addWidget(self.unexecuted_order_list_text)
        self.unexecuted_order_list_group.setLayout(self.unexecuted_order_list_layout)

        # 체결 내역 구획
        self.execution_list_group = QGroupBox('체결 내역')
        self.execution_list_layout = QVBoxLayout()
        self.execution_list_text = QTextEdit()
        self.execution_list_layout.addWidget(self.execution_list_text)
        self.execution_list_group.setLayout(self.execution_list_layout)

        # 메인 레이아웃 구성
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.account_group)
        left_layout.addWidget(self.market_group)
        left_layout.addWidget(self.grpah_group)
        left_layout.addWidget(self.profit_group)
        left_layout.addWidget(self.order_group)
        left_layout.addWidget(self.order_status_group)
        main_layout.addLayout(left_layout)

        mid_splitter = QSplitter(Qt.Vertical)
        mid_splitter.addWidget(self.order_list_group)
        mid_splitter.addWidget(self.unexecuted_order_list_group)
        mid_splitter.setStretchFactor(0, 4)
        mid_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(mid_splitter)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.execution_list_group)
        main_layout.addLayout(right_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 버튼 클릭 이벤트 연결
        self.open_gui_button.clicked.connect(self.open_gui)
        # self.open_mer_df_button.clicked.connect(self.open_df(self, mer=1))
        self.open_mer_df_button.clicked.connect(lambda: self.open_df(mer=1))
        # self.open_df_button.clicked.connect(self.open_df(self, mer=0))
        self.open_df_button.clicked.connect(lambda: self.open_df(mer=0))
        self.order_button.clicked.connect(self.place_order)
        self.buy_button.clicked.connect(self.select_buy)
        self.sell_button.clicked.connect(self.select_sell)
        self.modify_order_button.clicked.connect(self.modify_order_btn)  # 정정주문 버튼 클릭 이벤트 연결

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 255, 204))  # 연한 노랑색 설정
        self.setPalette(palette)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def open_gui(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gui_path = os.path.join(script_dir, 'gui.py')
        subprocess.Popen(['python', gui_path])

    def open_df(self, mer = 0):
        # print("mer: ",mer)
        try:
            if NP.nf <= NP.partition_size or mer == 0:
                df1 = NP.df
                df2 = NP2.df
            else:
                NP.merge_partitions()
                df1 = NP.merged_df
                NP2.merge_partitions()
                df2 = NP2.merged_df

            # 2x1 형식으로 그래프 그리기
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            # 첫 번째 그래프
            ax1_price = ax1.twinx()

            ax1.plot(df1['nf'], df1['cover_ordered'], color='red', label='Cover Ordered (NP)')
            ax1.plot(df2['nf'], df2['cover_ordered'], color='orange', label='Cover Ordered (NP2)')
            ax1_price.plot(df1['nf'], df1['price'], color='blue', label='Price')

            ax1.set_xlabel('nf')
            ax1.set_ylabel('Cover Ordered', color='red')
            ax1_price.set_ylabel('Price', color='blue')

            ax1.tick_params(axis='y', labelcolor='red')
            ax1_price.tick_params(axis='y', labelcolor='blue')

            ax1.set_title('Price and Cover Ordered (NP and NP2)')
            ax1.grid(True)
            ax1.legend()

            # 두 번째 그래프
            ax2_price = ax2.twinx()

            ax2.plot(df1['nf'], df1['profit_opt'], color='green', label='Profit Opt (NP)')
            ax2.plot(df2['nf'], df2['profit_opt'], color='lime', label='Profit Opt (NP2)')
            ax2_price.plot(df1['nf'], df1['price'], color='blue', label='Price')

            ax2.set_xlabel('nf')
            ax2.set_ylabel('Profit Opt', color='green')
            ax2_price.set_ylabel('Price', color='blue')

            ax2.tick_params(axis='y', labelcolor='green')
            ax2_price.tick_params(axis='y', labelcolor='blue')

            ax2.set_title('Price and Profit Opt (NP and NP2)')
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error opening NP.df or NP2.df file: {e}")

    def select_buy(self):
        self.buy_sell_selection = "02"  # 매수
        self.buy_button.setStyleSheet("background-color: red")  # 매수 버튼 색상을 빨간색으로 변경
        self.sell_button.setStyleSheet("")  # 매도 버튼 색상을 기본색으로 변경

    def select_sell(self):
        self.buy_sell_selection = "01"  # 매도
        self.sell_button.setStyleSheet("background-color: blue")  # 매도 버튼 색상을 파란색으로 변경
        self.buy_button.setStyleSheet("")  # 매수 버튼 색상을 기본색으로 변경

    def place_order(self):
        asyncio.create_task(send_order(bns=self.buy_sell_selection))  # 주문 요청
        self.order_button.setStyleSheet("background-color: green")  # 주문 버튼 색상을 녹색으로 변경

    def modify_order_btn(self):
        global temp_id, manu_reorder

        try:
            new_price = self.order_price_edit.text()  # 주문가격 입력창에서 새로운 가격 가져오기
        except:
            new_price = prc_o1

        try:
            print("선택된 주문을 정정합니다.", temp_id, new_price)
            manu_reorder = 1
            self.modify_order_button.setStyleSheet("background-color: green")  # 주문 버튼 색상을 녹색으로 변경
            # modify_order(odno=temp_id, ord_qty=str(qty), prc_o1=str(new_price))
        except:
            print("정정주문 오류")

    def toggle_order_ban(self):
        global chkForb

        if chkForb == 0:
            chkForb = 1
            self.order_ban_button.setText('주문금지')  # 버튼 텍스트 변경
            self.order_ban_button.setStyleSheet("background-color: red;")  # 주문금지 시 배경색을 빨간색으로 변경
        else:
            chkForb = 0
            self.order_ban_button.setText('주문가능')  # 버튼 텍스트 변경
            self.order_ban_button.setStyleSheet("background-color: green;")  # 주문가능 시 배경색을 녹색으로 변경

    def update_gui(self):
        global elap, orders, unexecuted_orders

        # 계좌 정보 업데이트
        self.account_num_label.setText(f'계좌번호: {account}')
        self.login_status_label.setText(f'로그인 상태: {access_token != ""}')  # 토큰 존재 여부로 로그인 상태 표시
        self.elap_label.setText(f'nf, elap: {nf}, {elap:.1f}')

        # 시가 등
        if which_market == 1:
            self.stock_code_label.setText(f'종목코드: {code}')
        if which_market == 2:
            self.stock_code_label.setText(f'종목코드: {code_ovs}')
        self.stock_open_label.setText(f'시가: {price}')
        self.stock_bid_label.setText(f'매수호가 잔량: {lblBqty1v}')
        self.stock_ask_label.setText(f'매도호가 잔량: {lblSqty1v}')
        # self.order_list_text.setText(str(orders))  # 주문 내역 표시

        # 주문 상태 업데이트
        self.cover_ordered_label.setText(f'커버주문: {NP.cover_ordered}, {NP2.cover_ordered}')
        self.cover_price_label.setText(f'주문가격: {NP.cover_in_prc}, {NP2.cover_in_prc}')
        self.cover_time_label.setText(f'주문시각: {NP.cover_in_time}, {NP2.cover_in_time}')
        self.cover_type_label.setText(f'유형: {NP.type}, {NP2.type}')

        # 누적수익 업데이트
        self.profit_label.setText(f'누적수익: {NP.profit_opt:.1f}, {NP2.profit_opt:.1f}')
        self.npp_label.setText(f'NPPs: {npp}, {npp2}')

        # # 미체결 내역 업데이트
        # self.unexecuted_order_list_text.setText(str(unexecuted_orders))  # 미체결 내역 표시

        # 미체결 내역 업데이트
        unexecuted_order_text = ""
        for order_no, order_info in unexecuted_orders.items():
            odno, ord_tmd, trad_dvsn_name, ord_qty, ord_idx = order_info
            if trad_dvsn_name[-2:] == '매수':
                unexecuted_order_text += f"{odno[-5:]}, <span style='color:red;'>{trad_dvsn_name}</span>, {ord_idx}, @ {ord_tmd}<br>"
            elif trad_dvsn_name[-2:] == '매도':
                unexecuted_order_text += f"{odno[-5:]}, <span style='color:blue;'>{trad_dvsn_name}</span>, {ord_idx}, @ {ord_tmd}<br>"
            else:
                unexecuted_order_text += f"{odno[-5:]}, {trad_dvsn_name}, {ord_idx}, @ {ord_tmd}<br>"

        self.unexecuted_order_list_text.setHtml(unexecuted_order_text)  # HTML 형식으로 표시

        # 미체결내역 조회  => main에서 처리
        # now = datetime.now()
        # if now.second % 5 == 0:
        #     check_unexecuted_orders('BTC/USDT')

#####################################################################

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
reordered = 0

global price, lblSqty2v, lblSqty1v, lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1

price = 0
lblSqty2v = 0
lblSqty1v = 0
lblShoga1v = 0
lblBqty1v = 0
lblBqty2v = 0
prc_o1 = 0
if which_market == 1:
    slack = 0.02 * 1
if which_market == 2:
    slack = 0.25 * 1

nf = 0
npp = 0
npp2 = 0
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
elap = 0

print("jump to NP")
NP = NProb.Nprob()
NP2 = NProb2.Nprob()
print("NP..Laoded")

# code = "105V04"
# account = "60025978"
# qty =1

#####################################################################
## 텔레그램 봇 정보

bot = telegram.Bot(token=telegram_token)

async def send_messages(chat_id, text):
    try:
        await bot.sendMessage(chat_id=chat_id, text=text)
    except:
        pass

text1 = " *** (MG7) 시스템 가동 시작 ***, Start == 0"
try:
    if bot_alive == 1:
        bot.sendMessage(chat_id="322233222", text=text1)
except:
    pass

# 비동기 HTTP 요청 클라이언트 생성
async def create_session():
    return aiohttp.ClientSession()

access_token_issued = 0
# access_token = ""

# access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6ImRlMGZjYmZlLTU0YzctNDJmMy05Y2VhLTI0ZDM1OTQ0NzRlNCIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEyMjAxNzU5LCJpYXQiOjE3MTIxMTUzNTksImp0aSI6IlBTcFJWc0tTTllqZE9UbXZjclBOMEMwTXl1cUVaQmFleTJBQyJ9.KWWBGo5EfVrxNDpFRhezpoWpTY-RUJoN5-HyHT06V7VCCbAPun-4N6Cm6rEVMBLP1QuMeZ-rZfbBOCj6-G2nGQ"

#####################################################################
# 토큰 갱신 함수

async def refresh_token(session):
    global token_update_time, access_token, access_token_issued

    try:
        # 파일에서 읽어들이기
        with open('token_update_time.txt', 'r') as f:
            token_update_time_str = f.read().strip()

        # 문자열을 datetime 객체로 변환
        token_update_time = datetime.strptime(token_update_time_str, '%Y-%m-%d %H:%M:%S')
    except:
        token_update_time = None

    while True:

        # print("refresh_token....access_token_issued = ",access_token_issued )
        # if access_token_issued == 1:
        #     with open('access_token.txt', 'r') as f:
        #         access_token = f.read().strip()
        #         logger.info("저장된 액세스 토큰 사용(발급직후): %s", access_token)

        try:
            if token_update_time is None or datetime.now() >= token_update_time + token_refresh_interval:
                await get_access_token(session)
            else:
                # 저장된 액세스 토큰 읽어오기
                try:
                    with open('access_token.txt', 'r') as f:
                        access_token = f.read().strip()
                        logger.info("저장된 액세스 토큰 사용: %s", access_token)
                except:
                    await get_access_token(session)
        except Exception as e:
            logger.error(f"토큰 갱신 중 오류 발생: {e}")

        await asyncio.sleep(30*1)  # 60초마다 토큰 갱신 체크

#####################################################################
# 토큰 발급 함수
async def get_access_token(session):
    global access_token, token_update_time, access_token_issued

    if real_demo == 1:
        url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"

        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": api_key,
            "appsecret": secret_key
        }

    if real_demo == 0:
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
                access_token_issued = 1

                # 액세스 토큰을 파일에 저장
                with open('access_token.txt', 'w') as f:
                    f.write(access_token)

                token_update_time = datetime.now()
                with open('token_update_time.txt', 'w') as f:
                    f.write(token_update_time.strftime('%Y-%m-%d %H:%M:%S'))
                logger.info("액세스 토큰 발급 완료")
            else:
                logger.error("액세스 토큰 발급 실패")
    except Exception as e:
        logger.error(f"액세스 토큰 발급 중 오류 발생: {e}")

#####################################################################
# 웹소켓 접속키 발급(코스피)

async def get_approval():
    global approval_key

    # if real_demo == 0:
        # url = 'https://openapivts.koreainvestment.com:29443' # 모의투자계좌
    if real_demo == 1:
        url = 'https://openapi.koreainvestment.com:9443'  # 실전투자계좌
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials",
            "appkey": api_key,
            "secretkey": secret_key}
    PATH = "oauth2/Approval"
    URL = f"{url}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    approval_key = res.json()["approval_key"]

    return approval_key

# 웹소켓 접속키 발급(해외선물)
def get_approval_ovs(key, secret):
    # url = https://openapivts.koreainvestment.com:29443' # 모의투자계좌
    url = 'https://openapi.koreainvestment.com:9443'  # 실전투자계좌
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials",
            "appkey": key,
            "secretkey": secret}
    PATH = "oauth2/Approval"
    URL = f"{url}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    approval_key = res.json()["approval_key"]
    return approval_key


#####################################################################
# 웹소켓 접속 및 데이터 수신

async def connect_websocket(session):
    global ord_sent, npp

    await get_approval()

    if real_demo == 0:
        url = 'ws://ops.koreainvestment.com:31000' # 모의투자계좌
    if real_demo == 1:
        url = 'ws://ops.koreainvestment.com:21000'  # 실전투자계좌

    code_list = [['1','H0IFASP0','101V09'],['1','H0IFCNT0','101V09'], # 지수선물호가, 체결가
                 ['1', 'H0IFASP0', code], ['1', 'H0IFCNT0', code],
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
                        pass

                    elif trid0 == "H0IFCNT0":  # 지수선물체결 데이터 처리
                        # print("#### 지수선물체결 ####")
                        data_cnt = int(recvstr[2])  # 체결데이터 개수

                        # npp 계산
                        asyncio.create_task(stockspurchase_futs(data_cnt, recvstr[3]))  # price 출력

                    elif trid0 == "H0IFCNI0":  # 선물옵션체결통보 데이터 처리
                        logger.info("#### 선물옵션체결통보 ####")
                        key = "AES_key"
                        iv = "AES_iv"
                        # asyncio.create_task(stocksigningnotice_futsoptn(recvstr[3], key, iv))

            except Exception as e:
                logger.error(f"Error while receiving data from websocket: {e}")

#####################################################################
# 해외선물

async def connect(session):

    g_appkey = api_key
    g_appsceret = secret_key
    g_approval_key = get_approval_ovs(g_appkey, g_appsceret)
    print("approval_key [%s]" % (g_approval_key))

    # url = 'ws://ops.koreainvestment.com:31000' # 모의투자계좌
    url = 'ws://ops.koreainvestment.com:21000'  # 실전투자계좌

    # 원하는 호출을 [tr_type, tr_id, tr_key] 순서대로 리스트 만들기

    ### 4. 해외선물옵션 호가, 체결가, 체결통보 ###
    # code_list = [['1','HDFFF020','FCAZ22']] # 해외선물체결
    # code_list = [['1','HDFFF010','FCAZ22']] # 해외선물호가
    # code_list = [['1','HDFFF020','OESH23 C3900']] # 해외옵션체결
    # code_list = [['1','HDFFF010','OESH23 C3900']] # 해외옵션호가
    # code_list = [['1','HDFFF2C0','HTS ID를 입력하세요']] # 해외선물옵션체결통보
    code_list = [['1', 'HDFFF020', code_ovs], ['1', 'HDFFF010', code_ovs], ['1', 'HDFFF2C0', 'jika79']] # ['1', code_ovs, 'OESH23 C3900'], ['1', code_ovs, 'OESH23 C3900'],

    senddata_list = []

    for i, j, k in code_list:
        temp = '{"header":{"approval_key": "%s","custtype":"P","tr_type":"%s","content-type":"utf-8"},"body":{"input":{"tr_id":"%s","tr_key":"%s"}}}' % (g_approval_key, i, j, k)
        senddata_list.append(temp)

    async with websockets.connect(url, ping_interval=None) as websocket:

        for senddata in senddata_list:
            await websocket.send(senddata)
            await asyncio.sleep(0.5)
            print(f"Input Command is :{senddata}")

        while True:

            try:

                data = await websocket.recv()
                await asyncio.sleep(0.5)
                # print(f"Recev Command is :{data}")  # 정제되지 않은 Request / Response 출력

                if data[0] == '0':
                    recvstr = data.split('|')  # 수신데이터가 실데이터 이전은 '|'로 나뉘어져있어 split
                    trid0 = recvstr[1]

                    if trid0 == "HDFFF010":  # 해외선물옵션호가 tr 일경우의 처리 단계
                        print("#### 해외선물옵션호가 ####")
                        stockhoka_overseafut(recvstr[3])
                        await asyncio.sleep(0.5)

                    elif trid0 == "HDFFF020":  # 해외선물옵션체결 데이터 처리
                        print("#### 해외선물옵션체결 ####")
                        data_cnt = int(recvstr[2])  # 체결데이터 개수
                        stockspurchase_overseafut(data_cnt, recvstr[3])
                        await asyncio.sleep(0.5)

                elif data[0] == '1':

                    recvstr = data.split('|')  # 수신데이터가 실데이터 이전은 '|'로 나뉘어져있어 split
                    trid0 = recvstr[1]

                    if trid0 == "HDFFF2C0":  # 해외선물옵션체결 통보 처리
                        stocksigningnotice_overseafut(recvstr[3], aes_key, aes_iv)

                else:

                    jsonObject = json.loads(data)
                    trid = jsonObject["header"]["tr_id"]

                    if trid != "PINGPONG":
                        rt_cd = jsonObject["body"]["rt_cd"]

                        if rt_cd == '1':  # 에러일 경우 처리

                            if jsonObject["body"]["msg1"] != 'ALREADY IN SUBSCRIBE':
                                print("### ERROR RETURN CODE [ %s ][ %s ] MSG [ %s ]" % (jsonObject["header"]["tr_key"], rt_cd, jsonObject["body"]["msg1"]))
                            break

                        elif rt_cd == '0':  # 정상일 경우 처리
                            print("### RETURN CODE [ %s ][ %s ] MSG [ %s ]" % (jsonObject["header"]["tr_key"], rt_cd, jsonObject["body"]["msg1"]))

                            # 체결통보 처리를 위한 AES256 KEY, IV 처리 단계
                            if trid == "HDFFF2C0":  # 해외선물옵션
                                aes_key = jsonObject["body"]["output"]["key"]
                                aes_iv = jsonObject["body"]["output"]["iv"]
                                print("### TRID [%s] KEY[%s] IV[%s]" % (trid, aes_key, aes_iv))

                    elif trid == "PINGPONG":
                        print("### RECV [PINGPONG] [%s]" % (data))
                        await websocket.pong(data)
                        print("### SEND [PINGPONG] [%s]" % (data))

            except websockets.ConnectionClosed:
                continue

#####################################################################
# file_check

async def file_check():
    global price, now_prc, npp, npp2

    now = datetime.now()

    if 1==0:
        # NP에서 데이터 입수 기록
        file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp_.txt'
        # file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp_.txt'
        f1 = open(file_path, 'r')
        f1_r = f1.readline()
        np1, prf1 = f1_r.strip().split(',')
        print("NP1 (-):", np1, prf1)
        f1.close()

        # (cover_b) -> cover_ordered = 1/0
        file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo_2/npp.txt'
        # file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp.txt'
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

    np1 = NP.cover_ordered
    prf1 = NP.profit_opt
    np2 = NP2.cover_ordered
    prf2 = NP2.profit_opt
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
    if recvvalue[0] == "101V09":

        lblSqty2v = float(recvvalue[23])
        lblSqty1v = float(recvvalue[22])
        lblShoga1v =float(recvvalue[2])

        lblBqty2v = float(recvvalue[28])
        lblBqty1v = float(recvvalue[27])
        lblBhoga1v = float(recvvalue[7])

# 해외선물옵션호가 출력라이브러리
def stockhoka_overseafut(data):
    global lblSqty2v, lblSqty1v, lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v

    # print(data)
    recvvalue = data.split('^')  # 수신데이터를 split '^'

    if recvvalue[0] == code_ovs:

        lblSqty2v = float(recvvalue[13])
        lblSqty1v = float(recvvalue[7])
        lblShoga1v =float(recvvalue[9])

        lblBqty2v = float(recvvalue[10])
        lblBqty1v = float(recvvalue[4])
        lblBhoga1v = float(recvvalue[6])

    # print("종목코드	 [" + recvvalue[0] + "]")
    # print("수신일자	 [" + recvvalue[1] + "]")
    # print("수신시각	 [" + recvvalue[2] + "]")
    # print("전일종가	 [" + recvvalue[3] + "]")
    # print("====================================")
    # print("매수1수량 	[" + recvvalue[4] + "]" + ",    매수1번호 	[" + recvvalue[5] + "]" + ",    매수1호가 	[" + recvvalue[6] + "]")
    # print("매도1수량 	[" + recvvalue[7] + "]" + ",    매도1번호 	[" + recvvalue[8] + "]" + ",    매도1호가 	[" + recvvalue[9] + "]")
    # print("매수2수량 	[" + recvvalue[10] + "]" + ",    매수2번호 	[" + recvvalue[11] + "]" + ",    매수2호가 	[" + recvvalue[12] + "]")
    # print("매도2수량 	[" + recvvalue[13] + "]" + ",    매도2번호 	[" + recvvalue[14] + "]" + ",    매도2호가 	[" + recvvalue[15] + "]")
    # print("매수3수량 	[" + recvvalue[16] + "]" + ",    매수3번호  	[" + recvvalue[17] + "]" + ",    매수3호가  	[" + recvvalue[18] + "]")
    # print("매도3수량 	[" + recvvalue[19] + "]" + ",    매도3번호 	[" + recvvalue[20] + "]" + ",    매도3호가 	[" + recvvalue[21] + "]")
    # print("매수4수량 	[" + recvvalue[22] + "]" + ",    매수4번호 	[" + recvvalue[23] + "]" + ",    매수4호가 	[" + recvvalue[24] + "]")
    # print("매도4수량 	[" + recvvalue[25] + "]" + ",    매도4번호 	[" + recvvalue[26] + "]" + ",    매도4호가 	[" + recvvalue[27] + "]")
    # print("매수5수량 	[" + recvvalue[28] + "]" + ",   매수5번호 	[" + recvvalue[29] + "]" + ",    매수5호가 	[" + recvvalue[30] + "]")
    # print("매도5수량 	[" + recvvalue[31] + "]" + ",    매도5번호 	[" + recvvalue[32] + "]" + ",    매도5호가 	[" + recvvalue[33] + "]")
    # print("====================================")
    # print("전일정산가 	[" + recvvalue[32] + "]")

#####################################################################
# 지수선물체결처리 출력라이브러리

async def stockspurchase_futs(data_cnt, data):
    global price, volume, cvolume, cgubun, count, npp, npp2, elap, NP, NP2
    global last_time, last_volume, cgubun_sum, cvolume_mid, cvolume_sum, count_mid, nf, ExedQty
    global lblSqty2v, lblSqty1v, lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1

    # print("============================================")
    menulist = "선물단축종목코드|영업시간|선물전일대비|전일대비부호|선물전일대비율|선물현재가|선물시가|선물최고가|선물최저가|최종거래량|누적거래량|누적거래대금|HTS이론가|시장베이시스|괴리율|근월물약정가|원월물약정가|스프레드|미결제약정수량|미결제약정수량증감|시가시간|시가대비현재가부호|시가대비지수현재가|최고가시간|최고가대비현재가부호|최고가대비지수현재가|최저가시간|최저가대비현재가부호|최저가대비지수현재가|매수비율|체결강도|괴리도|미결제약정직전수량증감|이론베이시스|선물매도호가|선물매수호가|매도호가잔량|매수호가잔량|매도체결건수|매수체결건수|순매수체결건수|총매도수량|총매수수량|총매도호가잔량|총매수호가잔량|전일거래량대비등락율|협의대량거래량|실시간상한가|실시간하한가|실시간가격제한구분"
    menustr = menulist.split('|')
    pValue = data.split('^')
    # print("code: ", pValue[0])

    # 매매대상 미니선물(월별) 코드 입력
    if pValue[0] == "105V07":
        # 현재가
        prc_o1 = float(pValue[5])

    # 기준 정규선물(분기별) 코드 입력
    elif pValue[0] == "101V09":
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
        if mt < 1:
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
            if 1==0:
                if NP.auto_cover == 1:
                    file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp_.txt'
                    # file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp_.txt'
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
                    # file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp.txt'
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
# 해외선물옵션체결처리 출력라이브러리
def stockspurchase_overseafut(data_cnt, data):
    global price, volume, cvolume, cgubun, count, npp, npp2, elap, NP, NP2
    global last_time, last_volume, cgubun_sum, cvolume_mid, cvolume_sum, count_mid, nf, ExedQty
    global lblSqty2v, lblSqty1v, lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1

    print("============================================")
    # menulist = "종목코드|영업일자|장개시일자|장개시시각|장종료일자|장종료시각|전일종가|수신일자|수신시각|본장_전산장구분|체결가격|체결수량|전일대비가|등락률|시가|고가|저가|누적거래량|전일대비부호|체결구분|수신시각2만분의일초|전일정산가|전일정산가대비|전일정산가대비가격|전일정산가대비율"
    # menustr = menulist.split('|')
    # pValue = data.split('^')
    # print(pValue)
    # i = 0
    # for cnt in range(data_cnt):  # 넘겨받은 체결데이터 개수만큼 print 한다
    #     print("### [%d / %d]" % (cnt + 1, data_cnt))
    #     for menu in menustr:
    #         print("%-13s[%s]" % (menu, pValue[i]))
    #         i += 1

    # print("============================================")
    menulist = "종목코드|영업일자|장개시일자|장개시시각|장종료일자|장종료시각|전일종가|수신일자|수신시각|본장_전산장구분|체결가격|체결수량|전일대비가|등락률|시가|고가|저가|누적거래량|전일대비부호|체결구분|수신시각2만분의일초|전일정산가|전일정산가대비|전일정산가대비가격|전일정산가대비율"
    menustr = menulist.split('|')
    pValue = data.split('^')
    # print("values: ", pValue[0:20])

    if pValue[0] != code_ovs:
        # 현재가
        prc_o1 = float(pValue[10])

    elif pValue[0] == code_ovs:
        # 현재가
        price = float(pValue[10])/100
        volume = pValue[11]
        if last_volume == 0:
            last_volume = int(volume) + 1
        cvolume = int(volume) - int(last_volume)  #pValue[9]
        last_volume = volume
        cgubun = pValue[5]
        if pValue[19] == "5": #2:매수체결 5:매도체결
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
        if mt < 1:
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
            try:
                npp = NP.nprob(price, timestamp, mt, count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v,
                               lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1)
                npp2 = NP2.nprob(price, timestamp, mt, count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v,
                               lblShoga1v, lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1)
            except:
                print("npp error")

            # 기록
            if 1==0:
                if NP.auto_cover == 1:
                    file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp_.txt'
                    # file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp_.txt'
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
                    # file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp.txt'
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


#####################################################################
# 주문 처리 함수

async def place_order():
    global npp, npp2, nf, np_count, cum_qty

    # test 주문
    if 1==0 and nf == 10:
        await send_order(bns = "02")

    if which_market == 1:
        if npp ==  4:
            await send_order(bns = "02")
        elif npp ==  -4:
            await send_order(bns = "01")

        if npp2 ==  4:
            await send_order(bns = "02")
        elif npp2 ==  -4:
            await send_order(bns = "01")

    if which_market == 2:
        if npp ==  4:
            await send_order_ovs(bns = "02")
        elif npp ==  -4:
            await send_order_ovs(bns = "01")

        if npp2 ==  4:
            await send_order_ovs(bns = "02")
        elif npp2 ==  -4:
            await send_order_ovs(bns = "01")

    # 기록
    if 1==0:
        try:
            # file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo/npp_.txt'
            file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp_.txt'
            f1 = open(file_path, 'r')
            f1_r = f1.readline()
            np1, prf1, chkForb = f1_r.strip().split(',')
            NP.np1 = int(np1)
            print("NP1 (-):", np1, prf1)
            f1.close()

            # (cover_b) -> cover_ordered = 1/0
            # file_path = '/Users/yugjingwan/PycharmProjects/MG_HanTo_2/npp.txt'
            file_path = 'C:/Users/Administrator/PycharmProjects/MG_HanTo/npp.txt'
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

        except:
            pass

    np1 = NP.cover_ordered
    prf1 = NP.profit_opt
    NP.np1 = int(np1)
    NP2.np1 = int(np1)

    np2 = NP2.cover_ordered
    prf2 = NP2.profit_opt
    NP.np2 = int(np2)
    NP2.np2 = int(np2)

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

    print('NP: ', npp, npp2)
    print("[sys] ****** np.exed : ", NP.exed_qty, NP2.exed_qty)
    print("[sys] Exed: ", ExedQty)

#####################################################################
# 일반 주문

async def send_order(bns):
    global orders, ord_sent, api_key, secret_key, price, qty, code, account, chkForb, auto_time, prc_o1
    global gui, NP, slack, which_market

    print("주문변수: ", account, bns, code, str(qty), str(prc_o1 - 1))

    if auto_time == 1:
        if real_demo == 0:
            url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order"
        if real_demo == 1:
            url = "https://openapi.koreainvestment.com:9443/uapi/domestic-futureoption/v1/trading/order"

        # 테스트용
        prc_o1 = 370.20
        code = "105V07"

        if bns == "02":
            adj_prc = float(prc_o1) - slack
        elif bns == "01":
            adj_prc = float(prc_o1) + slack

        payload = json.dumps({
            "ORD_PRCS_DVSN_CD": "02",
            "CANO": account,
            "ACNT_PRDT_CD": "03",
            "SLL_BUY_DVSN_CD": bns,
            "SHTN_PDNO": code,
            "ORD_QTY": str(qty),
            "UNIT_PRICE": str(adj_prc),
            "NMPR_TYPE_CD": "01",
            "KRX_NMPR_CNDT_CD": "0",
            "ORD_DVSN_CD": "01"
        })

        if real_demo == 1:
            tr_id = 'TTTO1101U'
        if real_demo == 0:
            tr_id = 'VTTO1103U'

        headers = {
            'content-type': 'application/json',
            'authorization': 'Bearer ' + str(access_token),
            'appkey': api_key,#'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
            'appsecret': secret_key, #'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
            'tr_id': 'TTTO1101U', #'tr_id': 'VTTO1101U',
            'hashkey': ''
        }
        # headers = {
        #     'content-type': 'application/json',
        #     'authorization': 'Bearer ' + access_token,  # 'Bearer ' + str(access_token),
        #     'appkey': "PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC",
        #     'appsecret': "KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=",
        #     'tr_id': 'TTTO1101U'
        # }

        if chkForb != 1:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload) as response:
                    result = await response.json()
                    print("신규주문 회신 data: ", result)
                    if result["rt_cd"] == "0":
                        ord_no = result["output"]["ODNO"]
                        NP.OrgOrdNo = str(ord_no)
                        logger.info(f"주문 요청 완료 - 주문번호: {ord_no}")
                        orders[ord_no] = (bns, qty, price, prc_o1, datetime.now().strftime("%H:%M"), False)
                        asyncio.create_task(send_messages(chat_id=chat_id, text=f"신규 주문 요청 - 주문번호: {ord_no}, 구분: {bns}, 주문가격: {str(prc_o1)}, 주문수량: {qty}"))
                        ord_sent = 1
                        await update_order_list()  # 주문 내역 업데이트
                    else:
                        logger.error("주문 요청 실패")
    # gui.order_button.setStyleSheet("")

#####################################################################
# 일반 주문(해외선물)

async def send_order_ovs(bns):
    global orders, ord_sent, api_key, secret_key, price, qty, code, account, chkForb, auto_time, prc_o1
    global gui, NP, slack, which_market

    print("주문변수: ", account, bns, code, str(qty), str(prc_o1 - 1))

    if auto_time == 1:
        # if real_demo == 0:
        #     url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order"
        if real_demo == 1:
            url = "https://openapi.koreainvestment.com:9443/uapi/overseas-futureoption/v1/trading/order"

        if bns == "02":
            adj_prc = float(prc_o1) - slack
        elif bns == "01":
            adj_prc = float(prc_o1) + slack

        payload = json.dumps({
            "CANO": account,
            "ACNT_PRDT_CD": "08",
            "OVRS_FUTR_FX_PDNO": code_ovs,
            "SLL_BUY_DVSN_CD": bns,
            "FM_LQD_USTL_CCLD_DT": "",
            "FM_LQD_USTL_CCNO": "",
            "PRIC_DVSN_CD": "1",  # 가격구분코드
            "FM_LIMIT_ORD_PRIC": str(adj_prc), #지정가인 경우 가격 입력 * 시장가, STOP주문인 경우, 빈칸("") 입력
            "FM_STOP_ORD_PRIC": "",
            "FM_ORD_QTY": str(qty),
            "FM_LQD_LMT_ORD_PRIC": "",
            "FM_LQD_STOP_ORD_PRIC": "",
            "CCLD_CNDT_CD": "6",  # 체결조건코드 : 일반적으로 6 (EOD, 지정가) GTD인 경우 5, 시장가인 경우만 2
            "CPLX_ORD_DVSN_CD": "0",
            "ECIS_RSVN_ORD_YN": "N",
            "FM_HDGE_ORD_SCRN_YN": "N"
        })

        headers = {
            'content-type': 'application/json',
            'authorization': 'Bearer ' + str(access_token),
            'appkey': 'PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC',
            'appsecret': 'KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=',
            'tr_id': 'OTFM3001U',
            'hashkey': '46149c294a68a8fc71336b76fbafe5193698f28b5f83c3c3193f2c5f8f9c1aa4'
        }

        # OTFM3001U : ASFM선물옵션주문신규
        # [POST API 대상] Client가 요청하는 Request Body를 hashkey api로 생성한 Hash값  * API문서 > hashkey 참조

        if chkForb != 1:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload) as response:
                    result = await response.json()
                    print("신규주문 회신 data: ", result)
                    if result["rt_cd"] == "0":
                        ord_no = result["output"]["ODNO"]
                        NP.OrgOrdNo = str(ord_no)
                        logger.info(f"주문 요청 완료 - 주문번호: {ord_no}")
                        orders[ord_no] = (bns, qty, price, prc_o1, datetime.now().strftime("%H:%M"), False)
                        asyncio.create_task(send_messages(chat_id=chat_id, text=f"신규 주문 요청 - 주문번호: {ord_no}, 구분: {bns}, 주문가격: {str(prc_o1)}, 주문수량: {qty}"))
                        ord_sent = 1
                        await update_order_list()  # 주문 내역 업데이트
                    else:
                        logger.error("주문 요청 실패")
    # gui.order_button.setStyleSheet("")


#####################################################################

async def update_order_list():
    global orders, ord_sent, api_key, secret_key, price, qty, code, account, chkForb, auto_time, prc_o1
    global gui

    order_text = ""
    for ord_no, order_info in orders.items():
        print("order_info : ", ord_no, order_info)
        bns, qty, price, prc_o1, time, is_modified = order_info
        if bns == '02':
            order_text += f"{ord_no[-5:]}, <span style='color:red;'>매수</span>, {prc_o1}, @ {time}<br>"
        else:
            order_text += f"{ord_no[-5:]}, <span style='color:blue;'>매도</span>, {prc_o1}, @ {time}<br>"
    gui.order_list_text.setText(order_text)
    gui.order_button.setStyleSheet("")

#####################################################################
# 미체결 주문 확인 및 처리

async def check_unexecuted_orders(session):
    global access_token, prc_o1, api_key, secret_key
    global unexecuted_orders, orders, orders_che, reordered
    global temp_id, manu_reorder, gui

    while True:

        if real_demo == 0:
            url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/inquire-ccnl"
        if real_demo == 1:
            url = "https://openapi.koreainvestment.com:9443/uapi/domestic-futureoption/v1/trading/inquire-ccnl"

        today = datetime.now().strftime("%Y%m%d")
        # access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6Ijc1N2NjNDZkLWJlNTktNGU1OS05MGFlLTFmMTBmNDU1NTY4MCIsImlzcyI6InVub2d3IiwiZXhwIjoxNzEyOTY3MjEzLCJpYXQiOjE3MTI4ODA4MTMsImp0aSI6IlBTcFJWc0tTTllqZE9UbXZjclBOMEMwTXl1cUVaQmFleTJBQyJ9.IKmyzWgbcAlefsraDfKnSUMl7fIG0oWYcmgPgp5D6cbDPllomPwCCYJhrNadD1qXpF3dvudkh-_te1JEU69dYw"

        # print("today :",today)
        # print("access_token :", access_token)

        # demo
        # payload = {
        #     "CANO": account,
        #     "ACNT_PRDT_CD": "03",
        #     "STRT_ORD_DT": today,
        #     "END_ORD_DT": today,
        #     "SLL_BUY_DVSN_CD": "00",
        #     "CCLD_NCCS_DVSN": "00",
        #     "SORT_SQN": "DS",
        #     "STRT_ODNO": "",
        #     "PDNO": "",
        #     "MKET_ID_CD": "00",
        #     "CTX_AREA_FK200": "",
        #     "CTX_AREA_NK200": ""
        # }
        #

        # headers = {
        #     'content-type': 'application/json',
        #     'authorization': 'Bearer ' + str(access_token),
        #     'appkey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
        #     'appsecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
        #     'tr_id': 'VTTO5201R' #VTTO5201R
        # }

        # REAL

        today = datetime.now()
        formatted_date = today.strftime("%Y%m%d")

        payload = {
            "CANO": account,
            "ACNT_PRDT_CD": "03",
            "STRT_ORD_DT": formatted_date,
            "END_ORD_DT": formatted_date,
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
            'authorization': 'Bearer ' + access_token,  # 'Bearer ' + str(access_token),
            'appkey': "PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC",
            'appsecret': "KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=",
            'tr_id': 'TTTO5201R'  # VTTO5201R
        }

    # while True:

        # #(1) 시스템상
        # try:
        #     for ord_no, order_info in orders.items():
        #         if ord_no not in orders_che:
        #             unexecuted_orders[ord_no] = order_info
        #     print("unexecuted_orders(시스템) : ", unexecuted_orders)
        # except Exception as e:
        #     logger.error(f"미체결 주문 확인 중 오류 발생(시스템): {e}")

        # (2) 증권사 조회
        try:
            response = requests.request("GET", url, headers=headers, params=payload)
            data = response.json()
            # print("API 응답:", data)  # 응답 데이터 출력

            if "output1" in data and len(data["output1"]) > 0:
                df = pd.DataFrame(data["output1"])
                # print(df)
                df['qty'] = df['qty'].astype(int)
                df['ord_idx'] = df['ord_idx'].astype(float)
                df['tot_ccld_qty'] = df['tot_ccld_qty'].astype(float)

                # 정정 및 취소 주문 제외 처리
                df_che = df[df['tot_ccld_qty'] != 0]
                revised_canceled_df = df_che[df_che['trad_dvsn_name'].isin(['취소확인'])]
                revised_canceled_odnos = revised_canceled_df['odno'].tolist()
                df_che = df_che[~df_che['odno'].isin(revised_canceled_odnos)]

                # 체결 주문 필터링
                executed_df = df_che[(df_che['qty'] == 0) & (df_che['ord_idx'] != 0)][['ord_dt', 'odno', 'ord_tmd', 'trad_dvsn_name', 'ord_qty', 'ord_idx']]

                try:
                    # orders_che 딕셔너리 초기화
                    orders_che = {}

                    for _, order in executed_df.iterrows():
                        ord_no = order['odno']
                        order_info = (order['odno'], order['ord_tmd'], order['trad_dvsn_name'][-2:], order['ord_qty'], order['ord_idx'])
                        orders_che[ord_no] = order_info

                    # print("orders_che(체결 주문) : ", orders_che)
                    asyncio.create_task(update_execution_list())  # 체결 내역 업데이트
                except Exception as e:
                    logger.error(f"체결 주문 확인 중 오류 발생(증권사): {e}")

                # 미체결 주문 필터링
                filtered_df = df[(df['qty'] > 0) & (df['ord_idx'] != 0)][['ord_dt', 'odno', 'ord_tmd', 'trad_dvsn_name', 'ord_qty', 'ord_idx']]
                if filtered_df.empty:
                    unexecuted_orders = {}
                try:
                    for _, order in filtered_df.iterrows():
                        ord_no = order['odno']
                        if ord_no not in orders_che:
                            order_info = (order['odno'], order['ord_tmd'], order['trad_dvsn_name'][-2:], order['ord_qty'], order['ord_idx'])
                            unexecuted_orders[ord_no] = order_info
                    print("unexecuted_orders(정정주문 이전) : ", unexecuted_orders)
                except Exception as e:
                    logger.error(f"미체결 주문 확인 중 오류 발생(증권사): {e}")

                # 미체결 주문 정정
                if reordered == 0:
                    current_time = datetime.now()
                    for _, row in filtered_df.iterrows():
                        odno = int(row['odno'])
                        ord_qty = int(row['ord_qty'])
                        ord_dt = row['ord_dt']  # 주문일자 추출
                        ord_tmd = row['ord_tmd']

                        # 주문일자와 주문시각을 조합하여 datetime 객체 생성
                        ord_datetime = datetime.strptime(ord_dt + ord_tmd, '%Y%m%d%H%M%S')

                        # 주문시각과 현재시각 비교
                        elapsed_time = (current_time - ord_datetime).total_seconds()
                        # print("times : ", elapsed_time, current_time, ord_datetime)  # 수정된 부분

                        if 1==1 and elapsed_time > 20 and datetime.now().second % 5 == 0:
                            # 현재 가격으로 정정주문 요청
                            print(f"주문번호 {odno} - 주문 시간 경과({elapsed_time}초), 정정 주문 실행")
                            await modify_order(odno, ord_qty, prc_o1 - 1)
                        else:
                            print(f"주문번호 {odno}은 주문시각으로부터 20초 이내입니다. 정정주문 미실행.")

                        if temp_id != "" and manu_reorder == 1:
                            await modify_order(temp_id, ord_qty, prc_o1 - 1)
                            manu_reorder = 0
                            gui.modify_order_button.setStyleSheet("")
                            print("manu_reordered..reset manu_reorder to : ", manu_reorder)

                    # 기존 미체결주문 목록 삭제
                    unexecuted_orders.clear()

                    # 새로운 미체결주문 목록 업데이트
                    for _, order in filtered_df.iterrows():
                        ord_no = order['odno']
                        if ord_no not in orders_che:
                            order_info = (order['odno'], order['ord_tmd'], order['trad_dvsn_name'][-2:], order['ord_qty'], order['ord_idx'])
                            unexecuted_orders[ord_no] = order_info

                    print("unexecuted_orders(정정주문 이후) : ", unexecuted_orders)

            # else:
            #     print("증권사 API 응답에 'output1' 키가 없습니다.")
            #     with open('access_token.txt', 'r') as f:
            #         access_token = f.read().strip()
            #         logger.info("저장된 액세스 토큰 사용(발급직후): %s", access_token)
            elif "msg1" in data:
                print("(미체결) 응답 메시지 : ", data["msg1"])

            else:
                print("(미체결) 증권사 API 응답에 'output1' 키와 'msg1' 키가 없습니다.")
                with open('access_token.txt', 'r') as f:
                    access_token = f.read().strip()
                    logger.info("(미체결) 저장된 액세스 토큰 사용(발급직후): %s", access_token)


        except Exception as e:
            logger.error(f"미체결 주문 확인 중 오류 발생(증권사): {e}")

        await asyncio.sleep(2)  # 5초 간격으로 미체결 주문 확인

#####################################################################
# 정정주문

async def modify_order(odno, ord_qty, prc_o1):
    global price, gui, access_token, real_demo, slack

    # if real_demo == 0:
    #     url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-futureoption/v1/trading/order-rvsecncl"
    # if real_demo == 1:
    url = "https://openapi.koreainvestment.com:9443/uapi/domestic-futureoption/v1/trading/order-rvsecncl"

    # ORD_PRCS_DVSN_CD: 주문처리구분코드(02: 실전투자)
    # CANO: 계좌번호
    # ACNT_PRDT_CD: 계좌상품코드(03: 선물옵션)
    # RVSE_CNCL_DVSN_CD: 정정취소구분코드(01:정정, 02: 취소)
    # ORGN_ODNO: 원주문번호
    # ORD_QTY: 주문수량, TTTO1103U(선물옵션 정정취소 주간) 사용 중 전량 취소일 경우 0(혹은 아무 숫자)로 입력 부탁드립니다.
    # UNIT_PRICE: 주문가격
    # NMPR_TYPE_CD: 호가유형코드(01: 지정가)
    # KRX_NMPR_CNDT_CD: KRX호가조건코드(0: 없음)
    # RMN_QTY_YN: 잔량처리여부(Y: 잔량전부)
    # ORD_DVSN_CD: 주문구분코드(01: 일반주문)

    print(account, str(odno), str(ord_qty), str(prc_o1 - 1))

    payload = json.dumps({
        "ORD_PRCS_DVSN_CD": "02",
        "CANO": account,
        "ACNT_PRDT_CD": "03",
        "RVSE_CNCL_DVSN_CD": "01",
        "ORGN_ODNO": str(odno),
        "ORD_QTY": str(ord_qty),
        "UNIT_PRICE": str(prc_o1),
        "NMPR_TYPE_CD": "01",
        "KRX_NMPR_CNDT_CD": "0",
        "RMN_QTY_YN": "Y",
        "ORD_DVSN_CD": "01"
    })

    # if real_demo == 0:
    #     tr_id = 'VTTO1103U'
    # if real_demo == 1:
    #     tr_id = 'TTTO1103U'

    headers = {
        'Content-Type': 'application/json',
        'authorization': 'Bearer ' + str(access_token),
        # 'appKey': 'PSMID6MolzScnX0scR9WB7gZUK3cxrua4FwF',
        # 'appSecret': 'rTk4mvvNOEnF1iW6KV1/wCYR/ONhS1GjxktQN1YVC7YcguxMKWnin0x1XMfp8ansUwaNAo5a5mDPN+yNwgCc9HUWz5gaTyZWwB4VOCnXoXVjUfmkRzC3DEiyxL34lpPTz3woB7RJbKFKLHmxX7Rd3Iczla0p6y1Fst2TqT+52bN+Lmu1Z3s=',
        'appKey': "PSpRVsKSNYjdOTmvcrPN0C0MyuqEZBaey2AC",
        'appSecret': "KyTMYmD49Rbh+/DhtKYUuSRv6agjM9zxXs9IIHx9vz4UiCurqbpEPoawVFKNrx3DryhrLjxDy/vFbe/4acttdIU5hz6thCiPgeBLCEGpQcXvluQQWRNJg77ztOUPcPpqg3gVS+LxGOaOF9sB/n19fJmhf+O2cht6swH5Iz4aHUJZsZ0nrZM=",
        'tr_id': 'TTTO1103U',
        'hashkey': ''
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json()
        # print("정정주문 회신 data: ", data)

        if data["rt_cd"] == "0":
            new_odno = data['output']['ODNO']
            logger.info(f"주문 정정 성공: {odno}")
            str_odno = str(odno).zfill(10)  # odno를 문자열로 변환하고 10자리로 맞춤
            if str_odno in orders:
                side = orders[str_odno][0]  # 기존 주문의 side 값 가져오기
                orders[new_odno] = (side, ord_qty, price, prc_o1, datetime.now().strftime("%H:%M"), True)
                del orders[str_odno]  # 기존 주문번호 삭제
                await update_order_list()
                # 정정된 주문을 미체결주문 목록에 추가
                # unexecuted_orders[new_odno] = (new_odno, datetime.now().strftime("%H:%M:%S"), "", ord_qty, prc_o1)
                # if str_odno in unexecuted_orders or odno in unexecuted_orders: #
                #     del unexecuted_orders[str_odno]  # 기존 미체결주문 삭제
                #     del unexecuted_orders[odno]  # 기존 미체결주문 삭제
                #     gui.unexecuted_order_list_text.setText(str(unexecuted_orders))
                # print("unexecuted_orders_new :", unexecuted_orders)
            else:
                logger.warning(f"주문번호 {odno}에 해당하는 주문이 orders에 존재하지 않습니다.")
                # print("orders : ", orders)
        else:
            logger.error(f"주문 정정 실패: {odno}, 실패 사유: {data['msg1']}")

    except Exception as e:
        logger.error(f"주문 정정 중 오류 발생: {e}")

#####################################################################
# 선물옵션 체결통보 출력라이브러리

# def stocksigningnotice_futsoptn(data, key, iv):
#     global orders, orders_che, price, qty, code, account, cum_qty
#     global bns_che, qty_che, prc_o1_che, time_che
#     global gui
#
#     try:
#         # AES256 처리 단계
#         aes_dec_str = aes_cbc_base64_dec(key, iv, data)
#         logger.debug("aes_dec_str: %s", aes_dec_str)
#         pValue = aes_dec_str.split('^')
#         logger.debug("pValue: %s", pValue)
#
#         # 정상 체결 통보
#         if pValue[6] == '0':
#             logger.info("#### 지수선물옵션 체결 통보 ####")
#             menulist_sign = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|체결수량|체결단가|체결시간|거부여부|체결여부|접수여부|지점번호|주문수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#             menustr = menulist_sign.split('|')
#
#             try:
#                 bns_che = pValue[4]  # 매도매수구분
#                 qty_che = int(pValue[8])  # 체결수량
#                 if bns_che == "매도":
#                     qty_che = -qty_che
#                 cum_qty += qty_che
#
#                 ord_no_che = pValue[3]  # 원주문번호
#                 prc_o1_che = float(pValue[9])  # 체결단가
#                 time_che = pValue[10]  # 체결시간
#
#                 # 체결내역 결과 집계
#                 orders_che[ord_no_che] = (bns_che, qty_che, price, prc_o1_che, time_che)
#                 asyncio.create_task(update_execution_list())  # 체결 내역 업데이트
#
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#                 logger.debug("orders_che: %s", orders_che)
#
#                 if bns_che == "매수":
#                     NP.cover_ordered_exed = 1
#                 elif bns_che == "매도":
#                     NP.cover_ordered_exed = -1
#             except (IndexError, ValueError) as e:
#                 logger.error("체결 통보 데이터 처리 중 오류 발생: %s", e)
#
#         # 주문·정정·취소·거부 접수 통보
#         else:  # pValue[6] == 'L',
#             if pValue[5] == '1':  # 정정 접수 통보 (정정구분이 1일 경우)
#                 logger.info("#### 지수선물옵션 정정 접수 통보 ####")
#                 menulist_revise = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|정정수량|정정단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_revise.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#             elif pValue[5] == '2':  # 취소 접수 통보 (정정구분이 2일 경우)
#                 logger.info("#### 지수선물옵션 취소 접수 통보 ####")
#                 menulist_cancel = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|취소수량|주문단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_cancel.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#             elif pValue[11] == '1':  # 거부 접수 통보 (거부여부가 1일 경우)
#                 logger.info("#### 지수선물옵션 거부 접수 통보 ####")
#                 menulist_refuse = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|주문수량|주문단가|주문시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_refuse.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#             else:  # 주문 접수 통보
#                 logger.info("#### 지수선물옵션 주문 접수 통보 ####")
#                 menulist_order = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|주문수량|체결단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_order.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#     except Exception as e:
#         logger.error("체결 통보 메시지 처리 중 오류 발생: %s", e)


#####################################################################
# 해외선물옵션 체결통보 출력라이브러리

# def stocksigningnotice_overseafut(data, key, iv):
#     global orders, orders_che, price, qty, code, account, cum_qty
#     global bns_che, qty_che, prc_o1_che, time_che
#     global gui
#
#     try:
#         # AES256 처리 단계
#         aes_dec_str = aes_cbc_base64_dec(key, iv, data)
#         logger.debug("aes_dec_str: %s", aes_dec_str)
#         pValue = aes_dec_str.split('^')
#         logger.debug("pValue: %s", pValue)
#
#         # 정상 체결 통보
#         if pValue[6] == '0':
#             logger.info("#### 지수선물옵션 체결 통보 ####")
#             menulist_sign = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|체결수량|체결단가|체결시간|거부여부|체결여부|접수여부|지점번호|주문수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#             menustr = menulist_sign.split('|')
#
#             try:
#                 bns_che = pValue[4]  # 매도매수구분
#                 qty_che = int(pValue[8])  # 체결수량
#                 if bns_che == "매도":
#                     qty_che = -qty_che
#                 cum_qty += qty_che
#
#                 ord_no_che = pValue[3]  # 원주문번호
#                 prc_o1_che = float(pValue[9])  # 체결단가
#                 time_che = pValue[10]  # 체결시간
#
#                 # 체결내역 결과 집계
#                 orders_che[ord_no_che] = (bns_che, qty_che, price, prc_o1_che, time_che)
#                 asyncio.create_task(update_execution_list())  # 체결 내역 업데이트
#
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#                 logger.debug("orders_che: %s", orders_che)
#
#                 if bns_che == "매수":
#                     NP.cover_ordered_exed = 1
#                 elif bns_che == "매도":
#                     NP.cover_ordered_exed = -1
#             except (IndexError, ValueError) as e:
#                 logger.error("체결 통보 데이터 처리 중 오류 발생: %s", e)
#
#         # 주문·정정·취소·거부 접수 통보
#         else:  # pValue[6] == 'L',
#             if pValue[5] == '1':  # 정정 접수 통보 (정정구분이 1일 경우)
#                 logger.info("#### 지수선물옵션 정정 접수 통보 ####")
#                 menulist_revise = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|정정수량|정정단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_revise.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#             elif pValue[5] == '2':  # 취소 접수 통보 (정정구분이 2일 경우)
#                 logger.info("#### 지수선물옵션 취소 접수 통보 ####")
#                 menulist_cancel = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|취소수량|주문단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_cancel.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#             elif pValue[11] == '1':  # 거부 접수 통보 (거부여부가 1일 경우)
#                 logger.info("#### 지수선물옵션 거부 접수 통보 ####")
#                 menulist_refuse = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|주문수량|주문단가|주문시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_refuse.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#             else:  # 주문 접수 통보
#                 logger.info("#### 지수선물옵션 주문 접수 통보 ####")
#                 menulist_order = "고객ID|계좌번호|주문번호|원주문번호|매도매수구분|정정구분|주문종류|단축종목코드|주문수량|체결단가|체결시간|거부여부|체결여부|접수여부|지점번호|체결수량|계좌명|체결종목명|주문조건|주문그룹ID|주문그룹SEQ|주문가격"
#                 menustr = menulist_order.split('|')
#                 i = 0
#                 for menu in menustr:
#                     logger.info("%s  [%s]", menu, pValue[i])
#                     i += 1
#
#     except Exception as e:
#         logger.error("체결 통보 메시지 처리 중 오류 발생: %s", e)

#####################################################################

async def update_execution_list():
    global orders_che, gui

    execution_text = ""
    sorted_orders_che = sorted(orders_che.items(), key=lambda x: x[1][1], reverse=False)
    for ord_no, execution_info in sorted_orders_che:
        odno, ord_tmd, trad_dvsn_name, ord_qty, ord_idx = execution_info
        if trad_dvsn_name == '매수':
            execution_text += f"{ord_no[-5:]}, <span style='color:red;'>{trad_dvsn_name}</span>, {ord_idx}, {ord_qty}, @ {ord_tmd}<br>"
        elif trad_dvsn_name == '매도':
            execution_text += f"{ord_no[-5:]}, <span style='color:blue;'>{trad_dvsn_name}</span>, {ord_idx}, {ord_qty}, @ {ord_tmd}<br>"
        else:
            execution_text += f"{ord_no[-5:]}, <span style='color:brown;'>{trad_dvsn_name}</span>, {ord_idx}, {ord_qty}, @ {ord_tmd}<br>"

    gui.execution_list_text.setHtml(execution_text)

#####################################################################
# 텔레그램 메시지 관리

async def msg():
    global bot, df, nf, df1, msg_now, msg_last, msg_last_sent, bot_alive, OrgOrdNo
    global isblocked_msg, isreleased_msg, istimeblocked_msg, stat_out_org, stat_in_org, OrgOrdNo_Cov
    global started, sub, orders, NP, msg_out, auto_time, chkForb, price

    while True:

        if (datetime.now().minute % 7 == 0 and datetime.now().second <= 1):
            asyncio.create_task(send_messages(chat_id="322233222", text=str(NP.auto_cover) + "now : " + str(price) + "// prf : {:.2f}, {:.2f}".format(NP.profit_opt, NP2.profit_opt)))

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
                        asyncio.create_task(send_messages(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) + " (now): " + msg_now + ", (last) : " + msg_last))

                    msg_last = msg_now

                    if nf > 100 and (msg_now == "stop" or msg_now == "x" or msg_now == "000"):
                        if NP.cover_ordered != 0:
                            text = str(NP.auto_cover) + " = (한투) (MG7) 커버 진입상태임..확인필요 ="
                            chkForb = 1
                            if bot_alive == 1 or bot_alive == 2:
                                asyncio.create_task(send_messages(chat_id="322233222", text=text))
                        if NP.cover_ordered == 0:
                            chkForb = 1
                            text = str(NP.auto_cover) + " = (한투) (MG7) 매매를 중단합니다. ="
                            if bot_alive == 1 or bot_alive == 2:
                                asyncio.create_task(send_messages(chat_id="322233222", text=text))

                        if chkForb == 1 and isblocked_msg == 0:
                            text = str(NP.auto_cover) + " = (한투) (MG7) 매매가 중단된 상태입니다. ="
                            if bot_alive == 1 or bot_alive == 2:
                                asyncio.create_task(send_messages(chat_id="322233222", text=text))
                            isblocked_msg = 1
                            isreleased_msg = 0

                    if msg_now == "auto":
                        if auto_time == 0:
                            auto_time = 1
                        elif auto_time == 1:
                            auto_time = 0
                        asyncio.create_task(send_messages(chat_id="322233222", text="(한투)" + str(auto_time) + ", 0:release, 1:set"))

                    if (msg_now == "start" or msg_now == "111") and msg_out != "start":
                        chkForb = 0
                        text = "(한투)" + str(NP.auto_cover) + " = (한투) (MG7) 매매 중단을 해제합니다. ="
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                        if chkForb == 0 and isreleased_msg == 0:
                            text = "(한투)" + str(NP.auto_cover) + " = (한투) (MG7) 매매가 시작되었습니다. ="
                            if bot_alive == 1 or bot_alive == 2:
                                asyncio.create_task(send_messages(chat_id="322233222", text=text))
                            isreleased_msg = 1
                            isblocked_msg = 0
                            istimeblocked_msg = 0

                    if msg_now[:3] == "out":
                        msg_out = msg_now[4:]
                        asyncio.create_task(send_messages(chat_id="322233222", text= "(한투)" + "아래 명령어 제외 :" + msg_now[4:]))

                    if msg_now == "mout":
                        asyncio.create_task(send_messages(chat_id="322233222", text= "(한투)" + "제외된 명령어 :" + msg_out))

                    if msg_now == "cover":
                        text = "(한투)" + str(NP.auto_cover) + " (한투) cover_ordered: " + str(NP.cover_ordered) + ",  exed: " + str(NP.cover_order_exed)
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    if msg_now == "qty":
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text="(한투)" + str(cum_qty)))

                    if msg_now == "orders":
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) + " " + str(orders)))

                    if msg_now == "orders_che":
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) + " " + str(orders_che)))

                    if msg_now == "now":
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) +" " + msg_now))

                    if msg_now == "last":
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text= "(한투)" + str(NP.auto_cover) +" " + msg_last))

                    if msg_now == "stat":
                        text = "(한투)" + str(NP.auto_cover) + "stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + "(sub) stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    if msg_now == "statset":
                        stat_in_org = [0,0,0]
                        stat_out_org = [1,1,1]
                        text = "(한투)" + str(NP.auto_cover) + "stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + "(sub) stat_in_org: " + str(stat_in_org) + ",  stat_out_org: " + str(stat_out_org)
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    if msg_now == "ord":
                        text = "(한투)" + str(NP.auto_cover) + "Ord: " + str(OrgOrdNo)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) Ord: " + str(OrgOrdNo)
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    if msg_now == "cordnum":
                        text = "(한투)" + str(NP.auto_cover) + "Ord_Cov: " + str(OrgOrdNo_Cov)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) Ord_Cov: " + str(OrgOrdNo_Cov)
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    if msg_now == "corded":
                        text = "(한투)" + str(NP.auto_cover) + "Ord_Ordered: " + str(NP.cover_ordered)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) Ord_Ordered: " + str(NP.cover_ordered)
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    if msg_now == "cexed":
                        text = "(한투)" + str(NP.auto_cover) + "cover_order_exed: " + str(NP.cover_order_exed)
                        if sub == 1:
                            text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) cover_order_exed: " + str(NP.cover_order_exed)
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    # if msg_now == "reord":
                    #     text = "(한투)" + str(NP.auto_cover) + "reOrdered: " + str(reordered) + "  reOrdExed: " + str(reordered_exed)
                    #     if sub == 1:
                    #         text = "(한투)" + str(NP.auto_cover) + str(NP.auto_cover) + " (sub) reOrdered: " + str(reordered) + "  reOrdExed: " + str(reordered_exed)
                    #     if bot_alive == 1 or bot_alive == 2:
                    #         asyncio.create_task(send_messages(chat_id="322233222", text=text)

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
                            asyncio.create_task(send_messages(chat_id="322233222", text=text))

                    if msg_now == "list":
                        # text = " [block, stop, start, time, cover, plot, stat, statset, ord, cordnum, cexed, plotset, reord] "
                        if bot_alive == 1 or bot_alive == 2:
                            asyncio.create_task(send_messages(chat_id="322233222", text="(한투)" + ", ".join(msgs)))
            except:
                if msg_now != "no var in list":
                    asyncio.create_task(send_messages(chat_id="322233222", text="no var in list"))


            if msg_now != msg_last:
                if msg_now[:2] == "np":
                    # want_see = msg_now[2]
                    if bot_alive == 1 or bot_alive == 2:
                        asyncio.create_task(send_messages(chat_id="322233222", text="(한투)" + "np :" + str(NP.msg_now[3:])))

                if msg_now[:6] == "delord":
                    del orders[int(msg_now[6:])]
                    asyncio.create_task(send_messages(chat_id="322233222", text="(한투)" + "deleted : " + str(int(msg_now[6:]))))

                if msg_now[:7] == "deleted":
                    orders = []
                    asyncio.create_task(send_messages(chat_id="322233222", text="(한투)" + "dumped : " + str(NP.auto_cover)))


            if nf % 500 == 0 and nf != 0:
                if chkForb == 0:
                    text = str(NP.auto_cover) + " = (한투) (MG7) In Released Status=" + str(nf)
                    if sub == 1:
                        text = str(NP.auto_cover) + "(한투, sub) = (한투) (MG7) In Released Status=" + str(nf)
                    if bot_alive == 1 or bot_alive == 2:
                        asyncio.create_task(send_messages(chat_id="322233222", text=text))
                if chkForb == 1:
                    text = str(NP.auto_cover) + " = (한투) (MG7) In Blocked Status =" + str(nf)
                    if sub == 1:
                        text = str(NP.auto_cover) + " =(한투, sub) (MG7) In Blocked Status =" + str(nf)
                    if bot_alive == 1 or bot_alive == 2:
                        asyncio.create_task(send_messages(chat_id="322233222", text=text))

        await asyncio.sleep(5)

# def aes_cbc_base64_dec(key, iv, cipher_text):
#     """
#     :param key:  str type AES256 secret key value
#     :param iv: str type AES256 Initialize Vector
#     :param cipher_text: Base64 encoded AES256 str
#     :return: Base64-AES256 decodec str
#     """
#     cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
#     return bytes.decode(unpad(cipher.decrypt(b64decode(cipher_text)), AES.block_size))

def aes_cbc_base64_dec(key, iv, cipher_text):
    """
    :param key:  str type AES256 secret key value
    :param iv: str type AES256 Initialize Vector
    :param cipher_text: Base64 encoded AES256 str
    :return: Base64-AES256 decodec str
    """
    try:
        cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
        return bytes.decode(unpad(cipher.decrypt(b64decode(cipher_text)), AES.block_size))
    except (ValueError, UnicodeDecodeError) as e:
        logger.error("AES-256 CBC 복호화 중 오류 발생: %s", e)
        return ""

# #####################################################################
# 메인

async def main():
    # loop = asyncio.get_event_loop()
    global gui

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    gui = AutoTradeGUI()
    gui.show()

    with loop:
        loop.run_until_complete(run_async_tasks(gui, loop))
        loop.run_forever()

async def run_async_tasks(gui, loop):
    async with aiohttp.ClientSession() as session:

        if which_market == 1:
            await asyncio.gather(
                connect_websocket(session),
                refresh_token(session),
                check_unexecuted_orders(session),
                msg(),
                # loop=loop
            )
        elif which_market == 2:
            await asyncio.gather(
                connect(session),
                refresh_token(session),
                check_unexecuted_orders(session),
                msg(),
                # loop=loop
            )

# 프로그램 실행
if __name__ == "__main__":
    asyncio.run(main())
