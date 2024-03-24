# -*- coding: utf-8 -*-

import time
import pandas as pd
import scipy.stats as stat
from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
import numpy as np
# import talib
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
from keras.models import model_from_json
# from h5py import *
import h5py  # as h5py

# os.environ['KERAS_BACKEND'] = 'theano'
# from twilio.rest import Client
# from keras import backend as K
import keras as ks
# from keras.models import model_from_json
from keras.models import model_from_json
# from tensorflow.keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout
from keras import Sequential
import telegram
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# import tracemalloc
# tracemalloc.start()

import logging
log = logging.getLogger('apscheduler.executors.default')
log.setLevel(logging.INFO)  # DEBUG
fmt = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
h = logging.StreamHandler()
h.setFormatter(fmt)
log.addHandler(h)

bot_alive = 1

### main ###
# chat_token = "5018312596:AAFItTKnL20wxEmG-rLFXTD-vMYQt2MLg-4"
### sub ###
chat_token = "5095431220:AAF5hRJL8mQYCB7tmaqqn_VC06Nwg1ttYB8"

if bot_alive == 1:
    bot1 = telegram.Bot(token=chat_token)


# 테스트 ####
class Nprob:
    global bot_alive, bot1, chat_token

    def __init__(self):

        self.talib = 0

        # global df , nf
        self.nf = 0
        self.no = 0
        self.d_OMain = 0
        self.nfset = 0
        self.cvol_m_cri_start_nf = 0
        self.factor = 1.0
        self.type = 'start'
        self.ai_mode = 3
        self.ai_spot = 0
        self.ai = 0
        self.ai_bns = "n"
        self.ai_dbns = 0
        self.ai_spot_2 = 0
        self.ai_2 = 0
        self.ai_bns_2 = "n"
        self.ai_dbns_2 = 0
        self.ai_bns_3 = "n"
        self.ai_dbns_3 = 0
        self.ai_long_spot = 0
        self.ai_long = 0
        self.ai_long_bns = "n"
        self.ai_long_dbns = 0
        self.ai_bns2_bns = "n"
        self.ai_bns2_dbns = 0
        self.bns_check_mode = 1
        self.bns_check = 0
        self.bns_check_2 = 0
        self.bns_check_3 = 0
        self.bns_check_5 = 0
        self.bns_check_6 = 0
        self.last_cover_signal2 = 0
        self.now_cover_signal2 = 0
        self.last_prc_s_peak = 0
        self.now_prc_s_peak = 0
        self.long_prc_s_peak = 0
        self.last_rsi_peak = 0
        self.now_rsi_peak = 0
        self.peak_block = 0
        self.last_save = 0
        self.np1 = 0
        self.np2 = 0

        now = datetime.now()

        # choose which_market
        if now.hour >= 8 and now.hour <= 15:
            self.which_market = 3  # (1:bitmex, 2:upbit, 3:kospi, 4:e-mini, 5:A50, 6:Micro, 7:HS)
        else:
            self.which_market = 4
        # self.which_market = 1
        # if self.which_market == 4:
        #     chat_token = "5095431220:AAF5hRJL8mQYCB7tmaqqn_VC06Nwg1ttYB8"
        # self.which_market = 1

        # self.auto
        self.auto_cover = 2 # 0:off, 1:sell, 2:buy
        self.acc_uninfied = 1 # 0: accnt_cov seperated, 1:unified

        if self.auto_cover == 1:
            self.OrgMain = "b"
            self.stat_in_org = "111"
            self.stat_out_org = "000"
            # self.chkForb = 1
        if self.auto_cover == 2:
            self.OrgMain = "s"
            self.stat_in_org = "111"
            self.stat_out_org = "000"


        self.max_e_qty = 3
        self.dynamic_cover = 1  # Cover-Up in Dynamic mode  0:N/A, 1:Cover
        self.dyna_out_touch = 0

        self.Forb_setted_sent = 0

        print("which_market: ", self.which_market)

        if self.which_market == 1:  # Bit
            # self.prc_std_per_cri = 0.09 / 100
            self.mode = 0  # 1:trend, -1:inverse
            self.tick = 1
            self.cri_tick = 15  # 3  # Price_cri decide
            self.loop = 0.5
            self.cvol_m_cri = 0  # old in criteria, old prf_able decide
            # self.std_prc_cri = 0.1
            self.cvol_m_limit = 0.02 #0.0015  # 0.003
            self.std_prc_cvol_m_limit = 100 #30 #5
            self.std_std_prc_cvol_m_limit = 2000  # 250, test, 150 #(org)
            self.prc_s_limit = 100 #50  #30, 200
            self.cvol_m_start = 0.02
            self.cvol_m_start_count = 1
            self.ai_long_low = 0.25 #1.5
            self.rsi_limit_high = 70 #54
            self.rsi_limit_low = 30  #46
            # self.dxy_std_cri = 2.5  # no_use
            # self.count_m_act = 10  #self.count_m_act_ave substitute
            # self.count_m_deact = 5  # no_use
            # self.count_m_overact = 25 # no_use
            # self.dt_overact = self.count_m_overact / 2
            # self.cvol_t_act = 5000
            # self.cvol_t_low_act = 2000
            # self.cvol_s_act = 5
            # self.cvol_s_low_act = 3
            # self.dxy_200_medi_bottom = 250
            self.fee_rate = 0.00018 * 2
            self.profit_min_tick = 25  # profit-band decide
            self.loss_max_tick = 80
        # elif self.which_market == 2:  # UPBIT
        #     # self.prc_std_per_cri = 0.09 / 100
        #     self.mode = 0
        #     self.tick = 1
        #     self.cri_tick = 3
        #     self.loop = 0.5
        #     self.cvol_m_cri = 0.4
        #     self.std_prc_cri = 0.1
        #     # self.dxy_std_cri = 3
        #     # self.count_m_act = 10 * self.factor
        #     # self.count_m_deact = 3 * self.factor
        #     # self.count_m_overact = 25 * self.factor
        #     # self.dt_overact = self.count_m_overact / 2 * self.factor
        #     # self.cvol_t_act = 15 * self.factor
        #     # self.cvol_t_low_act = 5 * self.factor
        #     # self.cvol_s_act = 0.3 * self.factor
        #     # self.cvol_s_low_act = 0.05 * self.factor
        #     # self.dxy_200_medi_bottom = 5 * self.factor
        #     self.fee_rate = 0.0005 * 2
        #     self.profit_min_tick = 5
        #     self.loss_max_tick = 40
        elif self.which_market == 3:  # Kospi
            # self.prc_std_per_cri = 0.09 / 100
            self.mode = 0
            self.tick = 0.05
            self.cri_tick = 1
            self.loop = 0.5
            self.cvol_m_cri = 10
            # self.std_prc_cri = 0.1  # cri of std_(std_prc)
            self.cvol_m_limit = 0.02 #0.0075(23.8.11)
            self.std_prc_cvol_m_limit = 5 #5(23.8.11)
            self.std_std_prc_cvol_m_limit = 1.75 #1.25(24.1.16)
            self.prc_s_limit = 0.2 #0.2
            self.cvol_m_start = 0.02
            self.cvol_m_start_count = 1
            self.ai_long_low = 1.5
            self.rsi_limit_high = 54
            self.rsi_limit_low = 46
            # self.dxy_std_cri = 2.5
            # self.count_m_act = 10
            # self.count_m_deact = 5
            # self.count_m_overact = 20
            # self.dt_overact = self.count_m_overact / 2
            # self.cvol_t_act = 10000
            # self.cvol_t_low_act = 3000
            # self.cvol_s_act = 10
            # self.cvol_s_low_act = 5
            # self.dxy_200_medi_bottom = 250
            self.fee_rate = 0.00003 * 2
            self.profit_min_tick = 2
            self.loss_max_tick = 15
        elif self.which_market == 4:  # e-mini
            # self.prc_std_per_cri = 0.045 / 100
            self.mode = 0
            self.tick = 0.25
            self.cri_tick = 1
            self.loop = 1
            self.cvol_m_cri = 20
            # self.std_prc_cri = 0.4
            self.cvol_m_limit = 0.015  # 0.025
            self.std_prc_cvol_m_limit = 100
            self.std_std_prc_cvol_m_limit = 30
            self.prc_s_limit = 5
            self.cvol_m_start = 0.02
            self.cvol_m_start_count = 1
            self.ai_long_low = 0.5
            self.rsi_limit_high = 54
            self.rsi_limit_low = 46
            # self.dxy_std_cri = 100
            # self.count_m_act = 20
            # self.count_m_deact = 5
            # self.count_m_overact = 20
            # self.dt_overact = self.count_m_overact / 2
            # self.cvol_t_act = 10000
            # self.cvol_t_low_act = 3000
            # self.cvol_s_act = 10
            # self.cvol_s_low_act = 5
            # self.dxy_200_medi_bottom = 250
            self.fee_rate = 0.00007 * 2
            self.profit_min_tick = 3
            self.loss_max_tick = 15
        # elif self.which_market == 5:  # A50
        #     # self.prc_std_per_cri = 0.045 / 100
        #     self.mode = 0
        #     self.tick = 2.5
        #     self.cri_tick = 7.5
        #     self.loop = 1
        #     self.cvol_m_cri = 20
        #     self.std_prc_cri = 0.1
        #     # self.dxy_std_cri = 0.5
        #     # self.count_m_act = 20
        #     # self.count_m_deact = 5
        #     # self.count_m_overact = 20
        #     # self.dt_overact = self.count_m_overact / 2
        #     # self.cvol_t_act = 10000
        #     # self.cvol_t_low_act = 3000
        #     # self.cvol_s_act = 10
        #     # self.cvol_s_low_act = 5
        #     # self.dxy_200_medi_bottom = 250
        #     self.fee_rate = 0.00003 * 2
        #     self.profit_min_tick = 4
        #     self.loss_max_tick = 20
        # elif self.which_market == 6:  # Micro-e-mini
        #     # self.prc_std_per_cri = 0.045 / 100
        #     self.mode = 0
        #     self.tick = 0.25
        #     self.cri_tick = 1.5
        #     self.loop = 1
        #     self.cvol_m_cri = 20
        #     self.std_prc_cri = 0.1
        #     # self.dxy_std_cri = 100
        #     # self.count_m_act = 20
        #     # self.count_m_deact = 5
        #     # self.count_m_overact = 20
        #     # self.dt_overact = self.count_m_overact / 2
        #     # self.cvol_t_act = 10000
        #     # self.cvol_t_low_act = 3000
        #     # self.cvol_s_act = 10
        #     # self.cvol_s_low_act = 5
        #     # self.dxy_200_medi_bottom = 250
        #     self.fee_rate = 0.00003 * 2
        #     self.profit_min_tick = 3
        #     self.loss_max_tick = 15
        # elif self.which_market == 7:  # HS
        #     # self.prc_std_per_cri = 0.045 / 100
        #     self.mode = 0
        #     self.tick = 1
        #     self.cri_tick = 5
        #     self.loop = 1
        #     self.cvol_m_cri = 20
        #     self.std_prc_cri = 0.1
        #     # self.dxy_std_cri = 0.5
        #     # self.count_m_act = 20
        #     # self.count_m_deact = 5
        #     # self.count_m_overact = 20
        #     # self.dt_overact = self.count_m_overact / 2
        #     # self.cvol_t_act = 10000
        #     # self.cvol_t_low_act = 3000
        #     # self.cvol_s_act = 10
        #     # self.cvol_s_low_act = 5
        #     # self.dxy_200_medi_bottom = 250
        #     self.fee_rate = 0.00003 * 2
        #     self.profit_min_tick = 4
        #     self.loss_max_tick = 20

        self.demo = "real"
        if self.auto_cover == 0:
            self.stat_in_org = "000"
            self.stat_out_org = "111"
        self.OrgOrdNo = []
        self.OrgOrdNo_Cov = []
        self.inp = 0
        self.last_o = 0
        self.exed_qty = 0
        self.exed_qty_adj = 0
        self.ave_prc = 0
        self.profit = 0
        self.Profit = 0
        self.startime = time.time()
        self.chkCover = 1
        self.chkCoverSig_2 = 0
        self.chkInit_5 = 0
        self.cvol_c_in = 0
        self.chkForb = 0
        self.Indep = 1
        self.price_trend = 0.5
        self.price_trend_s = 0.5
        self.last_trend = 0.5
        self.now_trend = 0.5

        self.prc_per_10 = 0
        self.prc_per_20 = 0
        self.prc_per_80 = 0
        self.prc_per_90 = 0

        self.idx_1 = "rsi"
        self.idx_2 = "rsi"
        self.idx_3 = "rsi"
        self.idx_4 = "rsi"

        self.in_str_1 = 0
        self.in_str = 0
        self.in_signal_prc = 0
        self.in_sig_profit = 0
        self.out_sig_price = 0
        self.in_sig_price = 0
        self.piox = 0
        self.OrgMain = 'n'
        self.sum_peak = 0
        self.std_prc_slope = 0
        self.std_prc_peak = 0
        self.std_prc_peak_1000 = 0

        # self.AvePrc = 0
        # self.ExedQty = 0
        self.add_count = 1
        self.minus_count = 1
        self.add_limit = 6
        self.turnover = 0
        self.prf_sum = 0
        self.nowprf = 0
        self.prf = 0
        self.prf_able = 0
        self.prf_hit = 0
        self.prf_hit_inv = 0
        self.add = 0
        self.Add_Prf = 0
        self.minus = 0
        self.new_signal = 0
        self.old_signal = 0
        self.add_signal = 0
        self.cover_signal = 0
        self.cover_signal_c = 0
        self.cover_ordered = 0
        self.cover_order_exed = 0
        self.cover_by_opt = 0
        self.cover_in_prc = 0
        self.cover_in_nf = 0
        self.cover_in_time = 0
        self.cover_out_prc = 0
        # self.sys_force_out = 0
        self.profit_opt = 0
        self.prf_cover = 0
        self.last_cover_prc = 0
        self.test_signal = 0
        self.test_signal_mode = 0
        self.std_prc = 0
        self.prc_s = 0
        self.gold1 = 0
        self.gold2 = 0
        self.rsi_gold1 = 0
        self.rsi_gold2 = 0
        self.pvol_gold1 = 0
        self.pvol_gold2 = 0
        self.p1000_gold1 = 0
        self.p1000_gold2 = 0
        self.t_gray = 0
        self.t_gray_strong = 0

        self.check_gold = 0
        self.check_rsi_gold = 0
        self.pvol_rsi_gold = 0
        self.p1000_rsi_gold = 0

        self.OrgMain_new = "n"
        self.dOrgMain_new_s1 = 0
        self.dOrgMain_new_s2 = 0
        self.dOrgMain_new_b1 = 0
        self.dOrgMain_new_b2 = 0
        self.dOrgMain_new_b3 = 0
        self.dOrgMain_new_b4 = 0
        self.dOrgMain_new_bns = 0
        self.dOrgMain_new_bns2 = 0
        self.dOrgMain_new_bns_order = 0
        self.new_bns_mode = 0
        self.nfset_new = 0
        self.inp_new = 0
        self.last_o_new = 0
        self.ave_prc_new = 0
        self.exed_qty_new = 0
        self.type_new = "n"
        self.Profit_new = 0
        self.bns_check_2_lock = 0
        self.lock_price = 0
        self.bns_check_2_last = 0
        self.bns_check_last = 0
        self.bns_check_4_last = 0
        self.gold_last = 0
        self.rsi_last = 0
        self.pvol_last = 0
        self.p1000_last = 0
        self.triple_last_last = 0

        self.check2_pvt_point = 0

        self.reorder_msg = 0

        self.dxy_sig_dmain = 0
        self.signal_set = 0
        self.prc_s_peak = 0
        self.prc_s_peak_abs = 0
        self.cvol_s_peak = 0
        self.cvol_s_peak_2 = 0
        self.add_5 = 0
        self.time_out = 0
        self.pre_in_hit = 0
        self.in_hit = 0
        self.in_hit_touch = 0
        self.init5_touched = 0
        self.peak_touch = 0
        self.add_touch = 0
        self.at_test_changed = 0
        self.at_cover_changed = 0

        self.prc_call = 0
        self.prc_put = 0
        self.prc_c_Shoga1v = 0
        self.prc_c_Bhoga1v = 0
        self.prc_p_Shoga1v = 0
        self.prc_p_Bhoga1v = 0

        self.medi_200_m = 0
        self.medi_200_m_np = 0

        self.dxy_200_cri_sum_p = 0
        self.dxy_200_cri_count_p = 0
        self.dxy_200_cri_ave_p = 0
        self.dxy_200_cri_sum_m = 0
        self.dxy_200_cri_count_m = 0
        self.dxy_200_cri_ave_m = 0

        self.msg_sent = 0
        self.reorder_msg_done = 0
        self.reorder_msg_done_cov = 0
        self.block_doubl_cover_out = 0

        # self.count_m_cri_sum = 0
        # self.count_m_cri_count = 0
        # self.count_m_cri_ave = 0  self.count_m_act_ave

        self.mode_last = 0

        self.cvol_m_peak_ox = 0

        self.cvol_m_cri_sum = 0
        self.cvol_m_cri_count = 0
        self.cvol_m_cri_ave = 0

        self.std_cvol_s = 0

        self.cvol_s_cri_sum_p = 0
        self.cvol_s_cri_count_p = 0
        self.cvol_s_cri_ave_p = 0
        self.cvol_s_cri_sum_m = 0
        self.cvol_s_cri_count_m = 0
        self.cvol_s_cri_ave_m = 0

        self.cvol_t_cri_sum_p = 0
        self.cvol_t_cri_count_p = 0
        self.cvol_t_cri_ave_p = 0
        self.cvol_t_cri_sum_m = 0
        self.cvol_t_cri_count_m = 0
        self.cvol_t_cri_ave_m = 0

        self.count_m_act_sum = 0
        self.count_m_act_count = 0
        self.count_m_act_ave = 0

        self.count_m = 0
        self.count_m_start = 0.02
        self.count_m_start_count = 1

        self.count_m_sig_mode = 0
        self.std_count_m_peak = 0
        self.cover_signal_2 = 0
        self.cover_signal_2_out = 0

        self.init_prc = 0
        self.prc_dev = 0
        self.rsi_init = 0

        self.prc_std_sum = 0
        self.prc_std_count = 0
        self.prc_std_ave = 0

        self.prc_std_1000_sum = 0
        self.prc_std_1000_count = 0
        self.prc_std_1000_ave = 0

        self.std_std_prc_sum = 0
        self.std_std_prc_count = 0
        self.std_std_prc_ave = 0

        self.std_std_prc_cvol_m_max = 0
        self.std_std_prc_cvol_m_min = 0
        self.std_std_prc_cvol_m_peak = 0

        self.sec_15 = int(15 / self.loop)  # = 75  ns, nPXY, stPXY, a~e, ee_s, bump, abump, s1, s2_s, s3, s3_m_m
        self.sec_30 = int(30 / self.loop)  # = 150  mtm, PXYm, stXY, pindex, sXY_s, ee_s_slope, s2_c_m, s3_c, s3_m_short
        self.min_1 = int(
            60 / self.loop)  # = 300  ststPXY, pindex2, ee_s_ave, ee_s_ox, s3_m_m, dt_main1,2, org_in_2, cri, cri_r, ee_s_cri
        self.min_3 = int(180 / self.loop)  # = 900  ee_s_ave_long
        self.min_5 = int(300 / self.loop)
        print('init Nprob', self.nf)
        # a = pd.read_csv("index_mex.csv").columns.values.tolist()
        a = pd.read_csv("index_x.csv").columns.values.tolist()
        self.aa = a
        self.df = pd.DataFrame()
        self.df = pd.DataFrame(index=range(0, 1), columns=a)
        self.no = 0
        self.hist = pd.DataFrame()
        self.hist = pd.DataFrame(index=range(0, 1),
                                 columns=["no", "nf", "time", "bns", 'prc', 'prc_o', 'ap', 'qty', 'type', 'profit',
                                          'prf_opt', 'mode'])

        # AI

        self.ai_ = pd.read_csv("index_ai_sshort.csv").columns.values.tolist()
        self.ai_x = ['cvolume', 'std_prc', 'prc_s', 'rsi', 'count', 'cvol_c', 'cvol_m', 'cvol_s', 'cvol_t', 'y_1',
                     'y_2', 'y_3']
        self.ai_x_2 = ['prc_std_1000', 'dxy_med_200_s', 'std_prc', 'std_prc_1000', 'prc_s', 'cvol_c', 'count_m_per_5',
                       'cvol_m_ave', 'cvol_s', 'cvol_t']
        self.ai_x_3 = ['dxx', 'dyy', 'dxy', 'nPX', 'nPY', 'dxy_med_200_s', 'dxy_20_medi', 'cvol_c', 'count_m',
                       'cvol_m_ave', 'cvol_s', 'cvol_t']
        self.ai_x_bns = ['prc_std_1000', 'dxy_med_200_s', 'std_prc', 'std_prc_1000', 'prc_s', 'cvol_c', 'count_m_per_5',
                         'cvol_m_ave', 'cvol_s', 'cvol_t', 'dOrgMain_new_bns2']
        self.ai_x_bns = ['prc_std_1000', 'cover_signal_2', 'std_prc', 'std_prc_1000', 'prc_s', 'cvol_c',
                         'count_m_per_5', 'cvol_m_ave', 'cvol_s', 'ai', 'ai_long']
        self.time_span_lstm = 10
        self.time_span_lstm_c = 20  #
        self.time_span_lstm_alt = 50
        nb_features = len(self.ai_)
        self.force_on = 0
        self.force = 0

        if 1 == 0:
            self.model_lstm = Sequential()
            self.model_lstm.add(LSTM(64, input_shape=(self.time_span_lstm, nb_features), activation='relu'))
            # self.model_lstm.add(Dropout(0.2))
            self.model_lstm.add(Dense(32, activation='relu'))
            # self.model_lstm.add(Dense(32, activation='sigmoid'))
            self.model_lstm.add(Dense(16, activation='sigmoid'))
            self.model_lstm.add(Dense(6, activation='softmax'))
            # self.model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            pass

        # BNS mode
        if self.ai_mode == 1 or self.ai_mode == 3:

            # primary model(rsi포함)
            if self.which_market == 1:
                json_file = open('model_1.json', 'r')
            if self.which_market == 3:
                json_file = open('model_3.json', 'r')
            if self.which_market == 4:
                json_file = open('model_4.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.loaded_model = model_from_json(loaded_model_json)
            if self.which_market == 1:
                self.loaded_model.load_weights("model_1.h5")
            if self.which_market == 4:
                self.loaded_model.load_weights("model_4.h5")
            if self.which_market == 3:
                self.loaded_model.load_weights("model_3.h5")
            print("BNS model")
            self.loaded_model.summary()

            # secondary model(rsi미포함)
            if 1 == 0:
                if self.which_market == 1:
                    json_file_2 = open('model_1_2.json', 'r')
                if self.which_market == 3:
                    json_file_2 = open('model_3_2.json', 'r')
                if self.which_market == 4:
                    json_file_2 = open('model_4_2.json', 'r')
                loaded_model_json_2 = json_file_2.read()
                json_file_2.close()
                self.loaded_model_2 = model_from_json(loaded_model_json_2)
                if self.which_market == 1:
                    self.loaded_model_2.load_weights("model_1_2.h5")
                if self.which_market == 4:
                    self.loaded_model_2.load_weights("model_4_2.h5")
                if self.which_market == 3:
                    self.loaded_model_2.load_weights("model_3_2.h5")
                print("BNS_2 model")
                self.loaded_model_2.summary()

            # third model(y변경 2차미분)
            if self.which_market == 1:
                json_file_3 = open('model_1_3.json', 'r')
            if self.which_market == 3:
                json_file_3 = open('model_3_3.json', 'r')
            if self.which_market == 4:
                json_file_3 = open('model_4_3.json', 'r')
            loaded_model_json_3 = json_file_3.read()
            json_file_3.close()
            self.loaded_model_3 = model_from_json(loaded_model_json_3)
            if self.which_market == 1:
                self.loaded_model_3.load_weights("model_1_3.h5")
            if self.which_market == 4:
                self.loaded_model_3.load_weights("model_4_3.h5")
            if self.which_market == 3:
                self.loaded_model_3.load_weights("model_3_3.h5")
            print("BNS_3 model")
            self.loaded_model_3.summary()

            # bns2 mode1
            if 1 == 0:
                if self.which_market == 1:
                    json_file_bns = open('model_bns_1.json', 'r')
                if self.which_market == 3:
                    json_file_bns = open('model_bns_3.json', 'r')
                if self.which_market == 4:
                    json_file_bns = open('model_bns_4.json', 'r')
                loaded_model_bns_json = json_file_bns.read()
                json_file_bns.close()
                self.loaded_model_bns = model_from_json(loaded_model_bns_json)
                if self.which_market == 1:
                    self.loaded_model_bns.load_weights("model_bns_1.h5")
                if self.which_market == 4:
                    self.loaded_model_bns.load_weights("model_bns_4.h5")
                if self.which_market == 3:
                    self.loaded_model_bns.load_weights("model_bns_3.h5")
                print("BNS_bns model")
                self.loaded_model_bns.summary()

            # bns_ai mode1
            if 1 == 0:
                if self.which_market == 1:
                    json_file_bns_ai = open('model_bns_ai_1.json', 'r')
                if self.which_market == 3:
                    json_file_bns_ai = open('model_bns_ai_3.json', 'r')
                if self.which_market == 4:
                    json_file_bns_ai = open('model_bns_ai_4.json', 'r')
                loaded_model_bns_ai_json = json_file_bns_ai.read()
                json_file_bns_ai.close()
                self.loaded_model_bns_ai = model_from_json(loaded_model_bns_ai_json)
                if self.which_market == 1:
                    self.loaded_model_bns_ai.load_weights("model_bns_ai_1.h5")
                if self.which_market == 4:
                    self.loaded_model_bns_ai.load_weights("model_bns_ai_4.h5")
                if self.which_market == 3:
                    self.loaded_model_bns_ai.load_weights("model_bns_ai_3.h5")
                print("BNS_bns_ai model")
                self.loaded_model_bns_ai.summary()

            # Conversion model : load
            if 1 == 0:
                if self.which_market == 1 or self.which_market == 4:
                    json_file = open('model.json', 'r')
                if self.which_market == 3:
                    json_file_c = open('model_c_3.json', 'r')
                loaded_model_json_c = json_file_c.read()
                json_file_c.close()
                self.loaded_model_c = model_from_json(loaded_model_json_c)
                if self.which_market == 1:
                    self.loaded_model_c.load_weights("model_1_2.h5")
                if self.which_market == 4:
                    self.loaded_model_c.load_weights("model_4_2.h5")
                if self.which_market == 3:
                    self.loaded_model_c.load_weights("model_c_3.h5")
                print("Conversion_load model")
                self.loaded_model_c.summary()

            # Conversion model : create
            if 1 == 1:  # (0)model (model_lstm_c)  cate = 3
                self.model_lstm_c = Sequential()
                self.model_lstm_c.add(
                    LSTM(64, input_shape=(20, 12), activation='relu'))  # time_span_lstm=10, nb_features=12
                # self.model_lstm_c.add(Dropout(0.2))
                self.model_lstm_c.add(Dense(32, activation='relu'))
                # self.model_lstm_c.add(Dense(32, activation='sigmoid'))
                self.model_lstm_c.add(Dense(16, activation='sigmoid'))
                self.model_lstm_c.add(Dense(3, activation='softmax'))
                self.model_lstm_c.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                self.model_lstm_c.load_weights("model_c_" + str(self.which_market) + ".h5")
                print("Conversion model")
                self.model_lstm_c.summary()

            if 1 == 1:  # (1)model (model_lstm_c_1)  : main cate = 2
                self.model_lstm_c_1 = Sequential()
                self.model_lstm_c_1.add(LSTM(64, input_shape=(20, 12), activation='relu'))  # time_span_lstm=10,
                # nb_features=12
                # self.model_lstm_c_1.add(Dropout(0.2))
                self.model_lstm_c_1.add(Dense(32, activation='relu'))
                # self.model_lstm_c_1.add(Dense(32, activation='sigmoid'))
                self.model_lstm_c_1.add(Dense(16, activation='sigmoid'))
                self.model_lstm_c_1.add(Dense(2, activation='softmax'))
                self.model_lstm_c_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                self.model_lstm_c_1.load_weights("model_c_1_" + str(self.which_market) + ".h5")
                print("Conversion_1 model")
                self.model_lstm_c_1.summary()

            if 1 == 1:  # (1)model (model_lstm_c_alt) : alternative cate = 2
                self.model_lstm_c_alt = Sequential()
                self.model_lstm_c_alt.add(LSTM(64, input_shape=(50, 12), activation='relu'))  # time_span_lstm=10,
                # nb_features=12
                # self.model_lstm_c_alt.add(Dropout(0.2))
                self.model_lstm_c_alt.add(Dense(32, activation='relu'))
                # self.model_lstm_c_alt.add(Dense(32, activation='sigmoid'))
                self.model_lstm_c_alt.add(Dense(16, activation='sigmoid'))
                self.model_lstm_c_alt.add(Dense(2, activation='softmax'))
                self.model_lstm_c_alt.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                                              metrics=['accuracy'])
                self.model_lstm_c_alt.load_weights("model_c_alt_" + str(self.which_market) + ".h5")
                print("Conversion_alt model")
                self.model_lstm_c_alt.summary()

            if 1 == 1:  # (1)model (model_lstm_c_alt) : alternative cate = 2
                self.model_lstm_c_alt2 = Sequential()
                self.model_lstm_c_alt2.add(LSTM(64, input_shape=(50, 12), activation='relu'))  # time_span_lstm=10,
                # nb_features=12
                # self.model_lstm_c_alt2.add(Dropout(0.2))
                self.model_lstm_c_alt2.add(Dense(32, activation='relu'))
                # self.model_lstm_c_alt2.add(Dense(32, activation='sigmoid'))
                self.model_lstm_c_alt2.add(Dense(16, activation='sigmoid'))
                self.model_lstm_c_alt2.add(Dense(2, activation='softmax'))
                self.model_lstm_c_alt2.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                                               metrics=['accuracy'])
                self.model_lstm_c_alt2.load_weights("model_c_alt2_" + str(self.which_market) + ".h5")
                print("Conversion_alt2 model")
                self.model_lstm_c_alt2.summary()

        # LONG mode
        if self.ai_mode == 2 or self.ai_mode == 3:
            if 1 == 0:  # json
                if self.which_market == 1 or self.which_market == 4:
                    json_file_long = open('model.json', 'r')
                if self.which_market == 3:
                    json_file_long = open('model_long_3.json', 'r')
                loaded_long_model_json = json_file_long.read()
                json_file_long.close()
                self.loaded_long_model = model_from_json(loaded_long_model_json)
                if self.which_market == 1 or self.which_market == 4:
                    self.loaded_long_model.load_weights("model_long.h5")
                if self.which_market == 3:
                    self.loaded_long_model.load_weights("model_long_3.h5")
            if 1 == 1:  # h5 file
                if self.which_market == 1:
                    # json_file_long = open('model.json', 'r')
                    self.loaded_long_model = load_model('model_long_1.h5')
                if self.which_market == 4:
                    # json_file_long = open('model.json', 'r')
                    self.loaded_long_model = load_model('model_long_4.h5')
                if self.which_market == 3:
                    # json_file_long = open('model_long_3.json', 'r')
                    self.loaded_long_model = load_model('model_long_3.h5')
            print("BNS_long model")
            self.loaded_long_model.summary()

        if 1 == 1:
            self.para = pd.DataFrame()
            cols = ["count_m", "cvol_s_p", "cvol_s_m", "cvol_t_p", "cvol_t_m", "cvol_m", "dxy_200_p", "dxy_200_m",
                    "prc_std", "prc_std_1000", "std_std_prc"]
            self.para = pd.DataFrame(index=range(0, 1), columns=cols)
            if self.which_market == 4:
                self.para_pre = pd.read_csv("para_emini.csv")
            elif self.which_market == 5:
                self.para_pre = pd.read_csv("para_a50.csv")
            elif self.which_market == 6:
                self.para_pre = pd.read_csv("para_micro.csv")
            elif self.which_market == 7:
                self.para_pre = pd.read_csv("para_hs.csv")
            elif self.which_market == 1:
                self.para_pre = pd.read_csv("para_bit.csv")
            else:
                self.para_pre = pd.read_csv("para.csv")
            self.para_pre = self.para_pre[cols]
            print("self.para_pre")
            print(self.para_pre)
            self.start_data_index = self.para_pre[self.para_pre.count_m != 0].index[-1]
            print("### self.start_data_index: ", self.start_data_index)
            print('====================')

            self.cvol_m_cri_ave = self.para_pre.loc[1:self.start_data_index, "cvol_m"].mean()
            self.cvol_s_cri_ave_p = self.para_pre.loc[1:self.start_data_index, "cvol_s_p"].mean()
            self.cvol_s_cri_ave_m = self.para_pre.loc[1:self.start_data_index, "cvol_s_m"].mean()
            self.cvol_t_cri_ave_p = self.para_pre.loc[1:self.start_data_index, "cvol_t_p"].mean()
            self.cvol_t_cri_ave_m = self.para_pre.loc[1:self.start_data_index, "cvol_t_m"].mean()
            self.dxy_200_cri_ave_p = self.para_pre.loc[1:self.start_data_index, "dxy_200_p"].mean()
            self.dxy_200_cri_ave_m = self.para_pre.loc[1:self.start_data_index, "dxy_200_m"].mean()
            self.count_m_act_ave = self.para_pre.loc[1:self.start_data_index, "count_m"].mean()
            self.prc_std_ave = self.para_pre.loc[1:self.start_data_index, "prc_std"].mean()
            self.prc_std_1000_ave = self.para_pre.loc[1:self.start_data_index, "prc_std_1000"].mean()
            self.std_std_prc_ave = self.para_pre.loc[1:self.start_data_index, "std_std_prc"].mean()

            self.cvol_m_cri_sum = self.cvol_m_cri_ave * self.start_data_index
            self.cvol_m_cri_count = self.start_data_index
            print("self.cvol_m_cri_sum", self.cvol_m_cri_sum)
            print("self.cvol_m_cri_count", self.cvol_m_cri_count)
            print("self.cvol_m_cri_ave", self.cvol_m_cri_ave)

            self.cvol_s_cri_sum_p = self.cvol_s_cri_ave_p * self.start_data_index
            self.cvol_s_cri_count_p = self.start_data_index
            self.cvol_s_cri_sum_m = self.cvol_s_cri_ave_m * self.start_data_index
            self.cvol_s_cri_count_m = self.start_data_index
            print("self.cvol_s_cri_sum_p", self.cvol_s_cri_sum_p)
            print("self.cvol_s_cri_sum_m", self.cvol_s_cri_sum_m)

            self.cvol_t_cri_sum_p = self.cvol_t_cri_ave_p * self.start_data_index
            self.cvol_t_cri_count_p = self.start_data_index
            self.cvol_t_cri_sum_m = self.cvol_t_cri_ave_m * self.start_data_index
            self.cvol_t_cri_count_m = self.start_data_index
            print("self.cvol_t_cri_sum_p", self.cvol_t_cri_sum_p)
            print("self.cvol_t_cri_sum_m", self.cvol_t_cri_sum_m)

            self.dxy_200_cri_sum_p = self.dxy_200_cri_ave_p * self.start_data_index
            self.dxy_200_cri_count_p = self.start_data_index
            self.dxy_200_cri_sum_m = self.dxy_200_cri_ave_m * self.start_data_index
            self.dxy_200_cri_count_m = self.start_data_index
            print("self.dxy_200_cri_sum_p", self.dxy_200_cri_sum_p)
            print("self.dxy_200_cri_sum_m", self.dxy_200_cri_sum_m)

            self.count_m_act_sum = self.count_m_act_ave * self.start_data_index
            self.count_m_act_count = self.start_data_index
            print("self.count_m_act_sum", self.count_m_act_sum)

            self.prc_std_sum = self.prc_std_ave * self.start_data_index
            self.prc_std_count = self.start_data_index
            print("self.prc_std_sum", self.prc_std_sum)

            self.prc_std_1000_sum = self.prc_std_1000_ave * self.start_data_index
            self.prc_std_1000_count = self.start_data_index
            print("self.prc_std_1000_sum", self.prc_std_1000_sum)

            self.std_std_prc_sum = self.std_std_prc_ave * self.start_data_index
            self.std_std_prc_count = self.start_data_index
            print("self.std_std_prc_sum", self.std_std_prc_sum)

            # self.para_start = para_data.loc[self.start_data_index]
            # print("para_start", self.para_start)

        print(self.df)
        print(self.hist)

        # self.lstm()

    def nprob(self, price, timestamp, mt, count, cgubun_sum, cvolume_sum, volume, lblSqty2v, lblSqty1v, lblShoga1v,
              lblBqty1v, lblBhoga1v, lblBqty2v, prc_o1):  # lblShoga2v,, lblBhoga2v

        # cvolume_sum = cvolume_sum * self.cvol_adj

        t_start = time.time()
        now = datetime.now()
        self.df.at[self.nf, "nf"] = self.nf
        self.df.at[self.nf, "prc_call"] = self.prc_call
        self.df.at[self.nf, "prc_put"] = self.prc_put
        self.df.at[self.nf, "prc_c_Shoga1v"] = self.prc_c_Shoga1v
        self.df.at[self.nf, "prc_c_Bhoga1v"] = self.prc_c_Bhoga1v
        self.df.at[self.nf, "prc_p_Shoga1v"] = self.prc_p_Shoga1v
        self.df.at[self.nf, "prc_p_Bhoga1v"] = self.prc_p_Bhoga1v

        # if ((now.hour == 14 and now.minute >= 32) or now.hour == 15) and self.cover_ordered == 0 and self.Forb_setted_sent == 0:
        #     bot1.sendMessage(chat_id="322233222", text=str(self.auto_cover) + " / chkForb setted...cover_cleared...@2:32(nprob)" + str(self.profit_opt))
        #     self.Forb_setted_sent = 1

        print('###########')
        print('nf: %d  //prc: %0.2f/ /in: %d /out: %0.1f /prf: %d /last_o %0.2f  /turn: %d' % (
            self.nf, price, self.in_str, self.piox, self.prf_able, self.last_o, self.turnover))

        if self.nf != 0 and ((self.auto_cover == 1 and self.nf % 2000 == 0) or (self.auto_cover == 2 and self.nf % 2100 == 0)):

            try:
                self.start_data_index += 1
                self.para.at[self.start_data_index, "cvol_m"] = self.cvol_m_cri_ave
                self.para.at[self.start_data_index, "cvol_s_p"] = self.cvol_s_cri_ave_p
                self.para.at[self.start_data_index, "cvol_s_m"] = self.cvol_s_cri_ave_m
                self.para.at[self.start_data_index, "cvol_t_p"] = self.cvol_t_cri_ave_p
                self.para.at[self.start_data_index, "cvol_t_m"] = self.cvol_t_cri_ave_m
                self.para.at[self.start_data_index, "dxy_200_p"] = self.dxy_200_cri_ave_p
                self.para.at[self.start_data_index, "dxy_200_m"] = self.dxy_200_cri_ave_m
                self.para.at[self.start_data_index, "count_m"] = self.count_m_act_ave
                self.para.at[self.start_data_index, "prc_std"] = self.prc_std_ave
                self.para.at[self.start_data_index, "prc_std_1000"] = self.prc_std_1000_ave
                self.para.at[self.start_data_index, "std_std_prc"] = self.std_std_prc_ave

                self.para_after = self.para_pre.append(self.para[1:])
                print(self.para_after)
                if self.which_market == 4:
                    filename_para = "para_emini.csv"
                else:
                    filename_para = "para.csv"
                print('###### saving..')
                self.para_after.to_csv(filename_para)
            except:
                print("2000 error")

            self.btnSave_Clicked()

        if self.which_market == 3:
            if self.auto_cover == 1:
                if now.hour == 15 and now.minute == 31 and self.last_save != 1:
                    self.btnSave_Clicked()
            if self.auto_cover == 2:
                if now.hour == 15 and now.minute == 32 and self.last_save != 1:
                    self.btnSave_Clicked()

        regr = linear_model.LinearRegression()

        ###############################
        # Raw Data
        ###############################

        self.df.at[self.nf, "price"] = price
        self.df.at[self.nf, "cgubun"] = cgubun_sum
        self.df.at[self.nf, "cvolume"] = cvolume_sum
        self.df.at[self.nf, "volume"] = volume
        self.df.at[self.nf, "y2"] = lblSqty2v
        self.df.at[self.nf, "y1"] = lblSqty1v
        self.df.at[self.nf, "py1"] = lblShoga1v
        self.df.at[self.nf, "x1"] = lblBqty1v
        self.df.at[self.nf, "px1"] = lblBhoga1v
        self.df.at[self.nf, "x2"] = lblBqty2v

        # init_prc
        if self.nf == 200:
            self.init_prc = self.df.loc[self.nf - 200:self.nf - 1, "px1"].mean()

        if self.nf > 200:
            try:
                self.idx_1_data = self.df.at[self.nf - 1, self.idx_1]
                if self.nf > 1000:
                    self.idx_2_data = self.df.at[self.nf - 1, self.idx_2]
                else:
                    self.idx_2_data = self.idx_1_data
                self.idx_3_data = self.df.at[self.nf - 1, self.idx_3]
                self.idx_4_data = self.df.at[self.nf - 1, self.idx_4]
            except:
                pass

        # if self.nf > 30:
        #     ma=idx.moving_average(self.df,14)
        #     self.df.at[self.nf, "MA_14"] = ma.at[len(ma)-2, "MA_14"]

        # prc_avg
        if self.nf < self.min_1 + 11:
            prc_avg = lblBhoga1v
            if self.nf >= 1:
                if self.df.at[self.nf - 1, "prc_avg"] == 0:
                    self.df.at[self.nf - 1, "prc_avg"] = prc_avg
        if self.nf >= self.min_1 + 11:
            prc_avg = self.df.loc[self.nf - min(500, self.nf):self.nf - 1, "px1"].mean()

        self.df.at[self.nf, "prc_avg"] = prc_avg

        # prc_avg_1000
        if self.nf < self.min_1 + 11:
            prc_avg_1000 = lblBhoga1v
        if self.nf >= self.min_1 + 11:
            prc_avg_1000 = self.df.loc[self.nf - min(1000, self.nf):self.nf - 1, "px1"].mean()
        self.df.at[self.nf, "prc_avg_1000"] = prc_avg_1000

        # prc_std  @1000
        # if self.nf < 1001 or prc_avg_1000 == 0:
        #     prc_std_1000 = self.prc_std_1000_ave
        # if self.nf >= 1001 and prc_avg_1000 != 0:
        # prc_std_1000 = self.df.loc[self.nf - min(1000, self.nf):self.nf - 1, "px1"].std()
        if self.nf >= 101:  # and prc_avg != 0:
            prc_std_1000 = self.df.loc[self.nf - min(1000, self.nf):self.nf - 1, "px1"].std() * (
                        1000 / min(self.nf, 1000)) ** 0.5
        else:
            prc_std_1000 = self.prc_std_1000_ave
        self.df.at[self.nf, "prc_std_1000"] = prc_std_1000

        ######## prc_std
        # if self.nf < 501 or prc_avg == 0:
        #     prc_std = self.prc_std_ave
        if self.nf >= 11:  # and prc_avg != 0:
            prc_std = self.df.loc[self.nf - min(500, self.nf) + 1:self.nf - 1, "price"].std() * (
                        1000 / min(self.nf, 500)) ** 0.5
            # print("//prc_std_org", self.df.loc[self.nf - min(500, self.nf)+1:self.nf - 1, "price"].std())
            # print("//adj", (1000 / min(self.nf,500))count_m ** 0.5)
            # print("//test", self.nf - min(500, self.nf))
        else:
            prc_std = self.prc_std_ave
        self.df.at[self.nf, "prc_std"] = prc_std

        # prc_std  @100
        # if self.nf < 101 or prc_avg == 0:
        #     prc_std_100 = 0
        # if self.nf >= 101 and prc_avg != 0:
        #     prc_std_100 = self.df.loc[self.nf - min(100, self.nf):self.nf - 1, "px1"].std()
        if self.nf >= 101:  # and prc_avg != 0:
            prc_std_100 = self.df.loc[self.nf - min(100, self.nf):self.nf - 1, "px1"].std() * (
                        1000 / min(self.nf, 100)) ** 0.5
        else:
            prc_std_100 = 0
        self.df.at[self.nf, "prc_std_100"] = prc_std_100

        # EMA_prc_std
        if self.nf < 400 + 11:
            ema_20_prc_std = 0
            ema_50_prc_std = 0
            ema_200_prc_std = 0
        if self.nf >= 400 + 11:
            if self.talib == 1:
                ema_20_prc_std = talib.EMA(np.array(self.df['prc_std'], dtype=float), 20)[-1]
                ema_50_prc_std = talib.EMA(np.array(self.df['prc_std'], dtype=float), 50)[-1]
                ema_200_prc_std = talib.EMA(np.array(self.df['prc_std'], dtype=float), 200)[-1]

            if self.talib == 0:
                ema_20_prc_std = cal_ema(pd.Series(self.df['prc_std']), window=20).iloc[-1]
                ema_50_prc_std = cal_ema(pd.Series(self.df['prc_std']), window=50).iloc[-1]
                ema_200_prc_std = cal_ema(pd.Series(self.df['prc_std']), window=200).iloc[-1]

        self.df.at[self.nf, "ema_20_prc_std"] = ema_20_prc_std
        self.df.at[self.nf, "ema_50_prc_std"] = ema_50_prc_std
        self.df.at[self.nf, "ema_200_prc_std"] = ema_200_prc_std

        ema_25_prc_std = 0
        if ema_20_prc_std > ema_50_prc_std:
            ema_25_prc_std = 1
        if ema_20_prc_std < ema_50_prc_std:
            ema_25_prc_std = -1
        self.df.at[self.nf, "ema_25_prc_std"] = ema_25_prc_std

        ema_520_prc_std = 0
        if ema_50_prc_std > ema_200_prc_std:
            ema_520_prc_std = 1
            # if self.nf > self.sec_30 * 4 + 2:
            if self.df.at[self.nf - 1, "prc_s"] > 0 and self.test_signal != -3:
                if self.which_market == 1 and prc_std > 10:
                    ema_520_prc_std = 2
                if self.which_market == 2 and prc_std > 30:
                    ema_520_prc_std = 2
                if self.which_market == 3 and prc_std > 0.2:
                    ema_520_prc_std = 2
                if self.which_market == 4 and prc_std > 0.6:
                    ema_520_prc_std = 2
            if self.df.at[self.nf - 1, "prc_s"] < 0 and self.test_signal != 3:
                if self.which_market == 1 and prc_std > 10:
                    ema_520_prc_std = 2
                if self.which_market == 2 and prc_std > 30:
                    ema_520_prc_std = 2
                if self.which_market == 3 and prc_std > 0.2:
                    ema_520_prc_std = 2
                if self.which_market == 4 and prc_std > 0.6:
                    ema_520_prc_std = 2
        if ema_50_prc_std < ema_200_prc_std:
            ema_520_prc_std = -1
            # if self.nf > self.sec_30 * 4 + 2:
            # if self.df.at[self.nf-1, "prc_s"]<0 and self.test_signal != 3:
            #     if self.which_market == 2 and prc_std > 30:
            #         ema_520_prc_std = -2
            #     if self.which_market == 3 and prc_std > 0.2:
            #         ema_520_prc_std = -2
            #     if self.which_market == 4 and prc_std > 0.6:
            #         ema_520_prc_std = -2

        self.df.at[self.nf, "ema_520_prc_std"] = ema_520_prc_std

        # prc_sig
        prc_sig = 0
        if self.nf >= self.min_1 + 1:
            if lblBhoga1v - prc_avg < prc_std * -0.6:
                prc_sig = -1
            if lblBhoga1v - prc_avg > prc_std * 0.6:
                prc_sig = 1
        self.df.at[self.nf, "prc_sig"] = prc_sig

        # prc_in_out_sig
        prc_in_out_sig = 0
        if self.nf >= self.min_1 + 1 and self.df.at[self.nf - 1, "prc_sig"] != self.df.at[self.nf, "prc_sig"]:
            if prc_sig == 1:
                prc_in_out_sig = 1
            if prc_sig == -1:
                prc_in_out_sig = -1
            if prc_sig == 0:
                prc_in_out_sig = 0
        self.df.at[self.nf, "prc_in_out_sig"] = prc_in_out_sig

        ######### std_prc, mean_prc
        if prc_avg != 0 and prc_std != 0:
            self.df.at[self.nf, "mean_prc"] = lblBhoga1v - prc_avg
            std_prc = (lblBhoga1v - prc_avg) / prc_std
        else:
            std_prc = 0
        # if self.nf>=250:
        self.df.at[self.nf, "std_prc"] = std_prc
        self.std_prc = std_prc
        # print("//std_prc// ", std_prc)

        ######### std_prc_1000, mean_prc_1000
        self.std_prc_1000 = 0
        if prc_avg_1000 != 0 and prc_std_1000 != 0:
            self.df.at[self.nf, "mean_prc_1000"] = lblBhoga1v - prc_avg_1000
            std_prc_1000 = (lblBhoga1v - prc_avg_1000) / prc_std_1000
        else:
            std_prc_1000 = 0
        self.std_prc_1000 = std_prc_1000
        self.df.at[self.nf, "std_prc_1000"] = std_prc_1000
        # print('//std_prc_1000// %.3f'%std_prc_1000)

        # std_prc_slope
        if self.nf >= self.sec_30 * 4 + 1:
            d_y = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "std_prc"]
            d_x = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "stime"]
            self.std_prc_slope = regr.fit(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1)).coef_[0][0] * 100000
            # prc_b = regr.score(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1))
            # print("prc_s", prc_s)
            # print("prc_b", prc_b)
        else:
            self.std_prc_slope = 0
            # prc_b = 0
        self.df.at[self.nf, "std_prc_slope"] = self.std_prc_slope

        # std_prc_slope_s
        if self.nf >= self.sec_30 * 8 + 1:
            d_y = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "std_prc_slope"]
            d_x = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "stime"]
            self.std_prc_slope_s = regr.fit(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1)).coef_[0][0] * 100000
            # prc_b = regr.score(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1))
            # print("prc_s", prc_s)
            # print("prc_b", prc_b)
        else:
            self.std_prc_slope_s = 0
            # prc_b = 0
        self.df.at[self.nf, "std_prc_slope_s"] = self.std_prc_slope_s

        # std_prc_slope_200
        if self.nf >= self.sec_30 * 4 + 1:
            d_y = self.df.loc[self.nf - 200:self.nf - 1, "std_prc"]
            d_x = self.df.loc[self.nf - 200:self.nf - 1, "stime"]
            self.std_prc_slope_200 = regr.fit(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1)).coef_[0][0] * 100000
            # prc_b = regr.score(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1))
            # print("prc_s", prc_s)
            # print("prc_b", prc_b)
        else:
            self.std_prc_slope_200 = 0
            # prc_b = 0
        self.df.at[self.nf, "std_prc_slope_200"] = self.std_prc_slope_200

        # std_prc_peak
        if prc_avg != 0 and prc_std != 0:
            if self.std_prc_peak == 0:
                if std_prc > 1.6:
                    if std_prc > 2.1 or abs(self.std_prc_cvol_m) >= self.std_prc_cvol_m_limit * 0.5:  # and self.df.at[self.nf - 1, "cvol_m"] > self.cvol_m_cri_ave:
                        self.std_prc_peak = 2
                    # if self.add_5 != 0:
                    #     self.add_5 = 0
                if std_prc < -1.6:
                    if std_prc > 2.1 or abs(self.std_prc_cvol_m) >= self.std_prc_cvol_m_limit * 0.5:  # and self.df.at[self.nf - 1, "cvol_m"] > self.cvol_m_cri_ave:
                        self.std_prc_peak = -2
                    # if self.add_5 != 0:
                    #     self.add_5 = 0

            if abs(self.std_prc_peak) == 2:
                if std_prc < 1 and std_prc > 0:
                    self.std_prc_peak = 1
                if std_prc > -1 and std_prc < 0:
                    self.std_prc_peak = -1
                if abs(std_prc) <= 0.5:
                    self.std_prc_peak = 0

            if abs(self.std_prc_peak) == 1:
                if abs(std_prc) <= 0.5:
                    self.std_prc_peak = 0
        self.df.at[self.nf, "std_prc_peak"] = self.std_prc_peak

        # std_prc_peak_1000
        if prc_avg_1000 != 0 and prc_std_1000 != 0:
            if self.std_prc_peak_1000 == 0:
                if std_prc_1000 > 1.8:  # and self.df.at[self.nf - 1, "cvol_m"] > self.cvol_m_cri_ave:
                    self.std_prc_peak_1000 = 2
                    # if self.add_5 != 0:
                    #     self.add_5 = 0
                if std_prc_1000 < -1.8:  # and self.df.at[self.nf - 1, "cvol_m"] > self.cvol_m_cri_ave:
                    self.std_prc_peak_1000 = -2
                    # if self.add_5 != 0:
                    #     self.add_5 = 0

            if abs(self.std_prc_peak_1000) == 2:
                if std_prc_1000 < 0.75 and std_prc_1000 > 0:
                    self.std_prc_peak_1000 = 1
                if std_prc_1000 > -0.75 and std_prc_1000 < 0:
                    self.std_prc_peak_1000 = -1
                if abs(std_prc_1000) <= 0.5:
                    self.std_prc_peak_1000 = 0

            if abs(self.std_prc_peak_1000) == 1:
                if abs(std_prc_1000) <= 0.5:
                    self.std_prc_peak_1000 = 0
        self.df.at[self.nf, "std_prc_peak_1000"] = self.std_prc_peak_1000

        ############ std_std_prc
        if self.nf >= 51:  # and std_prc != 0:
            std_std_prc = self.df.loc[self.nf - 20: self.nf - 1, "std_prc"].std()
        else:
            std_std_prc = self.std_std_prc_ave
        self.df.at[self.nf, "std_std_prc"] = std_std_prc

        ####### self.prc_std_per_cri
        self.prc_std_per_cri = self.tick * self.cri_tick / price
        # print("prc_cri : " , self.prc_std_per_cri*100)
        self.df.at[self.nf, "prc_cri"] = self.prc_std_per_cri

        # prc_s
        if self.nf >= self.sec_30 * 4 + 1:
            d_y = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "px1"]
            d_x = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "stime"]
            prc_s = regr.fit(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1)).coef_[0][0] * 100000
            prc_b = regr.score(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1))
            # print("prc_s", prc_s)
            # print("prc_b", prc_b)
        else:
            prc_s = 0
            prc_b = 0
        self.prc_s = prc_s
        self.df.at[self.nf, "prc_s"] = prc_s
        self.df.at[self.nf, "prc_b"] = prc_b

        # prc_s_std
        if self.nf >= self.sec_30 * 5 + 1 and self.nf > 501:
            prc_s_std = self.df.loc[self.nf - 400: self.nf - 1, "prc_s"].std()
        else:
            prc_s_std = 0
        self.df.at[self.nf, "prc_s_std"] = prc_s_std

        # prc_s_peak
        if self.nf >= self.sec_30 + 1 and self.nf > 501:

            if abs(self.prc_s_peak) == 0:
                if prc_s < prc_s_std * 1 and prc_s > 0:
                    self.prc_s_peak = 1
                if prc_s > prc_s_std * -1 and prc_s < 0:
                    self.prc_s_peak = -1

            if self.prc_s_peak == 0 or self.prc_s_peak == 1:
                if prc_s > prc_s_std * 2 and prc_s > self.prc_s_limit * 0.25:
                    self.prc_s_peak = 2
                    if self.add_5 != 0:
                        self.add_5 = 0
            if self.prc_s_peak == 0 or self.prc_s_peak == -1:
                if prc_s < prc_s_std * -2 and prc_s < self.prc_s_limit * -0.25:
                    self.prc_s_peak = -2
                    if self.add_5 != 0:
                        self.add_5 = 0

            if abs(self.prc_s_peak) == 2:
                if prc_s < prc_s_std * 1 and prc_s > 0:
                    self.prc_s_peak = 1
                if prc_s > prc_s_std * -1 and prc_s < 0:
                    self.prc_s_peak = -1
                if abs(prc_s) <= prc_s_std * 0.5:
                    self.prc_s_peak = 0

            if abs(self.prc_s_peak) == 1:
                if abs(prc_s) <= prc_s_std * 0.5:
                    self.prc_s_peak = 0

            #prs_s_peak + bns2
            self.prc_s_peak_abs = self.prc_s_peak
            if self.prc_s_peak > 0 and self.dOrgMain_new_bns2 < 0:
                self.prc_s_peak_abs = 0
            if self.prc_s_peak < 0 and self.dOrgMain_new_bns2 > 0:
                self.prc_s_peak_abs = 0

        print("[%d:%d] prc_std: %0.2f /prc_avg: %0.2f /**std_prc: %0.2f **/std_prcs_peak: %d" % (
        now.hour, now.minute, prc_std, prc_avg, std_prc, self.std_prc_peak))
        print("        prc_s_peak: %d /s+std: %d  /std_std_prc: %0.2f" % (
        self.prc_s_peak, self.prc_s_peak + self.std_prc_peak, std_std_prc))

        self.df.at[self.nf, "prc_s_peak"] = self.prc_s_peak
        self.df.at[self.nf, "prc_s_peak_abs"] = self.prc_s_peak_abs
        self.sum_peak = self.std_prc_peak + self.prc_s_peak
        self.df.at[self.nf, "sum_peak"] = self.sum_peak

        # Y
        if self.nf > 151:
            self.df.at[self.nf - 50, "y_1+"] = (price - self.df.at[self.nf - 50, "price"]) / self.tick
            self.df.at[self.nf - 100, "y_2+"] = (price - self.df.at[self.nf - 100, "price"]) / self.tick
            self.df.at[self.nf - 20, "y_3+"] = (price - self.df.at[self.nf - 20, "price"]) / self.tick
            self.df.at[self.nf, "y_1"] = (price - self.df.at[self.nf - 50, "price"]) / self.tick
            self.df.at[self.nf, "y_2"] = (price - self.df.at[self.nf - 10, "price"]) / self.tick
            self.df.at[self.nf, "y_3"] = (price - self.df.at[self.nf - 150, "price"]) / self.tick

        # xnet, ynet
        if self.nf < 2:
            dx1 = 0
            dy1 = 0
            cvol = cvolume_sum
        if self.nf >= 2:
            if self.which_market != 3:
                py1 = lblShoga1v
                px1 = lblBhoga1v
                cvol = cvolume_sum
                y1 = lblSqty1v
                x1 = lblBqty1v
                y2 = lblSqty2v
                x2 = lblBqty2v
                n1px1 = self.df.loc[self.nf - 1, "px1"]
                n1py1 = self.df.loc[self.nf - 1, "py1"]
                n1x1 = self.df.x1[self.nf - 1]
                n1x2 = self.df.x2[self.nf - 1]
                n1y1 = self.df.y1[self.nf - 1]
                n1y2 = self.df.y2[self.nf - 1]
                dx1 = xnet(px1, n1px1, cvol, cgubun_sum, x1, n1x1, x2, n1x2)
                dy1 = ynet(py1, n1py1, cvol, cgubun_sum, y1, n1y1, y2, n1y2)
            else:
                py1 = float(lblShoga1v)
                px1 = float(lblBhoga1v)
                cvol = int(cvolume_sum)
                y1 = int(lblSqty1v)
                x1 = int(lblBqty1v)
                y2 = int(lblSqty2v)
                x2 = int(lblBqty2v)
                n1px1 = float(self.df.loc[self.nf - 1, "px1"])
                n1py1 = float(self.df.loc[self.nf - 1, "py1"])
                n1x1 = int(self.df.x1[self.nf - 1])
                n1x2 = int(self.df.x2[self.nf - 1])
                n1y1 = int(self.df.y1[self.nf - 1])
                n1y2 = int(self.df.y2[self.nf - 1])
                dx1 = xnet(px1, n1px1, cvol, cgubun_sum, x1, n1x1, x2, n1x2)
                dy1 = ynet(py1, n1py1, cvol, cgubun_sum, y1, n1y1, y2, n1y2)

        self.df.at[self.nf, "dy1"] = dy1
        self.df.at[self.nf, "dx1"] = dx1

        # dxx, dyy, dxy
        if cgubun_sum == "Buy":
            wx = 0
            wy = cvol
        elif cgubun_sum == "Sell":
            wx = cvol
            wy = 0
        else:
            wx = 0
            wy = 0
        #     print("gubun error @ ", self.nf)

        dxx = dx1 + wy
        dyy = dy1 + wx
        dxy = dxx - dyy
        self.df.at[self.nf, "dxx"] = dxx
        self.df.at[self.nf, "dyy"] = dyy
        self.df.at[self.nf, "dxy"] = dxy

        # dxx,dyy ave_20
        if self.nf < self.sec_30 + 1:
            dxx_20 = 0
            dyy_20 = 0
        if self.nf >= self.sec_30 + 1:
            dxx_20 = self.df.loc[self.nf - 20:self.nf - 1, "dxx"].sum()
            dyy_20 = self.df.loc[self.nf - 20:self.nf - 1, "dyy"].sum()
        self.df.at[self.nf, "dxx_20"] = dxx_20
        self.df.at[self.nf, "dyy_20"] = dyy_20

        # dxx,dyy med_20
        if self.nf < self.sec_30 + 1:
            dxx_20_medi = 0
            dyy_20_medi = 0
        if self.nf >= self.sec_30 + 1:
            dxx_20_medi = self.df.loc[self.nf - 50:self.nf - 1, "dxx_20"].median()
            dyy_20_medi = self.df.loc[self.nf - 50:self.nf - 1, "dyy_20"].median()
        self.df.at[self.nf, "dxx_20_medi"] = dxx_20_medi
        self.df.at[self.nf, "dyy_20_medi"] = dyy_20_medi
        dxy_20_medi = dxx_20_medi - dyy_20_medi
        self.df.at[self.nf, "dxy_20_medi"] = dxy_20_medi
        # print 'x_20_medi: %d   /y_20_medi: %d' % (dxx_20_medi, dyy_20_medi)

        # dxy_20_medi_s
        if self.nf >= self.sec_30 + 1 and dxy_20_medi != 0:
            d_y = self.df.loc[self.nf - self.sec_30:self.nf - 1, "dxy_20_medi"]
            d_x = self.df.loc[self.nf - self.sec_30:self.nf - 1, "stime"]
            dxy_20_medi_s = regr.fit(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1)).coef_[0][0] * 1000

            # self.test = np.average(self.df['dxy_20_medi_s'][10:self.nf - 1], weights=self.df['nf'][10:self.nf - 1])
            # print("self.test", self.test)

        else:
            dxy_20_medi_s = 0
        self.df.at[self.nf, "dxy_20_medi_s"] = dxy_20_medi_s

        # dxy med_200
        if self.nf < 200 + 1:
            self.dxy_200_medi = 0
        if self.nf >= 200 + 1:
            self.dxy_200_medi = self.df.loc[self.nf - 200:self.nf - 1, "dxy_20_medi"].median()

            # print("dxy_200_medi", dxy_200_medi)
            # self.test1 = np.average(self.df['dxy_200_medi'][10:self.nf - 1], weights=self.df['nf'][10:self.nf - 1])
            # print("self.test1", self.test1)

        self.df.at[self.nf, "dxy_200_medi"] = self.dxy_200_medi

        # dxy_med_std_50
        if self.nf < self.min_1 + 11:
            dxy_med_std_50 = 0
        if self.nf >= self.min_1 + 11:
            dxy_med_std_50 = self.df.loc[self.nf - min(50, self.nf):self.nf - 1, "dxy_20_medi"].std()
        self.df.at[self.nf, "dxy_med_std_50"] = dxy_med_std_50

        # dxy med_std
        if self.nf < self.min_1 + 11:
            dxy_med_std = 0
        if self.nf >= self.min_1 + 11:
            dxy_med_std = self.df.loc[self.nf - min(500, self.nf):self.nf - 1, "dxy_200_medi"].std()
        self.df.at[self.nf, "dxy_med_std"] = dxy_med_std

        # dxy med_200_s
        if self.nf >= self.sec_30 + 1 and self.dxy_200_medi != 0:
            d_y = self.df.loc[self.nf - self.sec_30:self.nf - 1, "dxy_200_medi"]
            d_x = self.df.loc[self.nf - self.sec_30:self.nf - 1, "stime"]
            dxy_med_200_s = regr.fit(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1)).coef_[0][0] * 1000
        else:
            dxy_med_200_s = 0
        self.df.at[self.nf, "dxy_med_200_s"] = dxy_med_200_s

        ###### (1) dxy med_200_sig
        med_200_sig = 0
        if self.nf >= self.min_1 * 3 / 2 + 1:
            if dxy_20_medi > 0 and dxy_20_medi > self.dxy_200_medi and self.dxy_200_medi > self.dxy_200_cri_ave_p:
                med_200_sig = 1
            if dxy_20_medi < 0 and dxy_20_medi < self.dxy_200_medi and self.dxy_200_medi < self.dxy_200_cri_ave_m:
                med_200_sig = -1
        self.df.at[self.nf, "med_200_sig"] = med_200_sig

        # dxy_sig
        dxy_sig = 0
        if self.nf >= self.min_1 * 3 / 2 + 1:
            if self.dxy_200_medi < dxy_med_std * -0.9:
                dxy_sig = -1
            if self.dxy_200_medi > dxy_med_std * 0.9:
                dxy_sig = 1
        self.df.at[self.nf, "dxy_sig"] = dxy_sig

        # dxy_sig_main
        if self.nf >= self.min_1 * 3 / 2 + 1:
            if dxy_sig == 1:
                # if self.dxy_sig_dmain == 0 or self.dxy_sig_dmain == -1:
                self.dxy_sig_dmain = 1
            if dxy_sig == -1:
                # if self.dxy_sig_dmain == 0 or self.dxy_sig_dmain == -1:
                self.dxy_sig_dmain = -1
        self.df.at[self.nf, "dxy_sig_dmain"] = self.dxy_sig_dmain

        # dxy_decay
        if self.nf < self.sec_15 * 2 + 11:
            self.dxy_decay = 0
        if self.nf >= self.sec_15 * 2 + 11:
            if self.dxy_decay == 0:
                self.dxy_decay = dxy
            if self.dxy_decay != 0:
                self.dxy_decay = self.dxy_decay * 0.99
                if self.dxy_decay > 0:
                    # if dxy>0:
                    #     if dxy>self.dxy_decay:
                    #         self.dxy_decay = dxy
                    # if dxy<=0:
                    self.dxy_decay += dxy
                if self.dxy_decay < 0:
                    # if dxy<0:
                    #     if dxy<self.dxy_decay:
                    #         self.dxy_decay = dxy
                    # if dxy>=0:
                    self.dxy_decay += dxy
        # print("//dxy: ", dxy)
        # print("//dxy_decay: ", self.dxy_decay)
        self.df.at[self.nf, "dxy_decay"] = self.dxy_decay

        # EMA
        if self.nf < 400 + 11:
            ema_20 = 0
            ema_50 = 0
            ema_200 = 0
        if self.nf >= 400 + 11:
            if self.talib == 1:
                ema_20 = talib.EMA(np.array(self.df['px1'], dtype=float), 20)[-1]
                ema_50 = talib.EMA(np.array(self.df['px1'], dtype=float), 50)[-1]
                ema_200 = talib.EMA(np.array(self.df['px1'], dtype=float), 200)[-1]
            if self.talib == 0:
                ema_20 = cal_ema(pd.Series(self.df['px1']), window=20).iloc[-1]
                ema_50 = cal_ema(pd.Series(self.df['px1']), window=50).iloc[-1]
                ema_200 = cal_ema(pd.Series(self.df['px1']), window=200).iloc[-1]
            # print(ema_50)
            # print(ema_200)
        self.df.at[self.nf, "ema_20"] = ema_20
        self.df.at[self.nf, "ema_50"] = ema_50
        self.df.at[self.nf, "ema_200"] = ema_200

        ema_25 = 0
        if ema_20 > ema_50:
            ema_25 = 1
        if ema_20 < ema_50:
            ema_25 = -1
        self.df.at[self.nf, "ema_25"] = ema_25

        ema_520 = 0
        if ema_50 > ema_200:
            ema_520 = 1
            # if self.nf > self.sec_30 * 4 + 2:
            if self.df.at[self.nf - 1, "prc_s"] > 0 and self.test_signal != -3:
                if self.which_market == 1 and prc_std > 10:
                    ema_520 = 2
                if self.which_market == 2 and prc_std > 30:
                    ema_520 = 2
                if self.which_market == 3 and prc_std > 0.2:
                    ema_520 = 2
                if self.which_market == 4 and prc_std > 0.6:
                    ema_520 = 2
        if ema_50 < ema_200:
            ema_520 = -1
            # if self.nf > self.sec_30 * 4 + 2:
            if self.df.at[self.nf - 1, "prc_s"] < 0 and self.test_signal != 3:
                if self.which_market == 1 and prc_std > 10:
                    ema_520 = -2
                if self.which_market == 2 and prc_std > 30:
                    ema_520 = -2
                if self.which_market == 3 and prc_std > 0.2:
                    ema_520 = -2
                if self.which_market == 4 and prc_std > 0.6:
                    ema_520 = -2
        self.df.at[self.nf, "ema_520"] = ema_520

        # ema_520_std
        if abs(ema_520) == 1:
            if self.df.at[self.nf - 1, "ema_520_prc_std"] == 1:  # and self.df.at[self.nf-1, "ema_25_prc_std"] == 1:
                ema_520_std = ema_520
            else:
                ema_520_std = 0
        else:
            ema_520_std = 0
        self.df.at[self.nf, "ema_520_std"] = ema_520_std

        # RSI
        rsi = 50
        if self.nf < 300 + 11:
            rsi = 50
            self.prc_dev = 0
            self.rsi_init = 50
        if self.nf >= 300 + 11:
            if self.talib == 1:
                rsi = talib.RSI(np.array(self.df['price'], dtype=float), timeperiod=200)[-1]
            if self.talib == 0:
                rsi_value = cal_rsi(pd.Series(self.df['price']), window=200).iloc[-1]
            self.prc_dev = (price - self.init_prc) / self.init_prc * 100
            self.rsi_init = rsi + self.prc_dev
            # print("rsi: ", rsi)
        self.rsi = rsi
        self.df.at[self.nf, "rsi"] = rsi
        self.df.at[self.nf, "prc_dev"] = self.prc_dev
        self.df.at[self.nf, "rsi_init"] = self.rsi_init


        # RSI_peak
        self.rsi_peak = 0
        if self.nf > 500:
            if self.df_rsi[self.df_rsi >= self.rsi_limit_high].count() >= 1 and self.rsi > 50:
                if self.rsi < self.rsi_limit_high * 1:
                    self.rsi_peak = 1
                if self.rsi >= self.rsi_limit_high * 1:
                    self.rsi_peak = 2
                if self.rsi >= self.rsi_limit_high * 1.05:
                    self.rsi_peak = 3
            if self.df_rsi[self.df_rsi <= self.rsi_limit_low].count() >= 1 and self.rsi < 50:
                if self.rsi > self.rsi_limit_low * 1:
                    self.rsi_peak = -1
                if self.rsi <= self.rsi_limit_low * 1:
                    self.rsi_peak = -2
                if self.rsi <= self.rsi_limit_low * 0.95:
                    self.rsi_peak = -3
        self.df.at[self.nf, "rsi_peak"] = self.rsi_peak

        # sX, sY, sXY
        if self.nf == 0:
            sX = 0
            sY = 0
            sXY = 0
        else:
            sX = self.df.loc[self.nf - 1, "sX"] + dxx
            sY = self.df.loc[self.nf - 1, "sY"] + dyy
            sXY = self.df.loc[self.nf - 1, "sXY"] + dxy

        if self.which_market != 1:
            self.df.at[self.nf, "sX"] = sX
            self.df.at[self.nf, "sY"] = sY
            self.df.at[self.nf, "sXY"] = sXY

        if self.which_market == 1:
            self.df.at[self.nf, "sX"] = sX
            self.df.at[self.nf, "sY"] = sY
            self.df.at[self.nf, "sXY"] = sXY

        # stime
        if self.nf == 1:
            print("startime", self.startime)
        self.df.at[self.nf, "stime"] = timestamp  # n
        # owtime

        # dt => count
        self.df.at[self.nf, "count"] = count

        # count_m
        if self.nf < self.sec_15 + 1:
            count_m = 0
        if self.nf >= self.sec_15 + 1:
            count_m = self.df.loc[self.nf - 5:self.nf - 1, "count"].mean()
        self.df.at[self.nf, "count_m"] = count_m
        self.count_m = count_m

        if self.nf >= self.sec_15 + 25:
            if self.nf <= 200:
                if self.count_m_start < self.df.at[self.nf - 1, "count_m"]:
                    self.count_m_start = self.df.at[self.nf - 1, "count_m"]
            # date = datetime.strptime(df["time"][0], '%m-%d-%H-%M-%S')
            if self.which_market == 3:
                if self.nf == self.sec_15 + 25:
                    self.count_m_start = self.df.loc[self.sec_15 + 1:self.nf - 1, "count_m"].mean()
                if self.df.at[self.nf - 1, "time"][6:8] == "09" and int(self.df.at[self.nf - 1, "time"][9:11]) <= 3:
                    self.count_m_start = self.count_m_start * self.count_m_start_count + self.df.at[
                        self.nf - 1, "count_m"]
                    self.count_m_start = self.count_m_start / (self.count_m_start_count + 1)
                    self.count_m_start_count += 1

            self.df.at[self.nf, "count_m_start"] = self.count_m_start
            # print("count_m_start", self.count_m_start)

        # count_m_ave
        if self.nf < self.sec_15 * 5 + 1:
            count_m_ave = count_m
        if self.nf >= self.sec_15 * 5 + 1:
            count_m_ave = abs(self.df.loc[self.nf - self.sec_15 * 4:self.nf - 1, "count_m"]).mean()
        self.df.at[self.nf, "count_m_ave"] = count_m_ave

        # count_m_avg
        if self.nf < self.min_1 + 11:
            count_m_avg = count_m
        if self.nf >= self.min_1 + 11:
            count_m_avg = self.df.loc[self.nf - min(500, self.nf):self.nf - 1, "count_m"].mean()
        self.df.at[self.nf, "count_m_avg"] = count_m_avg

        # count_m_std
        if self.nf >= 11:  # and prc_avg != 0:
            self.count_m_std = self.df.loc[self.nf - min(500, self.nf) + 1:self.nf - 1, "count"].std() * (
                        1000 / min(self.nf, 500)) ** 0.5
        else:
            self.count_m_std = count_m_avg
        self.df.at[self.nf, "count_m_std"] = self.count_m_std

        # std_count_m
        if count_m_avg != 0 and self.count_m_std != 0:
            self.df.at[self.nf, "mean_count_m"] = count_m - count_m_avg
            self.std_count_m = (count_m - count_m_avg) / self.count_m_std
        else:
            self.std_count_m = 0
        self.df.at[self.nf, "std_count_m"] = self.std_count_m + 0.5

        # std_count_m_peak
        if count_m_avg != 0 and self.count_m_std != 0:
            if self.std_count_m_peak == 0:
                if self.std_count_m > 2:
                    self.std_count_m_peak = 2
                # if self.std_count_m < -1.8:
                #     self.std_count_m_peak = -2

            if abs(self.std_count_m_peak) == 2:
                if self.std_count_m < 1.2 and self.std_count_m > 0:
                    self.std_count_m_peak = 1
                # if self.std_count_m > -1 and self.std_count_m < 0:
                #     self.std_count_m_peak = -1
                if abs(self.std_count_m) <= 0.5:
                    self.std_count_m_peak = 0

            if abs(self.std_count_m_peak) == 1:
                if abs(self.std_count_m) <= 0.5:
                    self.std_count_m_peak = 0
        self.df.at[self.nf, "std_count_m_peak"] = self.std_count_m_peak

        ###### (1.5) count_m_sig
        count_m_sig = 0
        if self.nf >= self.sec_15 * 6 + 51 and count_m_ave != 0:
            if count_m > count_m_ave:
                if count_m > count_m_ave * 1: #self.count_m_act_ave * 1:
                    count_m_sig = 1
                if count_m > count_m_ave * 2: #self.count_m_act_ave * 2:
                    count_m_sig = 2
            if count_m < count_m_ave * 0.8:
                count_m_sig = -1
        self.df.at[self.nf, "count_m_sig"] = count_m_sig

        # count_m_sig_mode
        if self.nf > 101:
            count_m_sig_ave = self.df.loc[self.nf - 60:self.nf - 1, "count_m_sig"].mean()
            # if self.count_m_sig_mode == 0: # or self.count_m_sig_mode == -1:
            if count_m_sig_ave >= 0:
                if self.std_prc >= 0.45:
                    self.count_m_sig_mode = 1
                if self.std_prc < -0.45:
                    self.count_m_sig_mode = -1
            # if abs(self.count_m_sig_mode) == 1:
            if count_m_sig_ave < 0:
                self.count_m_sig_mode = 0

            self.df.at[self.nf, "count_m_sig_ave"] = count_m_sig_ave
        self.df.at[self.nf, "count_m_sig_mode"] = self.count_m_sig_mode

        # count_m_percentile
        if self.nf < 100 + 11:
            self.count_m_per_5 = 0
            self.count_m_per_10 = 0
            self.count_m_per_40 = 0
            self.count_m_per_80 = 0
        if self.nf >= 100 + 11:
            self.count_m_per_5 = np.percentile(np.array(self.df['count_m'], dtype=float)[self.nf - 52:self.nf - 2], 5)
            self.count_m_per_10 = np.percentile(np.array(self.df['count_m'], dtype=float)[self.nf - 52:self.nf - 2], 10)
            self.count_m_per_40 = np.percentile(np.array(self.df['count_m'], dtype=float)[self.nf - 52:self.nf - 2], 40)
            self.count_m_per_80 = np.percentile(np.array(self.df['count_m'], dtype=float)[self.nf - 52:self.nf - 2], 80)
            # print("count_m_percentile: ", count_m_per_5)
        self.df.at[self.nf, "count_m_per_5"] = self.count_m_per_5
        self.df.at[self.nf, "count_m_per_10"] = self.count_m_per_10
        self.df.at[self.nf, "count_m_per_40"] = self.count_m_per_40
        self.df.at[self.nf, "count_m_per_80"] = self.count_m_per_80

        # EMA_count_m
        if self.nf < 400 + 11:
            ema_20_count_m = 0
            ema_50_count_m = 0
            ema_200_count_m = 0
        if self.nf >= 400 + 11:
            if self.talib == 1:
                ema_20_count_m = talib.EMA(np.array(self.df['count_m_per_5'], dtype=float), 20)[-1]
                ema_50_count_m = talib.EMA(np.array(self.df['count_m_per_5'], dtype=float), 50)[-1]
                ema_200_count_m = talib.EMA(np.array(self.df['count_m_per_5'], dtype=float), 200)[-1]
            if self.talib == 0:
                ema_20_count_m = cal_ema(pd.Series(self.df['count_m_per_5']), window=20).iloc[-1]
                ema_50_count_m = cal_ema(pd.Series(self.df['count_m_per_5']), window=50).iloc[-1]
                ema_200_count_m = cal_ema(pd.Series(self.df['count_m_per_5']), window=200).iloc[-1]
            # print(ema_50)
            # print(ema_200)
        self.df.at[self.nf, "ema_20_count_m"] = ema_20_count_m
        self.df.at[self.nf, "ema_50_count_m"] = ema_50_count_m
        self.df.at[self.nf, "ema_200_count_m"] = ema_200_count_m

        # on basis of 'count_m_per_5'
        ema_25_count_m = 0
        if ema_20_count_m > ema_50_count_m:
            ema_25_count_m = 1
        if ema_20_count_m < ema_50_count_m:
            ema_25_count_m = -1
        self.df.at[self.nf, "ema_25_count_m"] = ema_25_count_m

        self.ema_520_count_m = 0
        multi_count = 1
        if self.which_market == 4:
            multi_count = 7
        if ema_50_count_m > ema_200_count_m:
            self.ema_520_count_m = 1
            if ema_20_count_m >= 2 * multi_count:
                self.ema_520_count_m = 2
        if ema_50_count_m < ema_200_count_m:
            self.ema_520_count_m = -1
        self.df.at[self.nf, "ema_520_count_m"] = self.ema_520_count_m

        # per_medi
        if self.nf < 300 + 1:
            per_medi = 0
        if self.nf >= 300 + 1:
            per_medi = self.df.loc[self.nf - 200:self.nf - 1, "count_m_per_40"].median()
        self.df.at[self.nf, "per_medi"] = per_medi

        #######
        # Static - Dynamic
        #######
        if self.nf > 5 and std_std_prc != 0:
            self.mode_spot = 0
            # self.mode_spot_t = 0
            # # if count_m_per_40 > self.count_m_act_ave and abs(std_prc) > 2: #self.std_prc_cri : # count_m_per_40 > self.count_m_act_ave:
            # #     # if prc_std_100 > self.cri_tick * 0.5:  #self.tick * 0.5:
            # #     self.mode_spot = 2
            # if std_std_prc <= 0.2 and ema_25_count_m == 1:
            #     self.mode_spot = 1
            # # if count_m_per_5 > self.count_m_act_ave or std_std_prc > self.std_prc_cri * 4:
            # #     self.mode_spot = 1

            ################## Dynamic -1  : sum of peaks == 4
            if 1 == 0:  # and abs(prc_std) >= 0.2:
                # self.cover_signal = 0
                # if self.prc_s_peak + self.std_prc_peak >= 2: # and self.mode_spot != 2:
                if ema_520 == 2 and ema_520_prc_std == 2:
                    self.mode_spot = 2
                    # self.mode_last = 2
                    if self.df.at[self.nf - 1, "mode_spot"] != 2 and self.cover_signal == 0:
                        if self.OrgMain == "s":
                            self.cover_signal = 1
                            if self.which_market == 3 and self.df.at[self.nf - 1, "cvol_m"] >= 0.03:
                                self.cover_signal = 3
                            # self.ave_prc = (self.ave_prc * self.exed_qty + float(lblShoga1v)) / (self.exed_qty + 1)
                            # self.exed_qty += 1
                            self.type = "b-dyna-1"
                            # self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                            #               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # if self.prc_s_peak + self.std_prc_peak <= -2: # and self.mode_spot != -2:
                if ema_520 == -2 and ema_520_prc_std == 2:  # NOT == -2
                    self.mode_spot = -2
                    # self.mode_last = -2
                    if self.df.at[self.nf - 1, "mode_spot"] != -2 and self.cover_signal == 0:
                        if self.OrgMain == "b":
                            self.cover_signal = -1
                            if self.which_market == 3 and self.df.at[self.nf - 1, "cvol_m"] >= 0.03:
                                self.cover_signal = -3
                            # self.ave_prc = (self.ave_prc * self.exed_qty + float(lblBhoga1v)) / (self.exed_qty + 1)
                            # self.exed_qty += 1
                            self.type = "s-dyna-1"
                            # self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                            #               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                if abs(self.cover_signal) == 1:
                    if abs(ema_520) != -2 or abs(ema_520_prc_std) != 2:
                        # if self.OrgMain == "n":
                        self.cover_signal = 0

            # mode_last (=mode)
            if self.mode_spot != 0:
                self.mode_last = self.mode_spot

            elif self.mode_spot == 0:
                if self.mode_last == 2:
                    if self.prc_s_peak + self.std_prc_peak >= 1 or std_prc > 0:
                        if self.test_signal != -3:
                            self.mode_last = 2
                    else:
                        self.mode_last = 0

                if self.mode_last == -2:
                    if self.prc_s_peak + self.std_prc_peak <= -1 or std_prc < 0:
                        if self.test_signal != 3:
                            self.mode_last = -2
                    else:
                        self.mode_last = 0

            ################## Dynamic -2  : btw test_signal_mode

            # self.last_cover_prc : price when dyna- position signal on, even in case of cover_sig is on, cover_order !=1 because it happen only when exeqty>=2
            # self.cover_out_prc : price when dyna- position out

            if 1 == 1 and self.dynamic_cover == 1:
                if self.OrgMain == "b" or self.OrgMain == "n":  # and self.test_signal_mode == 1:
                    # (in)
                    if self.last_cover_prc == 0:
                        if (
                                ema_520_prc_std >= 1 and ema_520 <= -1 and price < self.ave_prc - prc_std * 0.3 and std_prc < -1.2) or self.cover_signal_2 == -1:  # and self.test_signal >= 2:
                            if (
                                    self.cover_out_prc == 0 or price < self.cover_out_prc - prc_std * 0.4):  # and self.df.loc[self.nf - 3:self.nf - 1, "cvol_c_medi"].mean() <= 11:
                                self.cover_signal = -2
                                if self.nf > self.nfset + 2 and self.df.at[
                                    self.nf - 1, "cover_signal"] != -2 and self.cover_signal == -2 and dxy_20_medi_s < 0 and self.std_prc < 0:
                                    # if self.hist.at[self.no-1, "nf"] != self.nf:
                                    self.cover_type = "s-dyna-1"
                                    self.last_cover_prc = price
                                    # self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                    #               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1=self.prc_put)
                    # (out)
                    if self.last_cover_prc != 0 and self.cover_signal == -2:
                        if abs(self.last_cover_prc - price) >= prc_std * 0.3 or self.df.loc[self.nf - 50: self.nf - 1,
                                                                                "cover_signal"].mean() == -2:
                            if (ema_520 >= 1 and ema_520_prc_std >= 1) or self.test_signal < 2:
                                self.cover_signal = 0
                                # extending stay in out-mode
                                # extending stay in out-mode
                                # if self.cover_signal == 0:
                                if self.df.at[self.nf - 1, "cover_signal"] == -2:  # and self.test_signal != 3:
                                    if price < self.last_cover_prc + prc_std * 0.3 or std_prc < 0.5:
                                        self.cover_signal = -2
                                    # final out
                                    else:
                                        self.cover_signal = 0
                                        self.last_cover_prc = 0
                                        self.cover_out_prc = price
                        if self.df.at[self.nf - 1, "last_cover_prc"] != 0:
                            if (self.df.loc[self.nf - 6: self.nf - 3, "test_signal"].mean() >= 2 and self.df.loc[
                                                                                                     self.nf - 2: self.nf - 1,
                                                                                                     "test_signal"].mean() < 2) or self.peak_touch >= 1:
                                self.cover_signal = 0
                                self.last_cover_prc = 0
                                self.cover_out_prc = price
                            if self.df.at[self.nf - 2, "add_signal"] != -1 and self.df.at[
                                self.nf - 1, "add_signal"] == -1:
                                self.cover_signal = 0
                                self.last_cover_prc = 0
                                self.cover_out_prc = price
                        # if self.cover_signal_2 == -1:
                        #     self.cover_signal = -2

                if self.OrgMain == "s" or self.OrgMain == "n":  # and price > self.last_cover_prc + self.tick: # and self.test_signal_mode == -1:
                    # (in)
                    if self.last_cover_prc == 0:
                        if (
                                ema_520_prc_std >= 1 and ema_520 >= 1 and price > self.ave_prc + prc_std * 0.3 and std_prc > 1.2) or self.cover_signal_2 == 1:  # and self.test_signal <= -2:
                            if (
                                    self.cover_out_prc == 0 or price > self.cover_out_prc + prc_std * 0.4):  # and self.df.loc[self.nf - 3:self.nf - 1, "cvol_c_medi"].mean() >= 9:
                                self.cover_signal = 2
                                if self.nf > self.nfset + 2 and self.df.at[
                                    self.nf - 1, "cover_signal"] != 2 and self.cover_signal == 2 and dxy_20_medi_s > 0 and self.std_prc > 0:
                                    # if self.hist.at[self.no - 1, "nf"] != self.nf:
                                    self.cover_type = "b-dyna-1"
                                    self.last_cover_prc = price
                                    # self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                    #               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1=self.prc_call)
                    # (out)
                    if self.last_cover_prc != 0 and self.cover_signal == 2:
                        if abs(self.last_cover_prc - price) >= prc_std * 0.3 or self.df.loc[self.nf - 50: self.nf - 1,
                                                                                "cover_signal"].mean() == 2:
                            if (ema_520 <= -1 and ema_520_prc_std >= 1) or self.test_signal > -2:
                                self.cover_signal = 0
                                # extending stay in out-mode
                                # if self.cover_signal == 0:
                                if self.df.at[self.nf - 1, "cover_signal"] == 2:  # and self.test_signal != -3:
                                    if price > self.last_cover_prc - prc_std * 0.3 or std_prc > -0.5:
                                        self.cover_signal = 2
                                    # final out
                                    else:
                                        self.cover_signal = 0
                                        self.last_cover_prc = 0
                                        self.cover_out_prc = price
                        if self.df.at[self.nf - 1, "last_cover_prc"] != 0:
                            if (self.df.loc[self.nf - 6: self.nf - 3, "test_signal"].mean() <= -2 and self.df.loc[
                                                                                                      self.nf - 2: self.nf - 1,
                                                                                                      "test_signal"].mean() > -2) or self.peak_touch <= -1:
                                self.cover_signal = 0
                                self.last_cover_prc = 0
                                self.cover_out_prc = price
                            if self.df.at[self.nf - 2, "add_signal"] != 1 and self.df.at[
                                self.nf - 1, "add_signal"] == 1:
                                self.cover_signal = 0
                                self.last_cover_prc = 0
                                self.cover_out_prc = price
                        # if self.cover_signal_2 == 1:
                        #     self.cover_signal = 2

                if 1 == 0 and self.OrgMain == "n":  # caution ema_520_prc_std == 2 in any mode, not -2
                    if self.which_market == 3:
                        if self.df.at[self.nf - 1, "cvol_m"] >= 0.03:
                            if ema_520 == 2 and ema_520_prc_std == 2:
                                self.cover_signal = 2
                                self.type = "b-dyna-1"
                                self.last_cover_prc = price
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc,
                                              prc_o1=self.prc_call)
                            if ema_520 == -2 and ema_520_prc_std == 2:
                                self.cover_signal = -2
                                self.type = "s-dyna-1"
                                self.last_cover_prc = price
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc,
                                              prc_o1=self.prc_put)

            self.df.at[self.nf, "mode_spot"] = self.mode_spot
            self.df.at[self.nf, "mode_last"] = self.mode_last
            self.df.at[self.nf, "cover_signal"] = self.cover_signal

            # cover_signal_c

            if self.cover_signal_c == 0 and (self.cover_signal == 2 and self.df.at[self.nf - 1, "d_OMain"] == 4):
                self.cover_signal_c = 1
            if self.cover_signal_c == 0 and (self.cover_signal == -2 and self.df.at[self.nf - 1, "d_OMain"] == -4):
                self.cover_signal_c = -1

            if self.cover_signal_c == 1:
                if self.d_OMain == -4 or self.OrgMain == "n":
                    self.cover_signal_c = 0
            if self.cover_signal_c == -1:
                if self.d_OMain == 4 or self.OrgMain == "n":
                    self.cover_signal_c = 0

            self.df.at[self.nf, "cover_signal_c"] = self.cover_signal_c
            self.df.at[self.nf, "last_cover_prc"] = self.last_cover_prc
            self.df.at[self.nf, "cover_out_prc"] = self.cover_out_prc

        # cover_signal_2
        if self.nf > 101:
            # if self.which_market == 1:
            #     self.cover_signal_2 = 0
            if abs(self.count_m_sig_mode) == 1:  # and cvol_m>self.cvol_m_start * 0.8:
                if dxy_20_medi_s >= 0 and std_prc >= -0.15 and self.df.loc[self.nf - 20:self.nf - 1,
                                                          "cover_signal_2"].mean() >= 0:
                    if self.df.loc[self.nf - 20:self.nf - 1, "prc_sig"].mean() >= 0.3:# or self.df.loc[self.nf - 20:self.nf - 1, "prc_sig"].mean() > self.df.loc[self.nf - 100:self.nf - 1, "prc_sig"].mean():
                        if rsi >= 48 and self.ai >= 0.1:
                            self.cover_signal_2 = 1
                            self.cover_type = "b-dyna-21"
                    if self.test_signal <= -2 or (self.df.at[self.nf - 1, "std_prc_peak"] == -2 and self.df.at[self.nf, "std_prc_peak"] != -2):
                        if self.ai >= 0.1:
                            self.cover_signal_2 = 1
                            self.cover_type = "b-dyna-22"
                if dxy_20_medi_s <= 0 and std_prc <= 0.15 and self.df.loc[self.nf - 20:self.nf - 1,
                                                          "cover_signal_2"].mean() <= 0:
                    if self.df.loc[self.nf - 20:self.nf - 1, "prc_sig"].mean() <= -0.3:# or self.df.loc[self.nf - 20:self.nf - 1, "prc_sig"].mean() < self.df.loc[self.nf - 100:self.nf - 1, "prc_sig"].mean():
                        if rsi <= 53 and self.ai <= 0.5:
                            self.cover_signal_2 = -1
                            self.cover_type = "s-dynaf-21"
                    if self.test_signal >= 2 or (self.df.at[self.nf - 1, "std_prc_peak"] == 2 and self.df.at[self.nf, "std_prc_peak"] != 2):
                        if self.ai <= 0.5:
                            self.cover_signal_2 = -1
                            self.cover_type = "s-dyna-22"
            else:
                self.cover_signal_2 = 0

            self.cover_signal_2_out = 0
            if self.cover_signal_2 == 1:
                if abs(self.count_m_sig_mode) == 0:
                    self.cover_signal_2_out = -0.1
                if std_prc < 0 or rsi < 50:
                    self.cover_signal_2_out = -0.3
                if self.test_signal >= 0 or self.std_prc_peak >= 0:
                    self.cover_signal_2_out = -0.5

            if self.cover_signal_2 == -1:
                if abs(self.count_m_sig_mode) == 0:
                    self.cover_signal_2_out = 0.1
                if std_prc > 0 or rsi > 50:
                    self.cover_signal_2_out = 0.3
                if self.test_signal <= 0 or self.std_prc_peak <= 0:
                    self.cover_signal_2_out = 0.5

            if self.cover_signal_2 == 1 and std_prc < -0.5:
                if prc_sig == -1 or self.count_m_sig_mode == 0:
                    self.cover_signal_2_out = 0
            if self.cover_signal_2 == -1 and std_prc > 0.5:
                if prc_sig == 1 or self.count_m_sig_mode == 0:
                    self.cover_signal_2_out = 0

        self.df.at[self.nf, "cover_signal_2"] = self.cover_signal_2
        self.df.at[self.nf, "cover_signal_2_out"] = self.cover_signal_2_out

        if self.nf > 50:
            self.mode = 0
            if self.df.loc[self.nf - 30: self.nf - 1, "mode_spot"].sum() >= 10:
                self.mode = 1
            if self.df.loc[self.nf - 30: self.nf - 1, "mode_spot"].sum() <= -10:
                self.mode = -1
        else:
            self.mode = 0
        self.df.at[self.nf, "mode"] = self.mode

        # mt
        if self.nf == 0:
            mt = 0.5
        self.df.at[self.nf, "mt"] = mt

        # cvol_sum
        if self.nf < self.sec_15 + 1:
            cvol_sum = 0
        if self.nf >= self.sec_15 + 1:
            cvol_sum = self.df.loc[self.nf - 30: self.nf - 1, "cvolume"].sum()
        self.df.at[self.nf, "cvol_sum"] = cvol_sum

        # cvol_sum_RSI
        if self.nf < 250 + 1:
            cvol_sum_rsi = 50
        if self.nf >= 250 + 1:
            if self.talib == 1:
                cvol_sum_rsi = talib.RSI(np.array(self.df['cvol_sum'], dtype=float), timeperiod=200)[-1]
            if self.talib == 0:
                # print("before rsi")
                # print(pd.Series(self.df['cvol_sum']))
                cvol_sum_rsi = cal_rsi(pd.Series(self.df['cvol_sum']), window=20).iloc[-1]
                # print("after rsi")
            # print("rsi: ", rsi)
        self.df.at[self.nf, "cvol_sum_rsi"] = cvol_sum_rsi

        # cvol_c
        if self.nf < self.sec_15 + 1:
            cvol_c = 0
        if self.nf >= self.sec_15 + 1:
            cvol_c = self.df[self.nf - 20:self.nf - 1][self.df.cvolume[self.nf - 20:self.nf - 1] > 0].count()[0]
        self.df.at[self.nf, "cvol_c"] = cvol_c
        # print 'cvol_c: ', cvol_c

        # cvol_c_ave
        if self.nf < self.sec_15 * 4 + 1:
            self.cvol_c_ave = 10
        if self.nf >= self.sec_15 * 4 + 1:
            # cvol_c_ave = (self.df.loc[self.nf - 200, "cvol_c"]+ cvol_c)/2
            self.cvol_c_ave = self.df.loc[self.nf - 10:self.nf - 1, "cvol_c"].mean()
        self.df.at[self.nf, "cvol_c_ave"] = self.cvol_c_ave

        # cvol_c_medi
        if self.nf < self.sec_30 + 1:
            cvol_c_medi = 0
        if self.nf >= self.sec_30 + 1:
            cvol_c_medi = self.df.loc[self.nf - 50:self.nf - 1, "cvol_c"].median()
        self.df.at[self.nf, "cvol_c_medi"] = cvol_c_medi

        ###### (2) cvol_c_sig
        cvol_c_sig = 0
        if self.nf >= self.sec_30 + 1:
            if (cvol_c - 10) > (self.cvol_c_ave - 10) + 2 and cvol_c > 12:
                cvol_c_sig = 1
            if (cvol_c - 10) < (self.cvol_c_ave - 10) - 2 and cvol_c < 8:
                cvol_c_sig = -1
        self.df.at[self.nf, "cvol_c_sig"] = cvol_c_sig

        # cvol_c_sig_sum
        if self.nf < self.sec_30 + 201:
            self.cvol_c_sig_sum = 0
        if self.nf >= self.sec_30 + 201:
            self.cvol_c_sig_sum = self.df.loc[self.nf - 200:self.nf - 1, "cvol_c_sig"].sum()
        self.df.at[self.nf, "cvol_c_sig_sum"] = self.cvol_c_sig_sum

        # cvol_c_sig_sum_slope
        if self.nf >= self.sec_30 * 4 + 1:
            d_y = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "cvol_c_sig_sum"]
            d_x = self.df.loc[self.nf - self.sec_30 * 4:self.nf - 1, "stime"]
            self.cvol_c_sig_sum_slope = regr.fit(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1)).coef_[0][
                                            0] * 100000
            # prc_b = regr.score(d_x.values.reshape(-1, 1), d_y.values.reshape(-1, 1))
            # print("prc_s", prc_s)
            # print("prc_b", prc_b)
        else:
            self.cvol_c_sig_sum_slope = 0
            # prc_b = 0
        self.df.at[self.nf, "cvol_c_sig_sum_slope"] = self.cvol_c_sig_sum_slope

        # cvol_m_org
        if self.nf < self.sec_15 + 1:
            cvol_m_org = 0
        if self.nf >= self.sec_15 + 1:
            cvol_m_org = abs(self.df.loc[self.nf - self.sec_15:self.nf - 1, "cvolume"]).mean()
        self.df.at[self.nf, "cvol_m_org"] = cvol_m_org

        cvol_abs = abs(cvolume_sum)
        self.df.at[self.nf, "cvol_abs"] = cvol_abs

        # cvol_m
        if self.nf < self.sec_15 + 1:
            self.cvol_m = 0
        if self.nf >= self.sec_15 + 1:
            term_t = float(timestamp - self.df.at[self.nf - self.sec_15, "stime"])
            # print("timestamp: ", timestamp)
            # print("term_t: ",term_t)
            self.cvol_m = abs(self.df.loc[self.nf - self.sec_15:self.nf - 1, "cvol_abs"]).sum() / term_t  # * 1000
        self.df.at[self.nf, "cvol_m"] = self.cvol_m
        if self.nf >= self.sec_15 + 5:
            if self.nf <= 200:
                if self.cvol_m_start < self.df.at[self.nf - 1, "cvol_m"]:
                    self.cvol_m_start = self.df.at[self.nf - 1, "cvol_m"]
            if self.which_market == 3:
                if self.nf == self.sec_15 + 25:
                    self.cvol_m_start = self.df.loc[self.sec_15 + 1:self.nf - 1, "cvol_m"].mean()
                if self.df.at[self.nf - 1, "time"][6:8] == "09" and int(self.df.at[self.nf - 1, "time"][9:11]) <= 3:
                    self.cvol_m_start = self.cvol_m_start * self.cvol_m_start_count + self.df.at[self.nf - 1, "cvol_m"]
                    self.cvol_m_start = self.cvol_m_start / (self.cvol_m_start_count + 1)
                    self.cvol_m_start_count += 1

            self.df.at[self.nf, "cvol_m_start"] = self.cvol_m_start
            # print("cvol_m_start", self.cvol_m_start)

        # cvol_m_mean
        if self.nf < self.sec_15 + 1:
            cvol_m_mean = 0
        if self.nf >= self.sec_15 + 1:
            cvol_m_mean = abs(self.df.loc[self.nf - self.sec_15:self.nf - 1, "cvol_abs"]).mean()
        self.df.at[self.nf, "cvol_m_mean"] = cvol_m_mean

        # cvol_m_ave
        if self.nf < self.sec_15 * 2 + 1:
            cvol_m_ave = 0
        if self.nf >= self.sec_15 * 2 + 1:
            cvol_m_ave = abs(self.df.loc[self.nf - self.sec_15 * 2:self.nf - 1, "cvol_m"]).mean()
        self.df.at[self.nf, "cvol_m_ave"] = cvol_m_ave

        # cvol_m_decay
        if self.nf < self.sec_15 * 2 + 11:
            self.cvol_m_decay = 0
        if self.nf >= self.sec_15 * 2 + 11:
            if self.cvol_m_decay == 0:
                self.cvol_m_decay = self.cvol_m
            if self.cvol_m_decay != 0:
                self.cvol_m_decay = self.cvol_m_decay * 0.98
                if self.cvol_m_decay > 0:
                    # if cvol_m>0:
                    #     if cvol_m>self.cvol_m_decay:
                    #         self.cvol_m_decay = cvol_m
                    # if cvol_m<=0:
                    self.cvol_m_decay += self.cvol_m
                if self.cvol_m_decay < 0:
                    # if cvol_m<0:
                    #     if cvol_m<self.cvol_m_decay:
                    #         self.cvol_m_decay = cvol_m
                    # if cvol_m>=0:
                    self.cvol_m_decay += self.cvol_m
        # print("//cvol_m: ", cvol_m)
        # print("//cvol_m_decay: ", self.cvol_m_decay)
        self.df.at[self.nf, "cvol_m_decay"] = self.cvol_m_decay

        # EMA_cvol_m
        if self.nf < 400 + 11:
            ema_20_cvol_m = 0
            ema_50_cvol_m = 0
            ema_200_cvol_m = 0
        if self.nf >= 400 + 11:
            if self.talib == 1:
                ema_20_cvol_m = talib.EMA(np.array(self.df['cvol_m'], dtype=float), 20)[-1]
                ema_50_cvol_m = talib.EMA(np.array(self.df['cvol_m'], dtype=float), 50)[-1]
                ema_200_cvol_m = talib.EMA(np.array(self.df['cvol_m'], dtype=float), 200)[-1]
            if self.talib == 0:
                ema_20_cvol_m = cal_ema(pd.Series(self.df['cvol_m']), window=20).iloc[-1]
                ema_50_cvol_m = cal_ema(pd.Series(self.df['cvol_m']), window=50).iloc[-1]
                ema_200_cvol_m = cal_ema(pd.Series(self.df['cvol_m']), window=200).iloc[-1]
        self.df.at[self.nf, "ema_20_cvol_m"] = ema_20_cvol_m
        self.df.at[self.nf, "ema_50_cvol_m"] = ema_50_cvol_m
        self.df.at[self.nf, "ema_200_cvol_m"] = ema_200_cvol_m

        ema_25_cvol_m = 0
        if ema_20_cvol_m > ema_50_cvol_m:
            ema_25_cvol_m = 1
        if ema_20_cvol_m < ema_50_cvol_m:
            ema_25_cvol_m = -1
        self.df.at[self.nf, "ema_25_cvol_m"] = ema_25_cvol_m

        ema_520_cvol_m = 0
        if ema_50_cvol_m > ema_200_cvol_m:
            ema_520_cvol_m = 1
        if ema_50_cvol_m < ema_200_cvol_m:
            ema_520_cvol_m = -1
        self.df.at[self.nf, "ema_520_cvol_m"] = ema_520_cvol_m

        ###### (3) cvol_m_sig
        cvol_m_sig = 0
        if self.nf >= self.sec_15 + 51:
            if self.cvol_m > cvol_m_ave and self.cvol_m_cri_ave != 0 and self.cvol_m > self.cvol_m_cri_ave:
                cvol_m_sig = 1
            else:
                cvol_m_sig = 0
        self.df.at[self.nf, "cvol_m_sig"] = cvol_m_sig

        ###### (3) cvol_m_sig_ave
        self.cvol_m_sig = 0
        if self.nf >= self.sec_15 + 51:
            if self.cvol_m > cvol_m_ave and self.cvol_m_cri_ave != 0 and self.cvol_m > self.cvol_m_cri_ave:
                self.cvol_m_sig = 1
            else:
                self.cvol_m_sig = 0
        self.df.at[self.nf, "cvol_m_sig"] = self.cvol_m_sig

        ###### (3) cvol_m_sig_a  # good index
        cvol_m_sig_a = 0
        if self.nf >= self.sec_15 + 51:
            if self.cvol_m > cvol_m_ave and self.cvol_m_cri_ave != 0 and self.cvol_m > self.cvol_m_cri_ave * 1.25:
                cvol_m_sig_a = 1
            elif self.cvol_m > cvol_m_ave and self.cvol_m_cri_ave != 0 and self.cvol_m > self.cvol_m_cri_ave * 2:
                cvol_m_sig_a = 2
            if self.cvol_m < cvol_m_ave:
                cvol_m_sig_a = 0
        self.df.at[self.nf, "cvol_m_sig_a"] = cvol_m_sig_a

        ###### (3-1) cvol_m_peak
        cvol_m_peak = 0
        if self.nf >= self.sec_15 + 51:
            if self.cvol_m > cvol_m_ave * 2 and self.cvol_m_cri_ave != 0 and self.cvol_m > self.cvol_m_cri_ave: # and self.cvol_m > self.cvol_m_start * 0.3:
                if self.dxy_200_medi>0:
                    self.cvol_m_peak_ox = 1
                    cvol_m_peak = 1
                if self.dxy_200_medi<0:
                    self.cvol_m_peak_ox = -1
                    cvol_m_peak = -1
            if self.cvol_m < cvol_m_ave:
                cvol_m_peak = 0
        self.df.at[self.nf, "cvol_m_peak"] = cvol_m_peak

        ###### (3-2) cvol_m_inv_peak
        cvol_m_inv_peak = 0
        if self.nf >= self.sec_15 + 51:
            if self.cvol_m_peak_ox == 1 or self.cvol_m_peak_ox == -1:
                if self.cvol_m < cvol_m_ave * 2 and self.cvol_m_cri_ave != 0 and self.cvol_m < self.cvol_m_cri_ave:
                    cvol_m_inv_peak = -1
                    if self.cvol_m_peak_ox == 1:
                        self.cvol_m_peak_ox = 0
                else:
                    cvol_m_inv_peak = 0
                    if self.cvol_m_peak_ox == -1:
                        self.cvol_m_peak_ox = 0
        self.df.at[self.nf, "cvol_m_inv_peak"] = cvol_m_inv_peak
        self.df.at[self.nf, "cvol_m_peak_ox"] = self.cvol_m_peak_ox

        # start
        self.count_m_start_std = 0.5
        self.cvol_m_start_std = 0.5
        if self.nf > self.sec_15 + 30 and self.cvol_m_start != 0:
            self.count_m_start_std = count_m / self.count_m_start
            self.cvol_m_start_std = self.cvol_m / self.cvol_m_start
        self.df.at[self.nf, "count_m_start_std"] = self.count_m_start_std
        self.df.at[self.nf, "cvol_m_start_std"] = self.cvol_m_start_std

        if 1 == 1:
            ######### (220828) std_prc_cvol_m
            if self.nf < self.sec_15 + 1:
                self.std_prc_cvol_m = 0
            if self.nf >= self.sec_15 + 1:
                if self.nf > 310:
                    self.df_tail = self.df.tail(450)
                    # pxy_l_ave_loc = abs(50-self.df.loc[self.nf - 500: self.nf - 1, "PXY_l_ave"].mean())/100
                    pxy_l_term = self.df.loc[self.nf - 500: self.nf - 1, "rsi"]
                    if 1 == 1:
                        pxy_l_count1 = (pxy_l_term[pxy_l_term > 52].count() - pxy_l_term[pxy_l_term < 48].count()) / 500
                    if 1 == 1:
                        pxy_l_count2 = self.df_tail['rsi'].ewm(alpha=0.5).mean().iloc[-1]# * 2
                    # print("pxy_l_count: ", pxy_l_count2)
                    # pxy_l_ave_ewm = pxy_l_term.ewm(50).mean()[self.nf - 1]
                    pxy_l_ave_loc = abs(50 - pxy_l_count2) / 100 * 2
                    # pxy_l_ave_count = pxy_l_ave_loc[pxy_l_ave_loc > 0.8].count() / 1000
                else:
                    pxy_l_ave_loc = 0
                    pxy_l_count1 = 0
                    pxy_l_count2 = 50
                self.df.at[self.nf, "pxy_l_count1"] = pxy_l_count1
                self.df.at[self.nf, "pxy_l_count2"] = pxy_l_count2
                self.df.at[self.nf, "pxy_l_ave_loc"] = pxy_l_ave_loc

                if prc_avg != 0 and prc_std != 0:
                    self.df.at[self.nf, "mean_prc"] = lblBhoga1v - prc_avg
                    self.std_prc_cvol_m = ((lblBhoga1v - prc_avg) * (self.cvol_m) ** 0.5) * 100 * (1 + pxy_l_ave_loc)
                    if self.which_market == 1:
                        self.std_prc_cvol_m = ((lblBhoga1v - prc_avg) / 100 * (self.cvol_m ** 0.25)) * 100 * (
                                    1 + pxy_l_ave_loc)
                else:
                    self.std_prc_cvol_m = 0
            self.df.at[self.nf, "std_prc_cvol_m"] = self.std_prc_cvol_m

            # (220828) std_prc_peak_cvol_m
            if self.nf < self.sec_15 + 1:
                self.std_prc_peak_cvol_m = 0
            if self.nf >= self.sec_15 + 1:
                if 1 == 1:  # prc_avg != 0 and prc_std != 0:

                    mean_std_prc_cvol_m = self.df.loc[self.nf - 50:self.nf - 1, "std_prc_cvol_m"].mean()
                    if self.which_market == 3:
                        self.std_prc_peak_cvol_m = (mean_std_prc_cvol_m // self.std_prc_cvol_m_limit) * 1

                        # if self.std_prc_peak_cvol_m == 0:
                        #     if mean_std_prc_cvol_m > 4:
                        #         self.std_prc_peak_cvol_m = 2
                        #     if mean_std_prc_cvol_m < -4:
                        #         self.std_prc_peak_cvol_m = -2
                        #     if mean_std_prc_cvol_m > 8:
                        #         self.std_prc_peak_cvol_m = 3
                        #     if mean_std_prc_cvol_m < -8:
                        #         self.std_prc_peak_cvol_m = -3
                        #
                        # if abs(self.std_prc_peak_cvol_m) == 2:
                        #     if mean_std_prc_cvol_m < 4 and self.std_prc_cvol_m > 0:
                        #         self.std_prc_peak_cvol_m = 1
                        #     if mean_std_prc_cvol_m > -4 and self.std_prc_cvol_m < 0:
                        #         self.std_prc_peak_cvol_m = -1
                        #     if abs(mean_std_prc_cvol_m) <= 2.5:
                        #         self.std_prc_peak_cvol_m = 0
                        #
                        # if abs(self.std_prc_peak_cvol_m) == 1:
                        #     if abs(mean_std_prc_cvol_m) <= 2.5:
                        #         self.std_prc_peak_cvol_m = 0

                    if self.which_market == 4:
                        self.std_prc_peak_cvol_m = (mean_std_prc_cvol_m // 3) * 1
                        # if self.std_prc_peak_cvol_m == 0:
                        #     if mean_std_prc_cvol_m > 1.6:
                        #         self.std_prc_peak_cvol_m = 2
                        #     if mean_std_prc_cvol_m < -1.6:
                        #         self.std_prc_peak_cvol_m = -2
                        #     if mean_std_prc_cvol_m > 2.2:
                        #         self.std_prc_peak_cvol_m = 3
                        #     if mean_std_prc_cvol_m < -2.2:
                        #         self.std_prc_peak_cvol_m = -3
                        #
                        # if abs(self.std_prc_peak_cvol_m) >= 2:
                        #     if (mean_std_prc_cvol_m < 1.6 or self.std_prc_cvol_m < 0) and self.std_prc_cvol_m > 0:
                        #         self.std_prc_peak_cvol_m = 1
                        #     if (mean_std_prc_cvol_m > -1.6 or self.std_prc_cvol_m > 0) and self.std_prc_cvol_m < 0:
                        #         self.std_prc_peak_cvol_m = -1
                        #     if abs(mean_std_prc_cvol_m) <= 0.5:
                        #         self.std_prc_peak_cvol_m = 0
                        #
                        # if abs(self.std_prc_peak_cvol_m) == 1:
                        #     if abs(mean_std_prc_cvol_m) <= 0.5:
                        #         self.std_prc_peak_cvol_m = 0

                    if self.which_market == 1:
                        self.std_prc_peak_cvol_m = (mean_std_prc_cvol_m // 1) * 1
                        # if self.std_prc_peak_cvol_m == 0:
                        #     if mean_std_prc_cvol_m > 1:
                        #         self.std_prc_peak_cvol_m = 1
                        #     if mean_std_prc_cvol_m < -1:
                        #         self.std_prc_peak_cvol_m = -1
                        #     if mean_std_prc_cvol_m > 2:
                        #         self.std_prc_peak_cvol_m = 2
                        #     if mean_std_prc_cvol_m < -2:
                        #         self.std_prc_peak_cvol_m = -2
                        #
                        # if abs(self.std_prc_peak_cvol_m) == 2:
                        #     if (mean_std_prc_cvol_m < 1 or self.std_prc_cvol_m < 0) and self.std_prc_cvol_m > 0:
                        #         self.std_prc_peak_cvol_m = 1
                        #     if (mean_std_prc_cvol_m > -1 or self.std_prc_cvol_m > 0) and self.std_prc_cvol_m < 0:
                        #         self.std_prc_peak_cvol_m = -1
                        #     if abs(mean_std_prc_cvol_m) <= 1:
                        #         self.std_prc_peak_cvol_m = 0
                        #
                        # if abs(self.std_prc_peak_cvol_m) == 1:
                        #     if abs(mean_std_prc_cvol_m) <= 0.5:
                        #         self.std_prc_peak_cvol_m = 0

                self.df.at[self.nf, "std_prc_peak_cvol_m"] = self.std_prc_peak_cvol_m

            ######### std_std_prc_cvol_m
            if self.nf < self.sec_15 + 1:
                self.std_std_prc_cvol_m = 0
            if self.nf >= self.sec_15 + 1:
                if self.which_market == 1:
                    self.std_std_prc_cvol_m = self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].std() * 100
                if self.which_market == 3 or self.which_market == 4:
                    self.std_std_prc_cvol_m = self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].std()

            self.df.at[self.nf, "std_std_prc_cvol_m"] = self.std_std_prc_cvol_m

            if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_max:
                self.std_std_prc_cvol_m_max = self.std_std_prc_cvol_m
            self.df.at[self.nf, "std_std_prc_cvol_m_max"] = self.std_std_prc_cvol_m_max
            # print(' >>>>>>>>>> std_std_prc_cvol_m_max: ', self.std_std_prc_cvol_m_max)

            if self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_min:
                self.std_std_prc_cvol_m_min = self.std_std_prc_cvol_m
            self.df.at[self.nf, "std_std_prc_cvol_m_min"] = self.std_std_prc_cvol_m_min
            # print(' >>>>>>>>>> std_std_prc_cvol_m_min: ', self.std_std_prc_cvol_m_min)

        # cvol_s
        if self.nf < self.sec_30 + 1:
            cvol_s = 0
        if self.nf >= self.sec_30 + 1:
            c_y = self.df.loc[self.nf - 9:self.nf - 1, "cvol_m"]
            c_x = self.df.loc[self.nf - 9:self.nf - 1, "stime"]
            cvol_s = regr.fit(c_x.values.reshape(-1, 1), c_y.values.reshape(-1, 1)).coef_[0][0]
        self.df.at[self.nf, "cvol_s"] = cvol_s * 3600
        self.df.at[self.nf, "cvol_s_abs"] = abs(cvol_s) * 3600
        # print 'cvol_s: ', cvol_s

        # cvol_s_15
        if self.nf < self.min_1 + 1:
            cvol_s_15 = 0
        if self.nf >= self.min_1 + 1:
            c_y_3 = self.df.loc[self.nf - self.sec_30:self.nf - 1, "cvol_m"]
            c_x_3 = self.df.loc[self.nf - self.sec_30:self.nf - 1, "stime"]
            cvol_s_15 = regr.fit(c_x_3.values.reshape(-1, 1), c_y_3.values.reshape(-1, 1)).coef_[0][0]
        self.df.at[self.nf, "cvol_s_15"] = cvol_s_15 * 3600

        # cvol_s_std
        if self.nf < self.min_1 + 11:
            cvol_s_std = 0
        if self.nf >= self.min_1 + 11:
            cvol_s_std = self.df.loc[self.nf - 10:self.nf - 1, "cvol_s"].std()
        self.df.at[self.nf, "cvol_s_std"] = cvol_s_std

        ############   like : std_prc  ##############
        # cvol_s_avg
        if self.nf < self.min_1 + 11:
            cvol_s_avg = cvol_s
            if self.nf >= 1:
                if self.df.at[self.nf - 1, "cvol_s_avg"] == 0:
                    self.df.at[self.nf - 1, "cvol_s_avg"] = cvol_s_std
        if self.nf >= self.min_1 + 11:
            cvol_s_avg = self.df.loc[self.nf - min(500, self.nf):self.nf - 1, "cvol_s_std"].mean()
        self.df.at[self.nf, "cvol_s_avg"] = cvol_s_avg

        # cvol_s_std_500
        if self.nf >= 101:
            cvol_s_std_500 = self.df.loc[self.nf - min(500, self.nf):self.nf - 1, "cvol_s"].std() * (
                        1000 / min(self.nf, 500)) ** 0.5
        else:
            cvol_s_std_500 = 0
        self.df.at[self.nf, "cvol_s_std_500"] = cvol_s_std_500

        ##############
        # std_cvol_s
        if cvol_s_std_500 != 0:
            self.std_cvol_s = (cvol_s_std - cvol_s_avg) / cvol_s_std_500
        else:
            self.std_cvol_s = 0
        self.df.at[self.nf, "std_cvol_s"] = self.std_cvol_s

        # cvol_s_peak
        if cvol_s_avg != 0 and cvol_s_std != 0:
            if self.cvol_s_peak == 0:
                if self.std_cvol_s >= 1.8:  # self.std_cvol_s > 0.005:
                    self.cvol_s_peak = 2
                if self.std_cvol_s <= -1.8:  # self.std_cvol_s < -0.005:
                    self.cvol_s_peak = -2

            if abs(self.cvol_s_peak) == 2:
                if self.std_cvol_s < 1.8 and self.std_cvol_s > 0.5:
                    self.cvol_s_peak = 1
                if self.std_cvol_s > -1.8 and self.std_cvol_s < -0.5:
                    self.cvol_s_peak = -1
                if abs(self.std_cvol_s) <= 0.5:
                    self.cvol_s_peak = 0

            if abs(self.cvol_s_peak) == 1:
                if abs(self.std_cvol_s) <= 0.5:
                    self.cvol_s_peak = 0
        self.df.at[self.nf, "cvol_s_peak"] = self.cvol_s_peak
        ################################################

        # cvol_s_peak_2
        if self.cvol_s_peak_2 == 0:
            if cvol_s > cvol_s_std_500 * 2:  # and self.df.at[self.nf - 1, "cvol_m"] > self.cvol_m_cri_ave:
                self.cvol_s_peak_2 = 2
            if cvol_s < cvol_s_std_500 * -2:  # and self.df.at[self.nf - 1, "cvol_m"] > self.cvol_m_cri_ave:
                self.cvol_s_peak_2 = -2

        if abs(self.cvol_s_peak_2) == 2:
            if cvol_s < cvol_s_std_500 * 1 and cvol_s > 0:
                self.cvol_s_peak_2 = 1
            if cvol_s > cvol_s_std_500 * -1 and cvol_s < 0:
                self.cvol_s_peak_2 = -1
            if abs(cvol_s) <= cvol_s_std_500 * 0.5:
                self.cvol_s_peak_2 = 0

        if abs(self.cvol_s_peak_2) == 1:
            if abs(cvol_s) <= cvol_s_std_500 * 0.5:
                self.cvol_s_peak_2 = 0
        self.df.at[self.nf, "cvol_s_peak_2"] = self.cvol_s_peak_2

        # cvol_s_ave_long
        if self.nf < self.min_1 + 1:
            cvol_s_ave_long = 0
        if self.nf >= self.min_1 + 1:
            cvol_s_ave_long = self.df.loc[self.nf - self.sec_30 * 2:self.nf - 1, "cvol_s_std"].mean()
        self.df.at[self.nf, "cvol_s_ave_long"] = cvol_s_ave_long

        ###### (3-5) cvol_s_sig
        cvol_s_sig = 0
        if self.nf >= self.sec_15 + 51:
            if cvol_s_std > cvol_s_ave_long * 2 and self.cvol_s_cri_ave_p != 0:  # and abs(cvol_s) > abs(self.cvol_s_cri_ave_p):
                cvol_s_sig = 1
            # if cvol_s_std > cvol_s_ave_long * 2 and self.cvol_s_cri_ave_m!=0: # and abs(cvol_s) > abs(self.cvol_s_cri_ave_m):
            #     cvol_s_sig = -1
        self.df.at[self.nf, "cvol_s_sig"] = cvol_s_sig

        # cvol_n
        if self.nf < self.sec_30 + 1:
            cvol_n = 0
        if self.nf >= self.sec_30 + 1:
            cn_y = self.df.loc[self.nf - 9:self.nf - 1, "cvol_m"]
            cn_x = self.df.loc[self.nf - 9:self.nf - 1, "nf"]
            cvol_n = regr.fit(cn_x.values.reshape(-1, 1), cn_y.values.reshape(-1, 1)).coef_[0][0]
        self.df.at[self.nf, "cvol_n"] = cvol_n

        # cvol_t
        if self.nf < self.sec_15 + 1:
            cvol_t = 0
        if self.nf >= self.sec_15 + 1:
            cvol_t = cvolume_sum / mt
        self.df.at[self.nf, "cvol_t"] = cvol_t

        # cvol_t_ave
        if self.nf < self.sec_15 + 11:
            cvol_t_ave = 0
        if self.nf >= self.sec_15 + 11:
            cvol_t_ave = self.df.loc[self.nf - 10:self.nf - 1, "cvol_t"].mean()
        self.df.at[self.nf, "cvol_t_ave"] = cvol_t_ave

        # cvol_t_ave_long
        if self.nf < self.sec_30 + 1:
            cvol_t_ave_long = 0
        if self.nf >= self.sec_30 + 1:
            cvol_t_ave_long = self.df.loc[self.nf - self.sec_30:self.nf - 1, "cvol_t"].mean()
        self.df.at[self.nf, "cvol_t_ave_long"] = cvol_t_ave_long

        # cvol_t_decay
        if self.nf < self.sec_15 + 11:
            self.cvol_t_decay = 0
        if self.nf >= self.sec_15 + 11:
            if self.cvol_t_decay == 0:
                self.cvol_t_decay = cvol_t
            if self.cvol_t_decay != 0:
                self.cvol_t_decay = self.cvol_t_decay * 0.98
                if self.cvol_t_decay > 0:
                    # if cvol_t>0:
                    #     if cvol_t>self.cvol_t_decay:
                    #         self.cvol_t_decay = cvol_t
                    # if cvol_t<=0:
                    self.cvol_t_decay += cvol_t
                if self.cvol_t_decay < 0:
                    # if cvol_t<0:
                    #     if cvol_t<self.cvol_t_decay:
                    #         self.cvol_t_decay = cvol_t
                    # if cvol_t>=0:
                    self.cvol_t_decay += cvol_t
        # print("//cvol_t: ", cvol_t)
        # print("//cvol_t_decay: ", self.cvol_t_decay)
        self.df.at[self.nf, "cvol_t_decay"] = self.cvol_t_decay

        # cvol_t_t
        if self.nf < self.sec_30 + 1:
            cvol_t_t = 0
        if self.nf >= self.sec_30 + 1:
            cn_y_t = self.df.loc[self.nf - 6:self.nf - 1, "cvol_t"]
            cn_x_t = self.df.loc[self.nf - 6:self.nf - 1, "nf"]
            cvol_t_t = regr.fit(cn_x_t.values.reshape(-1, 1), cn_y_t.values.reshape(-1, 1)).coef_[0][0]
        self.df.at[self.nf, "cvol_t_t"] = cvol_t_t

        ###### (4) cvol_t_ox
        cvol_t_ox = 0
        if self.nf >= self.sec_15 + 51:
            if cvol_t_ave > 0 and self.cvol_t_cri_ave_p != 0 and cvol_t > self.cvol_t_cri_ave_p:
                if cvol_t_ave > (cvol_t_ave_long) * 2 and cvol_t > (cvol_t_ave_long) * 100:
                    cvol_t_ox = 1
            if cvol_t_ave < 0 and self.cvol_t_cri_ave_m != 0 and cvol_t < self.cvol_t_cri_ave_m:
                if cvol_t_ave < (cvol_t_ave_long) * 2 and cvol_t < (cvol_t_ave_long) * 100:
                    cvol_t_ox = -1
        self.df.at[self.nf, "cvol_t_ox"] = cvol_t_ox

        # ###### (4-1) cvol_t_ave_peak
        # cvol_t_ave_peak = 0
        # if self.nf >=  self.sec_15 + 51:
        #     if cvol_t_ave > cvol_t_ave_long * 2 and cvol_t_ave_long!=0: # and cvol_m > self.cvol_m_cri_ave:
        #         cvol_t_ave_peak = 1
        #     else:
        #         cvol_t_ave_peak = 0
        # self.df.at[self.nf, "cvol_t_ave_peak"] = cvol_t_ave_peak

        # sXY_s
        if self.nf >= self.sec_15 + 1:
            # c = range(0, 300, 5)
            ry = self.df.loc[self.nf - self.sec_15:self.nf - 1, "sXY"]  # .iloc[c]
            rx = self.df.loc[self.nf - self.sec_15:self.nf - 1, "stime"]  # .iloc[c]
            sXY_s = regr.fit(rx.values.reshape(-1, 1), ry.values.reshape(-1, 1)).coef_[0][0]
        else:
            sXY_s = 0
            p_value = 0
            std_err = 0
        self.df.at[self.nf, "sXY_s"] = sXY_s

        # ee
        if self.nf >= self.sec_15 + 1:
            if mt < 0.98:
                ee_mt = 1 + stat.norm.ppf((1 - mt) / float(1))
            if mt >= 0.98:
                ee_mt = 1
        else:
            ee_mt = 1
        if ee_mt < 0:
            ee_mt = 0.001
        self.df.at[self.nf, "ee_mt"] = ee_mt

        # ee_s
        if self.nf >= self.sec_15 + 1:
            ee_s = self.df.loc[self.nf - self.sec_15:self.nf - 1, "ee_mt"].mean()
        else:
            ee_s = 1
        self.df.at[self.nf, "ee_s"] = ee_s

        # # price_std & count_m_cri
        # # prc_std
        # if self.nf < self.sec_15+1:
        #     prc_std = 0
        # if self.nf >= self.sec_15+1:
        #     prc_std = self.df.loc[self.nf - 20:self.nf - 1, "price"].std()
        # self.df.at[self.nf, "prc_std"] = prc_std
        # prc_std_per = prc_std/price*100
        # self.df.at[self.nf, "prc_std_per"] = prc_std_per

        # price_mean
        price_mean = self.df.loc[self.nf - 5:self.nf - 1, "price"].mean()
        self.df.at[self.nf, "price_mean"] = price_mean
        # price_min = self.df.loc[self.nf - self.sec_15 / 2:self.nf - 1, "price"].min()
        # prc_std_per_mean = self.df.loc[self.nf - self.sec_15:self.nf - 1, "prc_std_per"].mean()

        # price_mean_long
        price_mean_long = self.df.loc[self.nf - self.sec_30:self.nf - 1, "price"].mean()
        self.df.at[self.nf, "price_mean_long"] = price_mean_long

        # Wilcoxon Test

        if self.nf > 50:  # and nf>20:
            y1_sX = self.df.loc[self.nf - 50:self.nf - 1, "sX"]
            y2_sX = self.df.loc[self.nf - 7:self.nf - 1, "sX"]
            try:
                if y1_sX.equals(y2_sX) == 1:
                    wc_sX = 50
                else:
                    u, wc_sX = stat.mannwhitneyu(y1_sX, y2_sX)
                    if y2_sX.mean() >= y1_sX.mean():
                        wc_sX = 50 + (0.5 - wc_sX) * 100
                    else:
                        wc_sX = 50 - (0.5 - wc_sX) * 100
            except:
                wc_sX = 50

            y1_sY = self.df.loc[self.nf - 50:self.nf - 1, "sY"]
            y2_sY = self.df.loc[self.nf - 7:self.nf - 1, "sY"]
            try:
                if y1_sY.equals(y2_sY) == 1:
                    wc_sY = 50
                else:
                    u, wc_sY = stat.mannwhitneyu(y1_sY, y2_sY)
                    if y2_sY.mean() >= y1_sY.mean():
                        wc_sY = 50 + (0.5 - wc_sY) * 100
                    else:
                        wc_sY = 50 - (0.5 - wc_sY) * 100
            except:
                wc_sY = 50

            y1_sXY = self.df.loc[self.nf - 50:self.nf - 1, "sXY"]
            y2_sXY = self.df.loc[self.nf - 7:self.nf - 1, "sXY"]
            try:
                if y1_sXY.equals(y2_sXY) == 1:
                    wc_sXY = 50
                else:
                    u, wc_sXY = stat.mannwhitneyu(y1_sXY, y2_sXY)
                    if y2_sXY.mean() >= y1_sXY.mean():
                        wc_sXY = 50 + (0.5 - wc_sXY) * 100
                    else:
                        wc_sXY = 50 - (0.5 - wc_sXY) * 100
            except:
                wc_sXY = 50

            if self.nf > 355:
                y1_sXY_l = self.df.loc[self.nf - 350:self.nf - 10, "sXY"]
                y2_sXY_l = self.df.loc[self.nf - 10:self.nf - 1, "sXY"]
                try:
                    if y1_sXY_l.equals(y2_sXY_l) == 1:
                        wc_sXY_l = 50
                    else:
                        u, wc_sXY_l = stat.mannwhitneyu(y1_sXY_l, y2_sXY_l)
                        if y2_sXY_l.mean() >= y1_sXY_l.mean():
                            wc_sXY_l = 50 + (0.5 - wc_sXY_l) * 100
                        else:
                            wc_sXY_l = 50 - (0.5 - wc_sXY_l) * 100
                except:
                    wc_sXY_l = 50
            else:
                wc_sXY_l = 50

            if self.nf > 1000:
                y1_sXY_l2 = self.df.loc[self.nf - 950:self.nf - 1, "sXY"]
                y2_sXY_l2 = self.df.loc[self.nf - 150:self.nf - 1, "sXY"]
                try:
                    if y1_sXY_l2.equals(y2_sXY_l2) == 1:
                        wc_sXY_l2 = 50
                    else:
                        u, wc_sXY_l2 = stat.mannwhitneyu(y1_sXY_l2, y2_sXY_l2)
                        if y2_sXY_l2.mean() >= y1_sXY_l2.mean():
                            wc_sXY_l2 = 50 + (0.5 - wc_sXY_l2) * 100
                        else:
                            wc_sXY_l2 = 50 - (0.5 - wc_sXY_l2) * 100
                except:
                    wc_sXY_l2 = 50
            else:
                wc_sXY_l2 = 50


        else:
            wc_sX = 50
            wc_sY = 50
            wc_sXY = 50
            wc_sXY_l = 50
            wc_sXY_l2 = 50
            wc_sXY_ave = 50

        wc_sXY_ave = (wc_sXY_l + wc_sXY_l2) / 2

        self.df.at[self.nf, "PX"] = wc_sX
        self.df.at[self.nf, "PY"] = wc_sY
        self.df.at[self.nf, "PXY"] = wc_sXY
        self.df.at[self.nf, "PXY_l"] = wc_sXY_l
        self.df.at[self.nf, "PXY_l2"] = wc_sXY_l2
        self.df.at[self.nf, "PXY_l_ave"] = wc_sXY_ave

        # PXYm
        if self.nf >= 102:
            PXYm = self.df.loc[self.nf - 50:self.nf - 1, "PXY"].mean()
        else:
            PXYm = 0
        self.df.at[self.nf, "PXYm"] = PXYm

        # nPX, nPY
        if self.nf < 50:
            nPX = 0
            nPY = 0
        if self.nf >= 50:
            nPX = float(3 * self.df.loc[self.nf - 50:self.nf - 1, "x1"].mean() + self.df.loc[self.nf - 50:self.nf - 1,
                                                                                 "x2"].mean()) / 4
            nPY = float(3 * self.df.loc[self.nf - 50:self.nf - 1, "y1"].mean() + self.df.loc[self.nf - 50:self.nf - 1,
                                                                                 "y2"].mean()) / 4

        self.df.at[self.nf, "nPX"] = nPX
        self.df.at[self.nf, "nPY"] = nPY
        nPXY = float(nPX + nPY) / 2
        nPYX = float(nPY - nPX)
        self.df.at[self.nf, "nPXY"] = nPXY
        self.df.at[self.nf, "nPYX"] = nPYX
        # ex.lbl_nPY.setText(str("%0.1f" % nPY))
        # ex.lbl_nPX.setText(str("%0.1f" % nPX))
        # ex.lbl_nPXY.setText(str("%0.1f" % nPXY))

        # defense/offense
        defense_avg = 0
        if self.nf >= 50:
            defense_avg = nPYX / cvol_m_org
        self.df.at[self.nf, "defense_avg"] = defense_avg

        defense_spot = 0
        if self.nf >= 50 and self.df.loc[self.nf - 5:self.nf - 1, "cvol_abs"].mean() != 0:
            nPX_s = float(3 * self.df.loc[self.nf - 5:self.nf - 1, "x1"].mean() + self.df.loc[self.nf - 5:self.nf - 1,
                                                                                 "x2"].mean()) / 4
            nPY_s = float(3 * self.df.loc[self.nf - 5:self.nf - 1, "y1"].mean() + self.df.loc[self.nf - 5:self.nf - 1,
                                                                                 "y2"].mean()) / 4
            nPYX_s = nPY_s - nPX_s
            defense_sopt = float(nPYX_s / self.df.loc[self.nf - 5:self.nf - 1, "cvol_abs"].mean())
        self.df.at[self.nf, "defense_spot"] = defense_spot

        # stXY
        if self.nf < 55:
            stXY = 0
        if self.nf >= 55:
            stXY = self.df.loc[self.nf - 50:self.nf - 1, "sXY"].std()
        self.df.at[self.nf, "stXY"] = stXY

        # # stXYm
        # if nf < 102:
        #     stXYm = 0
        # if nf >= 102:
        #     stXYm = df.loc[nf - 100:nf - 1, "stXY"].mean()
        # df_inst.loc[nf, "stXYm"] = stXYm
        # ex.lbl_stXYm.setText(str("%0.1f" % stXYm))

        # stPX
        if self.nf >= 102:  # and ns < self.nf - 1:
            stPX = self.df.loc[self.nf - 50:self.nf - 1, "PX"].std()
        else:
            stPX = 0
        self.df.at[self.nf, "stPX"] = stPX

        # stPY
        if self.nf >= 102:  # and ns < self.nf - 1:
            stPY = self.df.loc[self.nf - 50:self.nf - 1, "PY"].std()
        else:
            stPY = 0
        self.df.at[self.nf, "stPY"] = stPY

        # stPXY
        if self.nf >= 102:  # and ns < self.nf - 1:
            stPXY = self.df.loc[self.nf - 50:self.nf - 1, "PXY"].std()
        else:
            stPXY = 0
        self.df.at[self.nf, "stPXY"] = stPXY

        # # stPXYm
        # if self.nf >= 352:
        #     stPXYm = self.df.loc[self.nf - 250:self.nf - 1, "stPXY"].mean()
        # else:
        #     stPXYm = 0
        # self.df.at[self.nf, "stPXYm"] = stPXYm

        # PINDEX
        if self.nf < 102:
            pi1 = 0
            pi2 = 0
            pindex = 0
        if self.nf >= 102:
            try:
                pi1 = float(wc_sXY - 50) / 50
                if stPXY != 0:
                    pi2 = float(stat.norm.ppf((100 - stPXY) / 100)) / 2.5
                else:
                    pi2 = 1
                try:
                    normxy = stat.norm.ppf(float(wc_sXY) / 100)
                except:
                    normxy = 0
                try:
                    normx = stat.norm.ppf(float(wc_sX) / 100)
                except:
                    normx = 0
                try:
                    normy = stat.norm.ppf(float(wc_sY) / 100)
                except:
                    normy = 0
                pindex = 50 + 50 * normxy / float(2.5) * abs(normx - normy) / float(5) + 12.5 * pi1 * pi2
            except:
                pindex = 50

        self.df.at[self.nf, "pindex"] = pindex

        # PINDEX1
        if self.nf < 152:
            pindex1 = 0
        if self.nf >= 152:
            pindex1 = self.df.loc[self.nf - 50:self.nf - 1, "pindex"].mean()
        self.df.at[self.nf, "pindex1"] = pindex1

        # PINDEX2
        if self.nf < 102:
            pindex2 = 0
        if self.nf >= 102:
            pindex2 = self.df.loc[self.nf - 100:self.nf - 1, "pindex"].mean()
        self.df.at[self.nf, "pindex2"] = pindex2

        # PINDEX3
        if self.nf < 205:
            pindex3 = 0
        if self.nf >= 205:
            pindex3 = self.df.loc[self.nf - 200:self.nf - 1, "pindex"].mean()
        self.df.at[self.nf, "pindex3"] = pindex3

        # volatility
        if self.cvol_m >= self.cvol_m_start * 0.3 and count_m >= self.count_m_start * 0.3:
            self.vola = 1
        if self.cvol_m >= self.cvol_m_start * 0.5 and count_m >= self.count_m_start * 0.5:
            self.vola = 2
        if self.cvol_m >= self.cvol_m_start * 1 and count_m >= self.count_m_start * 1:
            self.vola = 3

        # gold
        if self.nf >= 550:
            self.prc_per_10 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 10)
            self.prc_per_20 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 15)
            self.df.at[self.nf, "prc_per_10"] = self.prc_per_10

            self.gold1 = 0
            if price > self.prc_per_10:
                self.gold1 = 1
            if price > self.prc_per_20:
                self.gold1 = 2
            self.df.at[self.nf, "gold1"] = self.gold1

            self.gold1_avg = self.df.loc[self.nf - 20:self.nf - 1, "gold1"].mean()
            self.df.at[self.nf, "gold1_avg"] = self.gold1_avg

            self.prc_per_90 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 90)
            self.prc_per_80 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 85)
            self.df.at[self.nf, "prc_per_90"] = self.prc_per_90

            self.gold2 = 0
            if price < self.prc_per_90:
                self.gold2 = -1
            if price < self.prc_per_80:
                self.gold2 = -2
            self.df.at[self.nf, "gold2"] = self.gold2

            self.gold2_avg = self.df.loc[self.nf - 20:self.nf - 1, "gold2"].mean()
            self.df.at[self.nf, "gold2_avg"] = self.gold2_avg

        # rsi_gold
        if self.nf >= 550:
            self.rsi_per_10 = np.percentile(np.array(self.df['rsi'], dtype=float)[self.nf - 500:self.nf - 2], 10)
            self.rsi_per_20 = np.percentile(np.array(self.df['rsi'], dtype=float)[self.nf - 500:self.nf - 2], 15)
            self.df.at[self.nf, "rsi_per_10"] = self.rsi_per_10

            self.rsi_gold1 = 0
            if self.rsi > self.rsi_per_10:
                self.rsi_gold1 = 1
            if self.rsi > self.rsi_per_20:
                self.rsi_gold1 = 2
            self.df.at[self.nf, "rsi_gold1"] = self.rsi_gold1

            self.rsi_gold1_avg = self.df.loc[self.nf - 20:self.nf - 1, "rsi_gold1"].mean()
            self.df.at[self.nf, "rsi_gold1_avg"] = self.rsi_gold1_avg

            self.rsi_per_90 = np.percentile(np.array(self.df['rsi'], dtype=float)[self.nf - 500:self.nf - 2], 90)
            self.rsi_per_80 = np.percentile(np.array(self.df['rsi'], dtype=float)[self.nf - 500:self.nf - 2], 85)
            self.df.at[self.nf, "rsi_per_90"] = self.rsi_per_90

            self.rsi_gold2 = 0
            if self.rsi < self.rsi_per_90:
                self.rsi_gold2 = -1
            if self.rsi < self.rsi_per_80:
                self.rsi_gold2 = -2
            self.df.at[self.nf, "rsi_gold2"] = self.rsi_gold2

            self.rsi_gold2_avg = self.df.loc[self.nf - 20:self.nf - 1, "rsi_gold2"].mean()
            self.df.at[self.nf, "rsi_gold2_avg"] = self.rsi_gold2_avg

        # pvol_gold
        if self.nf >= 550:
            self.pvol_per_10 = np.percentile(np.array(self.df['std_prc_cvol_m'], dtype=float)[self.nf - 500:self.nf - 2], 10)
            self.pvol_per_20 = np.percentile(np.array(self.df['std_prc_cvol_m'], dtype=float)[self.nf - 500:self.nf - 2], 15)
            self.df.at[self.nf, "pvol_per_10"] = self.pvol_per_10

            self.pvol_gold1 = 0
            if self.std_prc_cvol_m > self.pvol_per_10:
                self.pvol_gold1 = 1
            if self.std_prc_cvol_m > self.pvol_per_20:
                self.pvol_gold1 = 2
            self.df.at[self.nf, "pvol_gold1"] = self.pvol_gold1

            self.pvol_gold1_avg = self.df.loc[self.nf - 20:self.nf - 1, "pvol_gold1"].mean()
            self.df.at[self.nf, "pvol_gold1_avg"] = self.pvol_gold1_avg

            self.pvol_per_90 = np.percentile(np.array(self.df['std_prc_cvol_m'], dtype=float)[self.nf - 500:self.nf - 2], 90)
            self.pvol_per_80 = np.percentile(np.array(self.df['std_prc_cvol_m'], dtype=float)[self.nf - 500:self.nf - 2], 85)
            self.df.at[self.nf, "pvol_per_90"] = self.pvol_per_90

            self.pvol_gold2 = 0
            if self.std_prc_cvol_m < self.pvol_per_90:
                self.pvol_gold2 = -1
            if self.std_prc_cvol_m < self.pvol_per_80:
                self.pvol_gold2 = -2
            self.df.at[self.nf, "pvol_gold2"] = self.pvol_gold2

            self.pvol_gold2_avg = self.df.loc[self.nf - 20:self.nf - 1, "pvol_gold2"].mean()
            self.df.at[self.nf, "pvol_gold2_avg"] = self.pvol_gold2_avg

        # p1000_gold
        if self.nf >= 550 and prc_avg_1000 != 0 and prc_std_1000 != 0:
            self.p1000_per_10 = np.percentile(np.array(self.df['std_prc_1000'], dtype=float)[self.nf - 500:self.nf - 2], 10)
            self.p1000_per_20 = np.percentile(np.array(self.df['std_prc_1000'], dtype=float)[self.nf - 500:self.nf - 2], 15)
            self.df.at[self.nf, "p1000_per_10"] = self.p1000_per_10

            self.p1000_gold1 = 0
            if self.std_prc_1000 > self.p1000_per_10:
                self.p1000_gold1 = 1
            if self.std_prc_1000 > self.p1000_per_20:
                self.p1000_gold1 = 2
            self.df.at[self.nf, "p1000_gold1"] = self.p1000_gold1

            self.p1000_gold1_avg = self.df.loc[self.nf - 20:self.nf - 1, "p1000_gold1"].mean()
            self.df.at[self.nf, "p1000_gold1_avg"] = self.p1000_gold1_avg

            self.p1000_per_90 = np.percentile(np.array(self.df['std_prc_1000'], dtype=float)[self.nf - 500:self.nf - 2], 90)
            self.p1000_per_80 = np.percentile(np.array(self.df['std_prc_1000'], dtype=float)[self.nf - 500:self.nf - 2], 85)
            self.df.at[self.nf, "p1000_per_90"] = self.p1000_per_90

            self.p1000_gold2 = 0
            if self.std_prc_1000 < self.p1000_per_90:
                self.p1000_gold2 = -1
            if self.std_prc_1000 < self.p1000_per_80:
                self.p1000_gold2 = -2
            self.df.at[self.nf, "p1000_gold2"] = self.p1000_gold2

            self.p1000_gold2_avg = self.df.loc[self.nf - 20:self.nf - 1, "p1000_gold2"].mean()
            self.df.at[self.nf, "p1000_gold2_avg"] = self.p1000_gold2_avg

        ###### (5) in_signal_prc

        if self.nf >= self.sec_30 + 1:
            if price_mean > price_mean_long * (1 + self.prc_std_per_cri) and self.in_signal_prc != 1:

                if self.df.at[self.nf - 1, "in_signal_prc"] == 0 and self.cvol_m > self.cvol_m_cri_ave:
                    self.in_signal_prc = 1

                self.cvol_m_cri_count += 1
                self.cvol_m_cri_sum += self.cvol_m
                self.cvol_m_cri_ave = self.cvol_m_cri_sum / self.cvol_m_cri_count

                self.cvol_s_cri_count_p += 1
                self.cvol_s_cri_sum_p += cvol_s_std
                self.cvol_s_cri_ave_p = self.cvol_s_cri_sum_p / self.cvol_s_cri_count_p

                self.cvol_t_cri_count_p += 1
                self.cvol_t_cri_sum_p += cvol_t
                self.cvol_t_cri_ave_p = self.cvol_t_cri_sum_p / self.cvol_t_cri_count_p

                self.dxy_200_cri_count_p += 1
                self.dxy_200_cri_sum_p += self.dxy_200_medi  # dxy_20_medi
                self.dxy_200_cri_ave_p = self.dxy_200_cri_sum_p / self.dxy_200_cri_count_p

                self.count_m_act_count += 1
                self.count_m_act_sum += count_m
                self.count_m_act_ave = self.count_m_act_sum / self.count_m_act_count

                self.prc_std_count += 1
                self.prc_std_sum += prc_std
                self.prc_std_ave = self.prc_std_sum / self.prc_std_count

                self.prc_std_1000_count += 1
                self.prc_std_1000_sum += prc_std_1000
                self.prc_std_1000_ave = self.prc_std_1000_sum / self.prc_std_1000_count

                self.std_std_prc_count += 1
                self.std_std_prc_sum += std_std_prc
                self.std_std_prc_ave = self.std_std_prc_sum / self.std_std_prc_count

            if cvol_m_sig == 1 and self.df.at[self.nf - 1, "in_signal_prc"] == 1:
                if price_mean <= price_mean_long * (1 + self.prc_std_per_cri):  # and dxy_20_medi<0
                    self.in_signal_prc = 0

            if price_mean < price_mean_long * (1 - self.prc_std_per_cri) and self.in_signal_prc != -1:

                if self.df.at[self.nf - 1, "in_signal_prc"] == 0 and self.cvol_m > self.cvol_m_cri_ave:
                    self.in_signal_prc = -1

                self.cvol_m_cri_count += 1
                self.cvol_m_cri_sum += self.cvol_m
                self.cvol_m_cri_ave = self.cvol_m_cri_sum / self.cvol_m_cri_count

                self.cvol_s_cri_count_m += 1
                self.cvol_s_cri_sum_m += cvol_s_std
                self.cvol_s_cri_ave_m = self.cvol_s_cri_sum_m / self.cvol_s_cri_count_m

                self.cvol_t_cri_count_m += 1
                self.cvol_t_cri_sum_m += cvol_t
                self.cvol_t_cri_ave_m = self.cvol_t_cri_sum_m / self.cvol_t_cri_count_m

                self.dxy_200_cri_count_m += 1
                self.dxy_200_cri_sum_m += self.dxy_200_medi
                self.dxy_200_cri_ave_m = self.dxy_200_cri_sum_m / self.dxy_200_cri_count_m

                self.count_m_act_count += 1
                self.count_m_act_sum += count_m
                self.count_m_act_ave = self.count_m_act_sum / self.count_m_act_count

                self.prc_std_count += 1
                self.prc_std_sum += prc_std
                self.prc_std_ave = self.prc_std_sum / self.prc_std_count

                self.prc_std_1000_count += 1
                self.prc_std_1000_sum += prc_std_1000
                self.prc_std_1000_ave = self.prc_std_1000_sum / self.prc_std_1000_count

                self.std_std_prc_count += 1
                self.std_std_prc_sum += std_std_prc
                self.std_std_prc_ave = self.std_std_prc_sum / self.std_std_prc_count

            if cvol_m_sig == 1 and self.df.at[self.nf - 1, "in_signal_prc"] == -1:
                if price_mean >= price_mean_long * (1 - self.prc_std_per_cri):  # and dxy_20_medi>0
                    self.in_signal_prc = 0

        self.df.at[self.nf, "in_signal_prc"] = self.in_signal_prc
        if self.nf >= self.sec_30 + 1:
            if self.df.at[self.nf - 1, "in_signal_prc"] == 0 or self.df.at[self.nf - 1, "in_signal_prc"] == -1:
                if self.df.at[self.nf, "in_signal_prc"] == 1 and cvol_t_ox != -1:
                    self.in_sig_price = lblShoga1v
            if self.df.at[self.nf - 1, "in_signal_prc"] == 1:
                if self.df.at[self.nf, "in_signal_prc"] == 0 or self.df.at[self.nf, "in_signal_prc"] == -1:
                    self.out_sig_price = lblBhoga1v
                    self.in_sig_profit = self.out_sig_price - self.in_sig_price
            if self.df.at[self.nf - 1, "in_signal_prc"] == 0 or self.df.at[self.nf - 1, "in_signal_prc"] == 1:
                if self.df.at[self.nf, "in_signal_prc"] == -1 and cvol_t_ox != 1:
                    self.in_sig_price = lblBhoga1v
            if self.df.at[self.nf - 1, "in_signal_prc"] == -1:
                if self.df.at[self.nf, "in_signal_prc"] == 0 or self.df.at[self.nf, "in_signal_prc"] == 1:
                    self.out_sig_price = lblShoga1v
                    self.in_sig_profit = self.in_sig_price - self.out_sig_price
        self.df.at[self.nf, "in_sig_profit"] = self.in_sig_profit

        self.df.at[self.nf, "cvol_m_cri_sum"] = self.cvol_m_cri_sum
        self.df.at[self.nf, "cvol_m_cri_count"] = self.cvol_m_cri_count
        self.df.at[self.nf, "cvol_m_cri_ave"] = self.cvol_m_cri_ave

        self.df.at[self.nf, "cvol_s_cri_sum_p"] = self.cvol_s_cri_sum_p
        self.df.at[self.nf, "cvol_s_cri_count_p"] = self.cvol_s_cri_count_p
        self.df.at[self.nf, "cvol_s_cri_ave_p"] = self.cvol_s_cri_ave_p
        self.df.at[self.nf, "cvol_s_cri_sum_m"] = self.cvol_s_cri_sum_m
        self.df.at[self.nf, "cvol_s_cri_count_m"] = self.cvol_s_cri_count_m
        self.df.at[self.nf, "cvol_s_cri_ave_m"] = self.cvol_s_cri_ave_m

        self.df.at[self.nf, "cvol_t_cri_sum_p"] = self.cvol_t_cri_sum_p
        self.df.at[self.nf, "cvol_t_cri_count_p"] = self.cvol_t_cri_count_p
        self.df.at[self.nf, "cvol_t_cri_ave_p"] = self.cvol_t_cri_ave_p
        self.df.at[self.nf, "cvol_t_cri_sum_m"] = self.cvol_t_cri_sum_m
        self.df.at[self.nf, "cvol_t_cri_count_m"] = self.cvol_t_cri_count_m
        self.df.at[self.nf, "cvol_t_cri_ave_m"] = self.cvol_t_cri_ave_m
        self.cvol_t_cri_ave = (abs(self.cvol_t_cri_ave_p) + abs(self.cvol_t_cri_ave_m)) / 2
        self.df.at[self.nf, "cvol_t_cri_ave_ave"] = self.cvol_t_cri_ave

        self.df.at[self.nf, "dxy_200_cri_sum_p"] = self.dxy_200_cri_sum_p
        self.df.at[self.nf, "dxy_200_cri_count_p"] = self.dxy_200_cri_count_p
        self.df.at[self.nf, "dxy_200_cri_ave_p"] = self.dxy_200_cri_ave_p
        self.df.at[self.nf, "dxy_200_cri_sum_m"] = self.dxy_200_cri_sum_m
        self.df.at[self.nf, "dxy_200_cri_count_m"] = self.dxy_200_cri_count_m
        self.df.at[self.nf, "dxy_200_cri_ave_m"] = self.dxy_200_cri_ave_m
        self.df.at[self.nf, "dxy_200_cri_count_dif"] = self.dxy_200_cri_count_p - self.dxy_200_cri_count_m

        self.df.at[self.nf, "count_m_act_sum"] = self.count_m_act_sum
        self.df.at[self.nf, "count_m_act_count"] = self.count_m_act_count
        self.df.at[self.nf, "count_m_act_ave"] = self.count_m_act_ave

        # # count_m_cri & cvol_s_cri
        # if self.nf < self.min_1+5:
        #     count_m_cri = 0
        #     cvol_s_cri = 0
        # if self.nf >= self.min_1+5:
        #     if prc_std_per < self.prc_std_per_cri:
        #         count_m_cri = self.df.loc[self.nf - self.min_1:self.nf - 1, "count_m"].mean() * 5
        #         cvol_s_cri = self.df.loc[self.nf - self.min_1:self.nf - 1, "cvol_s"].mean() * 5
        #     else:
        #         count_m_cri = self.df.loc[self.nf - self.min_1:self.nf - 1, "count_m"].mean()
        #         cvol_s_cri = self.df.loc[self.nf - self.min_1:self.nf - 1, "cvol_s"].mean()
        # self.df.at[self.nf, "count_m_cri"] = count_m_cri
        # self.df.at[self.nf, "cvol_s_cri"] = cvol_s_cri
        #
        # # cri_effect
        # if self.nf < self.min_1+5:
        #     count_m_eff = 0
        #     cvol_s_eff = 0
        # else:
        #     if count_m > count_m_cri:
        #         count_m_eff = 1
        #     else:
        #         count_m_eff = 0
        #
        #         cvol_s_eff = 0
        #     if cvol_s > cvol_s_cri:
        #         cvol_s_eff = 1
        #     if cvol_s < cvol_s_cri:
        #         cvol_s_eff = -1
        #
        # self.df.at[self.nf, "count_m_eff"] = count_m_eff
        # self.df.at[self.nf, "cvol_s_eff"] = cvol_s_eff

        ###############################
        #  // PIOX //
        ###############################

        # if count_m < self.count_m_overact and abs(self.piox) !=1.5 and abs(self.piox) !=2.5:
        #     if self.piox > 0:
        #         if self.piox == 1 or self.piox == 0.5:
        #             if cvol_t > 0  and count_m<self.count_m_deact:
        #                self.piox = 0
        #         if self.piox == 2:
        #             if cvol_s > 0 and cvol_t > 0:
        #                 self.piox = 0
        #         if self.piox == 3:
        #             if cvol_t > 0:
        #                self.piox = 0
        #         if self.piox == 5 or self.piox == 4:
        #             if count_m < self.count_m_deact:
        #                self.piox = 0
        #
        #     if self.piox < 0:
        #         if self.piox == -1 or self.piox == -0.5:
        #             if cvol_t < 0 and count_m<self.count_m_deact:
        #                self.piox = 0
        #         if self.piox == -2:
        #             if cvol_s < 0 and cvol_t < 0:
        #                 self.piox = 0
        #         if self.piox == -3:
        #             if cvol_t < 0:
        #                self.piox = 0
        #         if self.piox == -5 or self.piox == -4:
        #             if count_m < self.count_m_deact:
        #                self.piox = 0
        #
        # if abs(self.piox) == 1.5:
        #     if count_m<self.count_m_deact:
        #         self.piox = 0

        if self.nf >= 10 and self.nf <= 11:
            self.df.at[10, "piox"] = 2
            self.df.at[11, "piox"] = -2

        if self.nf > 101:
            if count_m < self.count_m_act_ave * 2 and self.df.loc[self.nf - 5:self.nf - 1, "d_OMain"].sum() == 0:
                if self.piox > 0:
                    if dxy_20_medi < 0 and self.cover_signal_2 <= 0:
                        self.piox = 0
                if self.piox < 0:
                    if dxy_20_medi > 0 and self.cover_signal_2 >= 0:
                        self.piox = 0

        ###############################
        #  // In Decision - AI//
        ###############################

        if (self.ai_mode == 1 or self.ai_mode == 3) and self.nf > 215:
            # print("          AI working.....")

            if 1 == 1:

                # (1) Trend Model

                # DATA_PREPARATION
                # x1
                df_ai = self.df.loc[self.nf - self.time_span_lstm - 50: self.nf - 1, self.ai_x]
                try:
                    df_ai = df_ai.dropna(subset=['y_3'])  # 'y1', 'y2', 'y3',
                except:
                    pass
                nX = np.array(df_ai).astype(np.float32)
                rX = []
                rX.append(nX[len(nX) - 1 - self.time_span_lstm:len(nX) - 1])
                rX = np.array(rX)
                # print(rX.shape)

                # x2
                df_ai_2 = self.df.loc[self.nf - self.time_span_lstm - 50: self.nf - 1, self.ai_x_2]
                try:
                    df_ai_2 = df_ai_2.dropna(subset=['y_3'])  # 'y1', 'y2', 'y3',
                except:
                    pass
                nX_2 = np.array(df_ai_2).astype(np.float32)
                rX_2 = []
                rX_2.append(nX_2[len(nX_2) - 1 - self.time_span_lstm:len(nX_2) - 1])
                rX_2 = np.array(rX_2)

                # x3
                df_ai_3 = self.df.loc[self.nf - self.time_span_lstm - 50: self.nf - 1, self.ai_x_3]
                try:
                    df_ai_3 = df_ai_3.dropna(subset=['y_3'])  # 'y1', 'y2', 'y3',
                except:
                    pass
                nX_3 = np.array(df_ai_3).astype(np.float32)
                rX_3 = []
                rX_3.append(nX_3[len(nX_3) - 1 - self.time_span_lstm:len(nX_3) - 1])
                rX_3 = np.array(rX_3)

                # bns
                try:
                    df_ai_bns = self.df.loc[self.nf - self.time_span_lstm - 50: self.nf - 1, self.ai_x_bns]
                    try:
                        df_ai_bns = df_ai_bns.dropna(subset=['y_3'])  # 'y1', 'y2', 'y3',
                    except:
                        pass
                    nX_bns = np.array(df_ai_bns).astype(np.float32)
                    rX_bns = []
                    rX_bns.append(nX_bns[len(nX_bns) - 1 - self.time_span_lstm:len(nX_bns) - 1])
                    rX_bns = np.array(rX_bns)
                except:
                    pass
                # print(rX.shape)

                # Predict
                # SHORT MODE
                print("***************/// AI_SHORT ///***************  ")
                if self.ai_mode == 1 or self.ai_mode == 3:

                    # primary model(x rsi포함)
                    lstm_t = self.loaded_model.predict(rX)
                    self.ai_spot = np.argmax(lstm_t)
                    # print("             *** AI_spot *** : ", self.ai_spot)
                    self.df.at[self.nf, "ai_spot"] = self.ai_spot
                    if self.nf > 235:
                        self.ai = self.df.loc[self.nf - 20:self.nf - 1, "ai_spot"].mean()
                        self.df.at[self.nf, "ai"] = self.ai
                        print("             ***    AI    *** : ", self.ai)
                        self.ai_bns = "n"
                        self.ai_dbns = 0
                        if self.df.loc[self.nf - 30:self.nf - 1, "ai"].mean() < 0.2 and self.ai > 0.2:
                            self.ai_bns = "b"
                            self.ai_dbns = 1
                        if self.df.loc[self.nf - 30:self.nf - 1, "ai"].mean() > 0.8 and self.ai < 0.8:
                            self.ai_bns = "s"
                            self.ai_dbns = -1
                        self.df.at[self.nf, "ai_bns"] = self.ai_bns
                        self.df.at[self.nf, "ai_dbns"] = self.ai_dbns
                        print("             *** AI_BNS *** : ", self.ai_bns)

                    # secondary model(x rsi미포함)
                    if 1 == 0:
                        lstm_t_2 = self.loaded_model_2.predict(rX_2)
                        self.ai_spot_2 = np.argmax(lstm_t_2)
                        # print("             *** AI_spot_2 *** : ", self.ai_spot_2)
                        self.df.at[self.nf, "ai_spot_2"] = self.ai_spot_2
                        if self.nf > 235:
                            self.ai_2 = self.df.loc[self.nf - 20:self.nf - 1, "ai_spot_2"].mean()
                            self.df.at[self.nf, "ai_2"] = self.ai_2
                            print("             ***    AI_2    *** : ", self.ai_2)
                            self.ai_bns_2 = "n"
                            self.ai_dbns_2 = 0
                            if self.ai_2 > 0.8:
                                self.ai_bns_2 = "b"
                                self.ai_dbns_2 = 1
                            if self.ai_2 < 0.2:
                                self.ai_bns_2 = "s"
                                self.ai_dbns_2 = -1
                            self.df.at[self.nf, "ai_bns_2"] = self.ai_bns_2
                            self.df.at[self.nf, "ai_dbns_2"] = self.ai_dbns_2
                            print("             *** AI_BNS_2 *** : ", self.ai_bns_2)

                    # third model(y 변경 2차미분)
                    if 1 == 0:
                        lstm_t_3 = self.loaded_model_3.predict(rX_3)
                        self.ai_spot_3 = np.argmax(lstm_t_3)
                        # print("             *** AI_spot_3 *** : ", self.ai_spot_3)
                        self.df.at[self.nf, "ai_spot_3"] = self.ai_spot_3
                        if self.nf > 235:
                            self.ai_3 = 1 - self.df.loc[self.nf - 20:self.nf - 1, "ai_spot_3"].mean()
                            self.df.at[self.nf, "ai_3"] = self.ai_3
                            print("             ***    AI_3    *** : ", self.ai_3)
                            self.ai_bns_3 = "n"
                            self.ai_dbns_3 = 0
                            if self.ai_3 > 0.8:
                                self.ai_bns_3 = "b"
                                self.ai_dbns_3 = 1
                            if self.ai_3 < 0.2:
                                self.ai_bns_3 = "s"
                                self.ai_dbns_3 = -1
                            self.df.at[self.nf, "ai_bns_3"] = self.ai_bns_3
                            self.df.at[self.nf, "ai_dbns_3"] = self.ai_dbns_3
                            print("             *** AI_BNS_3 *** : ", self.ai_bns_3)

                # LONG mode
                print("***************/// AI_LONG ///***************  ")
                if self.ai_mode == 2 or self.ai_mode == 3:
                    lstm_long_t = self.loaded_long_model.predict(rX)
                    self.ai_long_spot = np.argmax(lstm_long_t)
                    # print("             *** AI_long_spot *** : ", self.ai_long_spot)
                    self.df.at[self.nf, "ai_long_spot"] = self.ai_long_spot
                    if self.nf > 235:
                        self.ai_long = self.df.loc[self.nf - 30:self.nf - 1, "ai_long_spot"].mean()
                        self.df.at[self.nf, "ai_long"] = self.ai_long
                        print("             ***    AI    *** : ", self.ai_long)
                        self.ai_long_bns = "n"
                        self.ai_long_dbns = 0
                        if self.df.loc[self.nf - 30:self.nf - 1, "ai_long"].mean() < 0.4 and self.ai_long > 0.4:
                            self.ai_long_bns = "b"
                            self.ai_long_dbns = 1
                        if self.df.loc[self.nf - 30:self.nf - 1, "ai_long"].mean() > 1.6 and self.ai_long < 1.6:
                            self.ai_long_bns = "s"
                            self.ai_long_dbns = -1
                        self.df.at[self.nf, "ai_long_bns"] = self.ai_long_bns
                        self.df.at[self.nf, "ai_long_dbns"] = self.ai_long_dbns
                        print("             *** AI_long_BNS *** : ", self.ai_long_bns)

                # BNS mode
                if 1 == 0:
                    try:
                        print("***************/// AI_BNS2 ///***************  ")
                        if self.ai_mode == 3:
                            lstm_bns = self.loaded_model_bns.predict(rX_bns)
                            self.ai_bns2_spot = np.argmax(lstm_bns)
                            # print("             *** AI_long_spot *** : ", self.ai_long_spot)
                            self.df.at[self.nf, "ai_bns2_spot"] = self.ai_bns2_spot
                            if self.nf > 235:
                                self.ai_bns2 = self.df.loc[self.nf - 30:self.nf - 1, "ai_bns2_spot"].mean()
                                self.df.at[self.nf, "ai_bns2"] = self.ai_bns2
                                print("             ***    AI_bns2    *** : ", self.ai_bns2)
                                self.ai_bns2_bns = "n"
                                self.ai_bns2_dbns = 0
                                if self.ai_bns2 > 1.6:
                                    self.ai_bns2_bns = "b"
                                    self.ai_bns2_dbns = 1
                                if self.ai_bns2 < 0.4:
                                    self.ai_bns2_bns = "s"
                                    self.ai_bns2_dbns = -1
                                self.df.at[self.nf, "ai_bns2_bns"] = self.ai_bns2_bns
                                self.df.at[self.nf, "ai_bns2_dbns"] = self.ai_bns2_dbns
                                print("             *** AI_bns2_BNS *** : ", self.ai_bns2_bns)
                    except:
                        pass

                # BNS_ai mode
                if 1 == 0:
                    try:
                        print("***************/// BNS_AI ///***************  ")
                        if self.ai_mode == 3:
                            # same input matrix with AI_BNS2(=rX_bns)(just change at fitting yy2(ai_keras_215) : cover_signal_2 -> ai)
                            lstm_bns_ai = self.loaded_model_bns_ai.predict(rX_bns)
                            self.ai_bns2_ai_spot = np.argmax(lstm_bns_ai)
                            # print("             *** AI_long_spot *** : ", self.ai_long_spot)
                            self.df.at[self.nf, "ai_bns2_ai_spot"] = self.ai_bns2_ai_spot
                            if self.nf > 235:
                                self.ai_bns2_ai = self.df.loc[self.nf - 30:self.nf - 1, "ai_bns2_ai_spot"].mean()
                                self.df.at[self.nf, "ai_bns2_ai"] = self.ai_bns2_ai
                                print("             ***    AI_bns2_ai    *** : ", self.ai_bns2_ai)
                                self.ai_bns2_ai_bns = "n"
                                self.ai_bns2_ai_dbns = 0
                                if self.ai_bns2_ai > 1.6:
                                    self.ai_bns2_ai_bns = "b"
                                    self.ai_bns2_ai_dbns = 1
                                if self.ai_bns2_ai < 0.4:
                                    self.ai_bns2_ai_bns = "s"
                                    self.ai_bns2_ai_dbns = -1
                                self.df.at[self.nf, "ai_bns2_ai_bns"] = self.ai_bns2_ai_bns
                                self.df.at[self.nf, "ai_bns2_ai_dbns"] = self.ai_bns2_ai_dbns
                                print("             *** AI_bns2_ai_BNS *** : ", self.ai_bns2_ai_bns)
                    except:
                        pass

            if 1 == 0:

                # (2) Conversion model

                # DATA_PREPARATION
                df_ai = self.df.loc[self.nf - self.time_span_lstm_c - 50: self.nf - 1, self.ai_x]
                nX = np.array(df_ai).astype(np.float32)
                rX = []
                rX.append(nX[len(nX) - 1 - self.time_span_lstm_c:len(nX) - 1])
                rX = np.array(rX)
                # print(rX.shape)

                # Predict
                if 1 == 1:
                    print("***************/// Conversion (cate = 3, x:20) (model_c_x) ///***************  ")
                    # try:
                    lstm_conver = self.model_lstm_c.predict(rX)
                    self.conversion_s = np.argmax(lstm_conver)
                    # print("             *** Conversion_s *** : ", self.conversion_s)
                    self.df.at[self.nf, "conversion_s"] = self.conversion_s
                    if self.nf > 235:
                        self.conversion = self.df.loc[self.nf - 30:self.nf - 1, "conversion_s"].mean()
                        self.df.at[self.nf, "conversion"] = self.conversion
                        print("             ***    conversion    *** : ", self.conversion)
                    # except:
                    # print("AI conversion error")

                if 1 == 1:
                    print("***************/// Conversion_1 (cate = 2, x:20) (model_c_1_x) ///***************  ")
                    # try:
                    lstm_conver_1 = self.model_lstm_c_1.predict(rX)
                    self.conversion_s_1 = np.argmax(lstm_conver_1)
                    # print("             *** Conversion_s_1 *** : ", self.conversion_s_1)
                    self.df.at[self.nf, "conversion_s_1"] = self.conversion_s_1
                    if self.nf > 235:
                        self.conversion_1 = self.df.loc[self.nf - 30:self.nf - 1, "conversion_s_1"].mean()
                        self.df.at[self.nf, "conversion_1"] = self.conversion_1
                        print("             ***    conversion_1    *** : ", self.conversion_1)
                    # except:
                    # print("AI conversion error")

                # DATA_PREPARATION
                df_ai = self.df.loc[self.nf - self.time_span_lstm_alt - 50: self.nf - 1, self.ai_x]
                nX = np.array(df_ai).astype(np.float32)
                rX = []
                rX.append(nX[len(nX) - 1 - self.time_span_lstm_alt:len(nX) - 1])
                rX = np.array(rX)
                # print(rX.shape)

                if 1 == 1 and self.which_market == 1:
                    print("***************/// Conversion_alt (cate = 2, x:50) (model_c_alt_x) ///***************  ")
                    # try:
                    lstm_conver_alt = self.model_lstm_c_alt.predict(rX)
                    self.conversion_s_alt = np.argmax(lstm_conver_alt)
                    # print("             *** Conversion_s_alt *** : ", self.conversion_s_alt)
                    self.df.at[self.nf, "conversion_s_alt"] = self.conversion_s_alt
                    if self.nf > 235:
                        self.conversion_alt = self.df.loc[self.nf - 30:self.nf - 1, "conversion_s_alt"].mean()
                        self.df.at[self.nf, "conversion_alt"] = self.conversion_alt
                        print("             ***    conversion_alt    *** : ", self.conversion_alt)
                    # except:
                    # print("AI conversion error")

                if 1 == 0:
                    print("***************/// Conversion_alt2 (cate = 2) (model_c_alt2_x ///***************  ")
                    # try:
                    lstm_conver_alt2 = self.model_lstm_c_alt2.predict(rX)
                    self.conversion_s_alt2 = np.argmax(lstm_conver_alt2)
                    # print("             *** Conversion_s_alt2 *** : ", self.conversion_s_alt2)
                    self.df.at[self.nf, "conversion_s_alt2"] = self.conversion_s_alt2
                    if self.nf > 235:
                        self.conversion_alt2 = self.df.loc[self.nf - 30:self.nf - 1, "conversion_s_alt2"].mean()
                        self.df.at[self.nf, "conversion_alt2"] = self.conversion_alt2
                        print("             ***    conversion_alt2    *** : ", self.conversion_alt2)
                    # except:
                    # print("AI conversion error")

                # FORCE
                if self.nf >= 600:

                    self.df_conv = self.df.loc[self.nf - 150: self.nf - 1, "conversion"]
                    self.df_conv1_long = self.df.loc[self.nf - 350: self.nf - 50, "conversion_1"]
                    self.df_conv1_short = self.df.loc[self.nf - 50: self.nf - 1, "conversion_1"]

                    if self.which_market == 3:
                        if (self.df_conv1_long[self.df_conv1_long > 0.25].count() >= 1) and (
                        self.df_conv1_short.mean()) <= 0.1:
                            if self.prc_s < 0:  # df_conv[df_conv>0].count() > 1
                                self.force = 1
                            if self.prc_s > 0:  # df_conv1_short.mean() < 0.1 and
                                self.force = -1
                        if self.df_conv[
                            self.df_conv > 0].count() > 1 and self.df_conv.mean() <= 0.05 and self.prc_s < 0:
                            self.force = 2

                    if self.which_market == 4:
                        if (self.df_conv1_long[self.df_conv1_long >= 0.25].count() >= 1) and (
                        self.df_conv1_short.mean()) <= 0.1:
                            if self.prc_s < 0:  # df_conv[df_conv>0].count() > 1
                                self.force = 1
                            if self.prc_s > 0:  # df_conv1_short.mean() == 0 and
                                self.force = -1
                        if self.df_conv[
                            self.df_conv > 0].count() > 1 and self.df_conv.mean() <= 0.05 and self.prc_s < 0:
                            self.force = 2

                    if self.which_market == 1:
                        if (self.df_conv1_long[self.df_conv1_long < 0.7].count() >= 1) and (
                        self.df_conv1_short.mean()) >= 0.95:
                            if self.prc_s > 0:
                                self.force = 1
                        if (self.df_conv[self.df_conv > 0.4].count() >= 1) and (self.df_conv.mean()) <= 0.1:
                            self.force = -1

                        # self.force = 0
                        # if self.conversion_alt >= 0.5 and self.df.loc[self.nf - 20: self.nf - 1,
                        #                                   "conversion_alt"].std() < 0.1:
                        #     self.force = 1
                        # if self.conversion_alt <= 0.5 and self.df.loc[self.nf - 20: self.nf - 1,
                        #                                   "conversion_alt"].std() < 0.1:
                        #     self.force = -1

                    self.df.at[self.nf, "force"] = self.force

        ###############################
        #  // In Decision - pre //
        ###############################

        self.df.at[self.nf, "exed_qty"] = self.exed_qty
        self.df.at[self.nf, "ave_prc"] = self.ave_prc

        if self.nf > self.min_1 + 1:

            # Signal_set
            if dxy_sig == -1 and prc_sig == -1:
                self.signal_set = 1
                # if self.mode == 1:
                #     self.signal_set = -2
            if self.signal_set == 1:  # and self.mode == -1:
                if dxy_sig == 0 or lblBhoga1v - (prc_avg - prc_std) > prc_std * 0.2:
                    self.signal_set = 2

            if dxy_sig == 1 and prc_sig == 1:
                self.signal_set = -1
                # if self.mode == 1:
                #     self.signal_set = 2
            if self.signal_set == -1:  # and self.mode == -1:
                if dxy_sig == 0 or (prc_avg + prc_std) - lblBhoga1v > prc_std * 0.2:
                    self.signal_set = -2
            self.df.at[self.nf, "signal_set"] = self.signal_set

            ### test_signal
            if self.ema_520_count_m >= 1:
                if self.prc_s_peak + self.std_prc_peak == -4 and self.dxy_decay < 0:
                    self.test_signal = 2
                if self.prc_s_peak + self.std_prc_peak == 4 and self.dxy_decay > 0:
                    self.test_signal = -2

            if self.test_signal >= 1 and self.prc_s_peak + self.std_prc_peak >= 0:
                self.test_signal = 0
                if 1 == 1 and self.df.at[self.nf - 1, "test_signal"] == 2 and self.test_signal == 0:
                    self.test_signal = 3
            if self.test_signal <= -1 and self.prc_s_peak + self.std_prc_peak <= 0:
                self.test_signal = 0
                if 1 == 1 and self.df.at[self.nf - 1, "test_signal"] == -2 and self.test_signal == 0:
                    self.test_signal = -3

            # self.df.at[self.nf, "prc_signal"] = 0
            self.df.at[self.nf, "test_signal"] = self.test_signal

            # price_trend
            if self.price_trend != 1:
                if (std_prc >= 0 or prc_s >= 0):  # and self.dxy_decay >= 0:
                    if (self.ai >= 0.8 and self.ai_long >= 1.6):  # and (self.ai_spot == 1 or self.ai_long_spot == 2):
                        self.price_trend = 1  # B_mode
            if self.price_trend != 0:
                if (std_prc <= 0 or prc_s <= 0):  # and self.dxy_decay <= 0:
                    if (
                            self.ai <= 0.2 and self.ai_long <= self.ai_long_low):  # and (self.ai_spot == 0):# and self.ai_long_spot == 0):
                        self.price_trend = 0  # S_mode
            if self.price_trend == 1:
                if std_prc <= 0 or prc_s <= 0 or self.dxy_decay <= 0:
                    if (
                            self.ai <= 0.8 and self.ai_long <= 1.6):  # and (self.ai_spot == 0): # and self.ai_long_spot == 0):
                        self.price_trend = 0.5
            if self.price_trend == 0:
                if std_prc >= 0 or prc_s >= 0 or self.dxy_decay >= 0:
                    if (
                            self.ai >= 0.2 and self.ai_long >= self.ai_long_low):  # and (self.ai_spot == 1 or self.ai_long_spot == 2):
                        self.price_trend = 0.5

            self.df.at[self.nf, "price_trend"] = self.price_trend

            # last_trend
            if self.now_trend != self.price_trend:
                self.last_trend = self.now_trend
                self.now_trend = self.price_trend

            self.df.at[self.nf, "now_trend"] = self.now_trend
            self.df.at[self.nf, "last_trend"] = self.last_trend

            if self.price_trend_s != 1:
                if self.std_prc_slope >= 0 and (self.ai >= 0.9 and self.ai_long >= 1.8):
                    self.price_trend_s = 1  # B_mode
            if self.price_trend_s != 0:
                if self.std_prc_slope <= 0 and (self.ai <= 0.1 and self.ai_long <= self.ai_long_low):
                    self.price_trend_s = 0  # S_mode
            if self.price_trend_s == 1:
                if self.std_prc_slope <= 0 or prc_s <= 0:
                    if (
                            self.ai <= 0.5 and self.ai_long <= 1):  # and (self.ai_spot == 0): # and self.ai_long_spot == 0):
                        self.price_trend_s = 0.5
            if self.price_trend_s == 0:
                if self.std_prc_slope >= 0 or prc_s >= 0:
                    if (self.ai >= 0.5 and self.ai_long >= 1):  # and (self.ai_spot == 1 and self.ai_long_spot == 2):
                        self.price_trend_s = 0.5

            self.df.at[self.nf, "price_trend_s"] = self.price_trend_s

            ### in_hit, in_hit_touch
            ### B_mode
            if self.in_hit == 0:
                if self.test_signal == 2:
                    if (ema_25_count_m == -1 and self.std_prc_peak >= -1 and cvol_t > 0) or prc_s > 0:
                        # self.test_signal = 4
                        self.test_signal_mode = 1
                        self.pre_in_hit = 1
                        if 1 == 1:
                            if self.cover_signal_2 != -1:
                                if self.df.loc[self.nf - 50: self.nf - 1, "count_m"].max() > self.count_m_start * 0.5:
                                    self.in_hit = 1
                                    self.in_hit_touch = 1

                if self.test_signal == 3:
                    self.test_signal_mode = 1
                    self.pre_in_hit = 1
                    if self.cover_signal_2 != -1:
                        self.in_hit = 1
                        self.in_hit_touch = 1

                if self.pre_in_hit == 1:
                    if self.cover_signal_2 != -1:
                        self.in_hit = 1
                        self.in_hit_touch = 1
                        self.pre_in_hit = 0

            if self.in_hit >= 1:
                if ema_520 != -2:
                    self.in_hit = 2
                if self.test_signal != 3:
                    self.in_hit = 0
            if self.in_hit_touch == 1:
                if (self.now_trend - self.last_trend) < 0 and self.std_prc_slope < 0:
                    self.in_hit_touch = 0
                # if self.df.at[self.nf-1, "test_signal"] >= 0 and self.df.at[self.nf, "test_signal"] == 0:
                #     self.in_hit = 2

            ### S_mode
            if self.in_hit == 0:
                if self.test_signal == -2:
                    if (ema_25_count_m == -1 and self.std_prc_peak <= 1 and cvol_t < 0) or prc_s < 0:
                        # self.test_signal = -4
                        self.test_signal_mode = -1
                        self.pre_in_hit = -1
                        if 1 == 1:
                            if self.cover_signal_2 != 1:
                                if self.df.loc[self.nf - 50: self.nf - 1, "count_m"].max() > self.count_m_start * 0.5:
                                    self.in_hit = -1
                                    self.in_hit_touch = -1

                if self.test_signal == -3:
                    self.test_signal_mode = -1
                    self.pre_in_hit = -1
                    if self.cover_signal_2 != 1:
                        self.in_hit = -1
                        self.in_hit_touch = -1

                if self.pre_in_hit == -1:
                    if self.cover_signal_2 != 1:
                        self.in_hit = -1
                        self.in_hit_touch = -1
                        self.pre_in_hit = 0

            if self.in_hit <= -1:
                if ema_520 != 2:
                    self.in_hit = -2
                if self.test_signal != -3:
                    self.in_hit = 0
            if self.in_hit_touch == -1:
                if (self.now_trend - self.last_trend) > 0 and self.std_prc_slope > 0:
                    self.in_hit_touch = 0
                # if self.df.at[self.nf-1, "test_signal"] >= 0 and self.df.at[self.nf, "test_signal"] == 0:
                #     self.in_hit = -2

            self.df.at[self.nf, "test_signal_mode"] = self.test_signal_mode
            self.df.at[self.nf, "pre_in_hit"] = self.pre_in_hit
            self.df.at[self.nf, "in_hit"] = self.in_hit
            self.df.at[self.nf, "in_hit_touch"] = self.in_hit_touch

            ### data ###
            if self.nf > 250:
                self.df_prc_s = self.df.loc[self.nf - 200: self.nf - 1, "prc_s"]
                self.df_std_prc = self.df.loc[self.nf - 150: self.nf - 1, "std_prc_cvol_m"]
                self.df_std_std = self.df.loc[self.nf - 200: self.nf - 1, "std_std_prc_cvol_m"]
                self.df_std_std_s = self.df.loc[self.nf - 75: self.nf - 1, "std_std_prc_cvol_m"]
                self.df_std_std_ss = self.df.loc[self.nf - 50: self.nf - 1, "std_std_prc_cvol_m"]
                if self.nf > 500:
                    self.cvol_m_peak = self.df.loc[self.nf - 100: self.nf - 1, "std_std_prc_cvol_m_peak"]
                self.df_rsi = self.df.loc[self.nf - 200: self.nf - 1, "rsi"]
                self.df_test_signal = self.df.loc[self.nf - 100: self.nf - 1, "test_signal"]
                if self.nf > 500:
                    self.df_gray_strong = self.df.loc[self.nf - 350: self.nf - 1, "gray_strong"]
            if self.nf > 250:
                if self.nf > 310:
                    self.df_bns2 = self.df.loc[self.nf - 300: self.nf - 1, "dOrgMain_new_bns2"]
                # self.df_bns2_2 = self.df.loc[self.nf - 50: self.nf - 1, "dOrgMain_new_bns2"]
                self.df_bns_check = self.df.loc[self.nf - 100: self.nf - 1, "bns_check"]
                self.df_bns_check_s = self.df.loc[self.nf - 50: self.nf - 1, "bns_check"]
                self.df_bns_check_ss = self.df.loc[self.nf - 75: self.nf - 25, "bns_check"]
                self.df_bns_check2 = self.df.loc[self.nf - 100: self.nf - 1, "bns_check_2"]
                self.df_bns_check2s = self.df.loc[self.nf - 20: self.nf - 1, "bns_check_2"]
                # self.df_bns_check2l = self.df.loc[self.nf - 500: self.nf - 1, "bns_check_2"]
                if self.nf > 500:
                    self.df_ai_s = self.df.loc[self.nf - 300: self.nf - 1, "ai"]
                    self.df_rsi_peak = self.df.loc[self.nf - 500: self.nf - 1, "rsi_peak"]
                self.df_rsi_peak_s = self.df.loc[self.nf - 75: self.nf - 1, "rsi_peak"]
                self.df_sum_peak = self.df.loc[self.nf - 250: self.nf - 1, "sum_peak"]
                self.df_cvol_m_sig = self.df.loc[self.nf - 100: self.nf - 1, "cvol_m_sig"]

                self.df_triple = self.df.loc[self.nf - 100: self.nf - 1, "triple_last"]


                self.std_std_prc_cvol_m_peak = 0
                if self.df_std_std[self.df_std_std >= self.std_std_prc_cvol_m_limit].count() >= 1:
                    if self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.5:
                        self.std_std_prc_cvol_m_peak = 1
                    if self.std_std_prc_cvol_m >= self.std_std_prc_cvol_m_limit * 1.5:
                        self.std_std_prc_cvol_m_peak = 2
                    if self.std_std_prc_cvol_m >= self.std_std_prc_cvol_m_limit * 2:
                        self.std_std_prc_cvol_m_peak = 3
                self.df.at[self.nf, "std_std_prc_cvol_m_peak"] = self.std_std_prc_cvol_m_peak

            ##########################
            # // In Decision - main //
            ##########################

            # pre Time Condition
            self.tc = 0
            if self.which_market == 3 and now.hour >= 8 and now.hour < 16:
                self.tc = 1
            if self.which_market == 4:  # and (now.hour < 8 or now.hour > 16):
                self.tc = 1
            if self.which_market == 1:
                self.tc = 1

            # at Forb state => set stat_in_org
            if self.OrgMain == 'n' and self.chkForb == 1:
                self.stat_in_org = "000"
                self.stat_out_org = "111"

            self.bns_check = 0
            self.bns_check_2 = 0

            # bns_check_2
            if 1 == 1 and self.nf >= 520:

                # rsi_peak
                # if (self.rsi_peak > self.df.loc[self.nf - 100: self.nf - 1, "rsi_peak"].mean() + 1.5):
                #     if self.rsi_peak > 0:
                #         self.bns_check_2 = 1
                #
                # if self.rsi_peak < self.df.loc[self.nf - 100: self.nf - 1, "rsi_peak"].mean() - 1.5:
                #     if self.rsi_peak < 0:
                #         self.bns_check_2 = -1

                if self.nf >= 520:
                    if self.rsi_peak == 3 and self.df_rsi_peak_s[self.df_rsi_peak_s == 3].count() == 0:
                        self.bns_check_2 = 1
                    if self.rsi_peak == -3 and self.df_rsi_peak_s[self.df_rsi_peak_s == -3].count() == 0:
                        self.bns_check_2 = -1

                # # bns_check_2 = 1.7
                # if self.which_market == 3 and self.dxy_200_medi > 200:
                #     if self.df.at[self.nf - 1, "bns_check_2"] == 1:
                #         self.bns_check_2 = -1.7
                #         self.bns_check_2_lock = -1
                # if self.which_market == 3 and self.dxy_200_medi < -200:
                #     if self.df.at[self.nf - 1, "bns_check_2"] == -1:
                #         self.bns_check_2 = 1.7
                #         self.bns_check_2_lock = 1

                # bns_check_2 = 2
                # if self.bns_check_2_lock == -1 and self.dxy_200_medi < -50:
                #     self.bns_check_2 = -2
                #     self.bns_check_2_last = -1
                #     if price >= self.lock_price * 1.003:
                #         self.bns_check_2 = -2.2
                #         self.lock_price = 0
                #     if self.bns_check_2_last == -1:
                #         self.lock_price = 0
                #     self.bns_check_2_lock = 0
                #
                # if self.bns_check_2_lock == 1 and self.dxy_200_medi > 50:
                #     self.bns_check_2 = 2
                #     self.bns_check_2_last = 1
                #     if price <= self.lock_price * 0.997:
                #         self.bns_check_2 = 2.2
                #         self.lock_price = 0
                #     if self.bns_check_2_last == 1:
                #         self.lock_price = 0
                #     self.bns_check_2_lock = 0

                # dOrgMain_new_bns2
                if self.dOrgMain_new_bns2 > self.df.loc[self.nf - 100: self.nf - 1, "dOrgMain_new_bns2"].mean() + 1.5:
                    if self.dOrgMain_new_bns2 != 0:
                        self.bns_check_2 = 0.7
                if self.dOrgMain_new_bns2 < self.df.loc[self.nf - 100: self.nf - 1, "dOrgMain_new_bns2"].mean() - 1.5:
                    if self.dOrgMain_new_bns2 != 0:
                        self.bns_check_2 = -0.7

            # auto_cover
            if self.nf > 210 and self.OrgMain == 'n':
                if self.auto_cover == 2:
                    self.in_str = 2
                    self.OrgMain = "b"
                    self.inp = float(lblShoga1v)
                if self.auto_cover == 1:
                    self.in_str = -2
                    self.OrgMain = "s"
                    self.inp = float(lblBhoga1v)
                self.nfset = self.nf
                self.last_o = float(lblShoga1v)
                self.ave_prc = float(lblShoga1v)
                self.exed_qty += 1
                self.type = "tr 0"
                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                              self.type,
                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                self.in_hit = 0
                self.in_hit_touch = 0
                self.stat_in_org = "111"
                self.stat_out_org = "000"

            if self.nf > 210 and self.OrgMain != 'n' and self.auto_cover != 0:
                self.stat_in_org = "111"
                self.stat_out_org = "000"

            ###################################
            # Conseq bns_check
            ###################################

            # (1) via cover_singal_2
            if 1 == 1 and self.nf >= 350:
                self.now_cover_signal2 = 0
                if self.cover_signal_2 == 1 and self.df.at[self.nf - 1, "cover_signal_2"] == 0:
                    self.now_cover_signal2 = 1
                if self.cover_signal_2 == -1 and self.df.at[self.nf - 1, "cover_signal_2"] == 0:
                    self.now_cover_signal2 = -1

                # 1st cover_signal_2
                if self.last_cover_signal2 == 0 and self.now_cover_signal2 == 1:
                    self.bns_check = 0.2
                    # self.bns_check_2 = 0.2
                if self.last_cover_signal2 == 0 and self.now_cover_signal2 == -1:
                    self.bns_check = -0.2
                    # self.bns_check_2 = -0.2

                # second cover_signal_2
                if self.last_cover_signal2 == 1 and self.now_cover_signal2 == 1:
                    if self.df_bns_check[self.df_bns_check == 0.3].count() <= 1:
                        if self.std_std_prc_cvol_m_peak <= 2:
                            self.bns_check = 0.3
                        # self.bns_check_2 = 0.3
                if self.last_cover_signal2 == -1 and self.now_cover_signal2 == -1:
                    if self.df_bns_check[self.df_bns_check == -0.3].count() <= 1:
                        if self.std_std_prc_cvol_m_peak <= 2:
                            self.bns_check = -0.3
                        # self.bns_check_2 = -0.3

                if self.last_cover_signal2 != 1:
                    if self.now_cover_signal2 == 1:
                        self.last_cover_signal2 = 1
                if self.last_cover_signal2 != -1:
                    if self.now_cover_signal2 == -1:
                        self.last_cover_signal2 = -1

            # (2) via prc_s_peak
            if 1 == 0 and self.nf >= 2000:
                self.now_prc_s_peak = 0
                if self.prc_s_peak == 1 and self.df.at[self.nf - 1, "prc_s_peak"] == 0:
                    self.now_prc_s_peak = 1
                if self.prc_s_peak == -1 and self.df.at[self.nf - 1, "prc_s_peak"] == 0:
                    self.now_prc_s_peak = -1

                # second prc_s_peak
                if self.last_prc_s_peak == 1 and self.now_prc_s_peak == 1:
                    self.bns_check = 0.4
                    # self.bns_check_2 = 0.4
                    self.long_prc_s_peak = 1
                if self.last_prc_s_peak == -1 and self.now_prc_s_peak == -1:
                    self.bns_check = -0.4
                    # self.bns_check_2 = -0.4
                    self.long_prc_s_peak = -1

                # conversion after 2 times of prc_s_peak
                # if 1 == 0 and self.nf > 520:
                #     if self.long_prc_s_peak == 1 and self.now_prc_s_peak == -1:
                #         self.long_prc_s_peak = 0
                #         if self.df_rsi_peak[self.df_rsi_peak > 0].count() >= 300 and self.df_rsi_peak[
                #             self.df_rsi_peak < 0].count() <= 100:
                #             self.bns_check = -0.9
                #     if self.long_prc_s_peak == -1 and self.now_prc_s_peak == 1:
                #         self.long_prc_s_peak = 0
                #         if self.df_rsi_peak[self.df_rsi_peak < 0].count() >= 300 and self.df_rsi_peak[
                #             self.df_rsi_peak > 0].count() <= 100:
                #             self.bns_check = 0.9

                if self.last_prc_s_peak != 1:
                    if self.now_prc_s_peak == 1:
                        self.last_prc_s_peak = 1
                if self.last_prc_s_peak != -1:
                    if self.now_prc_s_peak == -1:
                        self.last_prc_s_peak = -1

            # (3) via rsi_peak
            if 1 == 0 and self.nf >= 200:
                # self.now_rsi_peak = 0
                # self.bns_check_2 = 0
                # if self.rsi_peak >= 1 and self.df.at[self.nf - 1, "rsi_peak"] == 0:
                # if self.rsi_peak >= self.df.at[self.nf - 200, "rsi_peak"] + 2:
                if self.rsi_peak > self.df.loc[self.nf - 100: self.nf - 1, "rsi_peak"].mean() + 1.5:
                    # self.now_rsi_peak = 1
                    self.bns_check_2 = 1
                # if self.rsi_peak == -1 and self.df.at[self.nf - 1, "rsi_peak"] == 0:
                # if self.rsi_peak <= self.df.at[self.nf - 200, "rsi_peak"] - 2:
                if self.rsi_peak < self.df.loc[self.nf - 100: self.nf - 1, "rsi_peak"].mean() - 1.5:
                    # self.now_rsi_peak = -1
                    self.bns_check_2 = -1

            if 1 == 1 and self.nf > 1520:
                if self.df_rsi_peak_s[self.df_rsi_peak_s == -3].count() >= 5:
                    if self.rsi_peak >= -1:# and self.df.loc[self.nf - 25: self.nf - 1, "rsi_peak"].mean() < -2.5:
                        self.bns_check = 2.1
                        self.bns_check_2 = 2
                if self.df_rsi_peak_s[self.df_rsi_peak_s == 3].count() >= 5:
                    if self.rsi_peak <= 1:# and self.df.loc[self.nf - 25: self.nf - 1, "rsi_peak"].mean() > 2.5:
                        self.bns_check = -2.1
                        self.bns_check_2 = -2

                # # 1st rsi_peak
                # if self.bns_check == 0 and self.now_rsi_peak == 1:
                #     self.bns_check = 0.1
                # if self.bns_check == 0 and self.now_rsi_peak == -1:
                #     self.bns_check = -0.1

                # # 2nd rsi_peak
                # if self.last_rsi_peak == 1 and self.now_rsi_peak == 1:
                #     self.bns_check_2 = 0.1
                # if self.last_rsi_peak == -1 and self.now_rsi_peak == -1:
                #     self.bns_check_2 = -0.1

                # if self.last_rsi_peak != 1:
                #     if self.now_rsi_peak == 1:
                #         self.last_rsi_peak = 1
                # if self.last_rsi_peak != -1:
                #     if self.now_rsi_peak == -1:
                #         self.last_rsi_peak = -1

            # (4) via sum_peak
            # if 1 == 0 and self.nf >= 550:
            #     if self.sum_peak == 3 or self.sum_peak == 2:
            #         if self.df_sum_peak[self.df_sum_peak >= 3].count() >= 1:
            #             if self.sum_peak < self.df.at[self.nf - 1, "sum_peak"]:
            #                 self.bns_check = -0.5
            #     if self.sum_peak == -3 or self.sum_peak == -2:
            #         if self.df_sum_peak[self.df_sum_peak <= -3].count() >= 1:
            #             if self.sum_peak > self.df.at[self.nf - 1, "sum_peak"]:
            #                 self.bns_check = 0.5

            # (5) via dOrgMain_new_bns2 + ai
            # if 1 == 0 and self.nf >= 550:
            #     if self.df.at[self.nf - 2, "dOrgMain_new_bns2"] == -3 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] != -3:
            #         if self.which_market == 3 or self.which_market == 1:
            #             if self.df.loc[self.nf - 325: self.nf - 25, "ai"].mean() <= 0.05:
            #                 # if self.df.loc[self.nf - 20: self.nf - 1, "ai"].mean() >= 0.1:
            #                 self.bns_check = 0.6
            #         if self.which_market == 4:
            #             if self.df.loc[self.nf - 325: self.nf - 25, "ai_long"].mean() <= 0.1:
            #                 # if self.df.loc[self.nf - 20: self.nf - 1, "ai_long"].mean() > 0.1:
            #                 self.bns_check = 0.6
            #
            #     if self.df.at[self.nf - 2, "dOrgMain_new_bns2"] == 3 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] != 3:
            #         if self.which_market == 3 or self.which_market == 1:
            #             if self.df.loc[self.nf - 325: self.nf - 25, "ai"].mean() >= 0.5:
            #                 # if self.df.loc[self.nf - 20: self.nf - 1, "ai"].mean() < 0.1:
            #                 self.bns_check = -0.6
            #         if self.which_market == 4:
            #             if self.df.loc[self.nf - 325: self.nf - 25, "ai_long"].mean() >= 0.5:
            #                 # if self.df.loc[self.nf - 20: self.nf - 1, "ai_long"].mean() < 0.2:
            #                 self.bns_check = -0.6

                # (6) dOrgMain_new_bns2
            if 1 == 1 and self.nf >= 550:
                if self.df.loc[self.nf - 300:self.nf - 1, "dOrgMain_new_bns2"].mean() <= -1.3:
                    if self.df.loc[self.nf - 300:self.nf - 1, "cover_signal_2"].mean() <= -0.25:
                        if self.df.loc[self.nf - 300:self.nf - 1, "sum_peak"].mean() < -1:
                            if self.dOrgMain_new_bns2 >= 0 and self.cover_signal_2 >= 0:
                                if self.sum_peak >= 2:
                                    self.bns_check = 0.7
                if self.df.loc[self.nf - 300:self.nf - 1, "dOrgMain_new_bns2"].mean() >= 1.3:
                    if self.df.loc[self.nf - 300:self.nf - 1, "cover_signal_2"].mean() >= 0.25:
                        if self.df.loc[self.nf - 300:self.nf - 1, "sum_peak"].mean() > 1:
                            if self.dOrgMain_new_bns2 <= 0 and self.cover_signal_2 <= 0:
                                if self.sum_peak <= -2:
                                    self.bns_check = -0.7
            # if 1 == 0 and self.nf >= 550:
            #     if self.df.at[self.nf - 2, "dOrgMain_new_bns2"] == -3 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] != -3:
            #         self.bns_check = 0.7
            #     if self.df.at[self.nf - 2, "dOrgMain_new_bns2"] == 3 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] != 3:
            #         self.bns_check = -0.7


                # (7) test_signal(only bit)
            if 1 == 1 and self.nf >= 550 and self.which_market == 1:
                if self.df.at[self.nf - 2, "test_signal"] <= -3 and self.df.at[self.nf - 1, "test_signal"] > -3:
                    self.bns_check = -0.8
                if self.df.at[self.nf - 2, "test_signal"] >= 3 and self.df.at[self.nf - 1, "test_signal"] > 3:
                    self.bns_check = 0.8

                # (8) gold_signal
            if 1 == 0 and self.nf >= 550:# and self.which_market == 1:
                if self.df.loc[self.nf - 100:self.nf - 40, "gold1"].mean() <= 0.5:
                    if self.df.loc[self.nf - 30:self.nf - 1, "gold1"].mean() >= 1.9:
                        self.bns_check = 0.9
                        # self.bns_check_3 = 0.9
                if self.df.loc[self.nf - 100:self.nf - 40, "gold2"].mean() >= -0.3:
                    if self.df.loc[self.nf - 30:self.nf - 1, "gold2"].mean() <= -1.9:
                        self.bns_check = -0.9
                        # self.bns_check_3 = -0.9

                # (9) rsi_signal
            if 1 == 1 and self.nf >= 550:  # and self.which_market == 1:
                if self.df.loc[self.nf - 100:self.nf - 40, "rsi_gold1"].mean() <= 0.5:
                    if self.df.loc[self.nf - 30:self.nf - 1, "rsi_gold1"].mean() >= 1.9:
                        self.bns_check = 0.93
                        # self.bns_check_3 = 0.93
                if self.df.loc[self.nf - 100:self.nf - 40, "rsi_gold2"].mean() >= -0.3:
                    if self.df.loc[self.nf - 30:self.nf - 1, "rsi_gold2"].mean() <= -1.9:
                        self.bns_check = -0.93
                        # self.bns_check_3 = -0.93

                # (11) pvol_signal
            if 1 == 1 and self.nf >= 550:  # and self.which_market == 1:
                if self.df.loc[self.nf - 100:self.nf - 40, "pvol_gold1"].mean() <= 0.5:
                    if self.df.loc[self.nf - 30:self.nf - 1, "pvol_gold1"].mean() >= 1.9:
                        self.bns_check = 0.96
                        # self.bns_check_3 = 0.96
                if self.df.loc[self.nf - 100:self.nf - 40, "pvol_gold2"].mean() >= -0.3:
                    if self.df.loc[self.nf - 30:self.nf - 1, "pvol_gold2"].mean() <= -1.9:
                        self.bns_check = -0.96
                        # self.bns_check_3 = -0.96

                # (12) after bns_check = 3
            if 1 == 1 and self.nf >= 550 :
                if self.bns_check_4 == 1 and self.df_bns_check[self.df_bns_check <= -3].count() >= 20:
                    self.bns_check = 2.1
                if self.bns_check_4 == -1 and self.df_bns_check[self.df_bns_check >= 3].count() >= 20:
                    self.bns_check = -2.1

            # bns_check_2 = 1.7
            # if 1 == 0 and self.nf >= 550:
            #     if self.df.at[self.nf - 2, "bns_check_2"] == -1.5 and self.df.at[self.nf - 1, "bns_check_2"] != -1.5:
            #         self.bns_check_2 = -1.7
            #     if self.df.at[self.nf - 2, "bns_check_2"] == 1.5 and self.df.at[self.nf - 1, "bns_check_2"] != 1.5:
            #         self.bns_check_2 = 1.7
            #
            #     if abs(self.bns_check_2) == 0.3 or abs(self.bns_check_2) == 0.4:
            #         self.check2_pvt_point = self.nf
            #
            # self.wc_pvt = 50
            # if self.nf >= self.check2_pvt_point + 25 and self.nf >= 550:
            #     if self.df_bns_check2[
            #         abs(self.df_bns_check2) == 0.3].count() >= 1:  # or self.df_bnlsts_check2[abs(self.df_bns_check2) == 0.4].count() >= 1:
            #         before_pvt = self.df.loc[self.check2_pvt_point - 100:self.check2_pvt_point, "sXY"]
            #         after_pvt = self.df.loc[self.check2_pvt_point:self.nf - 1, "sXY"]
            #         try:
            #             if before_pvt.equals(after_pvt) == 1:
            #                 self.wc_pvt = 50
            #             else:
            #                 uu, wc = stat.mannwhitneyu(before_pvt, after_pvt)
            #                 if after_pvt.mean() >= before_pvt.mean():
            #                     self.wc_pvt = wc * 100  # 50 + (0.5 - wc) * 100
            #                 else:
            #                     self.wc_pvt = wc * 100  # 50 - (0.5 - wc) * 100
            #         except:
            #             self.wc_pvt = 50
            #
            # self.df.at[self.nf, "check2_pvt_point"] = self.check2_pvt_point
            # self.df.at[self.nf, "wc_pvt"] = self.wc_pvt

            ###################################
            # MAIN BNS
            ###################################

            # Time Condition
            if self.nf > 530 and self.cover_order_exed == 0 and (self.tc == 1 or self.chkForb == 1):
                # Position Condition(main)
                if 1 == 1 or (self.which_market == 1 or (
                        self.stat_in_org == "000" and self.stat_out_org == "111")) or self.chkForb == 1:
                    # Position Condition(nprob)
                    if 1 == 1 or (self.df.at[
                                      self.nf - 3, "OrgMain"] == 'n' and self.OrgMain == 'n') or self.bns_check_mode == 1:

                        ###################################
                        # BUY
                        ###################################
                        self.bns_check_3 = 0
                        self.bns_check_5 = 0
                        self.bns_check_6 = 0

                        # bns_check in
                        if self.df_bns_check[self.df_bns_check >= 0.3].count() >= 1 and self.nf > self.nfset + 25 and self.nfset != 0: # and self.df.at[self.nf-1, "bns_check"] != 1):# or self.bns_check == 0.7):
                            if self.df_bns_check[self.df_bns_check >= 0.2].count() >= 1 and self.df_bns_check[self.df_bns_check < 0].count() == 0:
                                if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.at[self.nfset, "std_prc_cvol_m"] * 0.98:
                                    if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                        if (self.which_market == 1 or (self.which_market == 3 and self.stat_in_org == "000" and self.stat_out_org == "111")):
                                            self.in_str = 2
                                            self.OrgMain = "b"
                                            self.nfset = self.nf
                                            self.inp = float(lblShoga1v)
                                            self.last_o = float(lblShoga1v)
                                            self.ave_prc = float(lblShoga1v)
                                            self.exed_qty += 1
                                            self.type = "tr 0"
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.in_hit = 0
                                            self.in_hit_touch = 0
                                # else:
                                #     self.bns_check_3 = -1.5

                            # if self.df.loc[self.nf - 25:self.nf - 5, "std_prc_cvol_m"].mean() <= self.df.at[self.nfset, "std_prc_cvol_m"] * 0.9:
                            #     self.bns_check = -0.1

                        if self.df_bns_check[self.df_bns_check <= -0.3].count() >= 1 and self.nf > self.nfset + 25 and self.nfset != 0: #and self.df.at[self.nf-1, "bns_check"] != -1):# or self.bns_check == -0.7):
                            if self.df_bns_check[self.df_bns_check <= -0.2].count() >= 1 and self.df_bns_check[self.df_bns_check > 0].count() == 0:
                                if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.at[self.nfset, "std_prc_cvol_m"] * 1.02:
                                    if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                        if (self.which_market == 1 or (self.which_market == 3 and self.stat_in_org == "000" and self.stat_out_org == "111")):
                                            self.in_str = -2
                                            self.OrgMain = "s"
                                            self.nfset = self.nf
                                            self.inp = float(lblShoga1v)
                                            self.last_o = float(lblShoga1v)
                                            self.ave_prc = float(lblShoga1v)
                                            self.exed_qty += 1
                                            self.type = "tr 0"
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.in_hit = 0
                                            self.in_hit_touch = 0
                                            self.init5_touched = 0
                                # else:
                                #     self.bns_check_3 = 1.5

                        # bns_check_2 display
                        if self.df_bns_check2[self.df_bns_check2 == -1].count() >= 1 and self.bns_check_2 == 0:
                            if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 0.8].count() >= 1 and self.std_std_prc_cvol_m <= self.std_std_prc_cvol_m_limit * 0.8:
                                if self.std_prc_cvol_m > self.std_prc_cvol_m_limit * -1.25 and self.dxy_200_medi < 0:
                                    if self.df_bns_check2[self.df_bns_check2 == -1.5].count() == 0:
                                    # if self.std_prc_cvol_m >= self.df.loc[self.nf - 25:self.nf - 5, "std_prc_cvol_m"].mean():
                                        self.bns_check_2 = 1.5
                                        self.bns_check_2_lock = 1
                                        if self.lock_price == 0:
                                            self.lock_price = price
                                        # if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                        #     if (self.which_market == 1 or (self.which_market == 3 and self.stat_in_org == "000" and self.stat_out_org == "111")):
                                        #         self.in_str = 2
                                        #         self.OrgMain = "b"
                                        #         self.nfset = self.nf
                                        #         self.inp = float(lblShoga1v)
                                        #         self.last_o = float(lblShoga1v)
                                        #         self.ave_prc = float(lblShoga1v)
                                        #         self.exed_qty += 1
                                        #         self.type = "tr 0_2"
                                        #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                        #                       self.type,
                                        #                       self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                        #         self.in_hit = 0
                                        #         self.in_hit_touch = 0

                        if self.df_bns_check2[self.df_bns_check2 == 1].count() >= 1 and self.bns_check_2 == 0:
                            if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 0.8].count() >= 1 and self.std_std_prc_cvol_m <= self.std_std_prc_cvol_m_limit * 0.8:
                                if self.std_prc_cvol_m < self.std_prc_cvol_m_limit * 1.25 and self.dxy_200_medi > 0:
                                    if self.df_bns_check2[self.df_bns_check2 == 1.5].count() == 0:
                                    # if self.std_prc_cvol_m <= self.df.loc[self.nf - 25:self.nf - 5, "std_prc_cvol_m"].mean():
                                        self.bns_check_2 = -1.5
                                        self.bns_check_2_lock = -1
                                        if self.lock_price == 0:
                                            self.lock_price = price
                                        # if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                        #     if (self.which_market == 1 or (self.which_market == 3 and self.stat_in_org == "000" and self.stat_out_org == "111")):
                                        #         self.in_str = -2
                                        #         self.OrgMain = "s"
                                        #         self.nfset = self.nf
                                        #         self.inp = float(lblShoga1v)
                                        #         self.last_o = float(lblShoga1v)
                                        #         self.ave_prc = float(lblShoga1v)
                                        #         self.exed_qty += 1
                                        #         self.type = "tr 0_2"
                                        #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                        #                       self.type,
                                        #                       self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                        #         self.in_hit = 0
                                        #         self.in_hit_touch = 0
                                        #         self.init5_touched = 0

                        # bns_check_2 in
                        if self.nf >= 1000 and self.df_bns_check2[self.df_bns_check2 == 1.5].count() >= 1 and self.bns_check_2 == 0:
                            if self.df.loc[self.nf - 10:self.nf - 1, "bns_check_2"].mean() == 0:
                                if self.df_bns_check2[self.df_bns_check2 == 2.5].count() == 0:
                                    self.bns_check_2 = 2.5
                                    if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                        if (self.which_market == 1 or (self.which_market == 3 and self.stat_in_org == "000" and self.stat_out_org == "111")):
                                            self.in_str = 2
                                            self.OrgMain = "b"
                                            self.nfset = self.nf
                                            self.inp = float(lblShoga1v)
                                            self.last_o = float(lblShoga1v)
                                            self.ave_prc = float(lblShoga1v)
                                            self.exed_qty += 1
                                            self.type = "tr 0_2"
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.in_hit = 0
                                            self.in_hit_touch = 0

                        if self.nf >= 1000 and self.df_bns_check2[self.df_bns_check2 == -1.5].count() >= 1 and self.bns_check_2 == 0:
                            if self.df.loc[self.nf - 10:self.nf - 1, "bns_check_2"].mean() == 0:
                                if self.df_bns_check2[self.df_bns_check2 == -2.5].count() == 0:
                                    self.bns_check_2 = -2.5
                                    if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                        if (self.which_market == 1 or (self.which_market == 3 and self.stat_in_org == "000" and self.stat_out_org == "111")):
                                            self.in_str = 2
                                            self.OrgMain = "s"
                                            self.nfset = self.nf
                                            self.inp = float(lblShoga1v)
                                            self.last_o = float(lblShoga1v)
                                            self.ave_prc = float(lblShoga1v)
                                            self.exed_qty += 1
                                            self.type = "tr 0_2"
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.in_hit = 0
                                            self.in_hit_touch = 0


                        ###################################
                        # Trend init 1(long ai_long high + not prc_s_peak peak)
                        ###################################
                        if self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 2 and self.df_bns2[
                            self.df_bns2 < 0].count() >= 10 and self.df_rsi_peak[self.df_rsi_peak < 0].count() >= 15:

                            # Orginal TR
                            if (self.which_market != 4 and self.ai >= 0.2) or (
                                    self.which_market == 4 and self.ai_long >= 0.4):

                                if self.ai_long >= 1 and self.df.loc[self.nf - 50:self.nf - 1, "ai_long"].mean() >= 1:
                                    # if self.df.loc[self.nf - 100:self.nf - 20, "ai"].mean() <= 0.5 and self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() > 0.5:
                                    if self.prc_s_peak != 2 and self.df_std_std[
                                        self.df_std_std >= self.std_std_prc_cvol_m_limit * 0.8].count() == 0:
                                        if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.2 :
                                            if self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.6:
                                                if self.df.loc[self.nf - 10:self.nf - 1,"rsi"].mean() >= self.df.loc[self.nf - 50:self.nf - 10,
                                                                        "rsi"].mean():
                                                    self.bns_check = 1
                                                    if self.which_market == 3 or self.which_market != 3:  # and self.df.loc[self.nf - 50:self.nf - 1,"count_m_per_80"].mean() >= 5
                                                        if self.bns_check_mode != 1 or (
                                                                self.OrgMain == 'n' and self.df.at[
                                                            self.nf - 3, "OrgMain"] == 'n'):
                                                            if (self.which_market == 1 or (
                                                                    self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                                if self.df.loc[self.nf - 50:self.nf - 1,
                                                                   "d_OMain"].mean() == 0:
                                                                    if self.df_bns_check[
                                                                        self.df_bns_check >= 1].count() >= 5:
                                                                        self.in_str = 2
                                                                        self.OrgMain = "b"
                                                                        self.nfset = self.nf
                                                                        self.inp = float(lblShoga1v)
                                                                        self.last_o = float(lblShoga1v)
                                                                        self.ave_prc = float(lblShoga1v)
                                                                        self.exed_qty += 1
                                                                        self.type = "tr 1"
                                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v,
                                                                                      self.exed_qty, self.OrgMain,
                                                                                      self.type,
                                                                                      self.profit, self.profit_opt,
                                                                                      self.mode, self.ave_prc, prc_o1)
                                                                        self.in_hit = 0
                                                                        self.in_hit_touch = 0
                                                                        self.init5_touched = 0

                                ###################################
                                # Trend init 2(PXY_l_ave)
                                ###################################
                                if self.nf >= 500 and self.df.loc[self.nf - 100:self.nf - 1, "PXY_l_ave"].mean() >= 80:
                                    if self.prc_s_peak != 2 and self.std_prc_cvol_m > -5:
                                        if self.std_std_prc_cvol_m_peak == 0 and self.df.loc[self.nf - 500:self.nf - 1,
                                                                                 "std_std_prc_cvol_m_peak"].mean() > 0:
                                            if self.df.loc[self.nf - 100:self.nf - 1,
                                               "std_prc_peak_1000"].mean() >= 0 and self.std_prc_peak_1000 >= 0:
                                                if self.std_prc_cvol_m >= self.df.loc[self.nf - 250:self.nf - 1,
                                                                          "std_prc_cvol_m"].max() * 0.8:
                                                    self.bns_check = 1.3
                                                    if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[
                                                        self.nf - 3, "OrgMain"] == 'n'):
                                                        if (self.which_market == 1 or (
                                                                self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                            if self.dOrgMain_new_bns2 > 0 and self.rsi_peak >= 0 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.8:
                                                                if self.df.loc[self.nf - 50:self.nf - 1,
                                                                   "d_OMain"].mean() == 0:
                                                                    if self.df_bns_check[
                                                                        self.df_bns_check >= 1].count() >= 5:
                                                                        self.in_str = 2
                                                                        self.OrgMain = "b"
                                                                        self.nfset = self.nf
                                                                        self.inp = float(lblShoga1v)
                                                                        self.last_o = float(lblShoga1v)
                                                                        self.ave_prc = float(lblShoga1v)
                                                                        self.exed_qty += 1
                                                                        self.type = "tr 2"
                                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v,
                                                                                      self.exed_qty, self.OrgMain,
                                                                                      self.type, self.profit,
                                                                                      self.profit_opt, self.mode,
                                                                                      self.ave_prc, prc_o1)
                                                                        self.in_hit = 0
                                                                        self.in_hit_touch = 0
                                                                        self.init5_touched = 0
                                                # else:
                                                #     self.bns_check = -1.3
                                                #     if self.bns_check_mode != 1 or self.OrgMain == 'n':
                                                #         self.in_str = -2
                                                #         self.OrgMain = "s"
                                                #         self.nfset = self.nf
                                                #         self.inp = float(lblBhoga1v)
                                                #         self.last_o = float(lblBhoga1v)
                                                #         self.ave_prc = float(lblBhoga1v)
                                                #         self.exed_qty += 1
                                                #         self.type = "tr 2 conv"
                                                #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                #                       self.type,
                                                #                       self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                                #         self.in_hit = 0
                                                #         self.in_hit_touch = 0
                                                #         self.init5_touched = 0
                                ###################################
                                # Trend init 3(dOrgMain_new_bns2)
                                ###################################
                                if self.df.loc[self.nf - 50:self.nf - 1,
                                   "dOrgMain_new_bns2"].mean() <= -1 and self.dOrgMain_new_bns2 >= 0 and self.std_prc_cvol_m > -5:
                                    if self.std_prc_1000 > 0 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.4:
                                        self.bns_check = 1.5
                                        if self.bns_check_mode != 1 or (
                                                self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                            if (self.which_market == 1 or (
                                                    self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                if self.rsi_peak <= -1 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.4:
                                                    if self.df.loc[self.nf - 50:self.nf - 1, "d_OMain"].mean() == 0:
                                                        self.in_str = 2
                                                        self.OrgMain = "b"
                                                        self.nfset = self.nf
                                                        self.inp = float(lblBhoga1v)
                                                        self.last_o = float(lblBhoga1v)
                                                        self.ave_prc = float(lblBhoga1v)
                                                        self.exed_qty += 1
                                                        self.type = "tr 3"
                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                                      self.OrgMain,
                                                                      self.type,
                                                                      self.profit, self.profit_opt, self.mode,
                                                                      self.ave_prc, prc_o1)
                                                        self.in_hit = 0
                                                        self.in_hit_touch = 0
                                                        self.init5_touched = 0
                                if 1==0 and self.df.loc[self.nf - 50:self.nf - 1,"dOrgMain_new_bns2"].mean() >= 0 and self.dOrgMain_new_bns2 >= 0:
                                    if self.std_prc_1000 > 0 and self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.8:
                                        if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 0.8].count() == 0:
                                            self.bns_check = -1.8
                                            if self.bns_check_mode != 1 or (
                                                    self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                                if (self.which_market == 1 or (
                                                        self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                    if self.rsi_peak <= 1 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.8:
                                                        if self.df.loc[self.nf - 50:self.nf - 1, "d_OMain"].mean() == 0:
                                                            self.in_str = -2
                                                            self.OrgMain = "s"
                                                            self.nfset = self.nf
                                                            self.inp = float(lblShoga1v)
                                                            self.last_o = float(lblShoga1v)
                                                            self.ave_prc = float(lblShoga1v)
                                                            self.exed_qty += 1
                                                            self.type = "tr 3 conv"
                                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v,
                                                                          self.exed_qty, self.OrgMain,
                                                                          self.type, self.profit,
                                                                          self.profit_opt, self.mode,
                                                                          self.ave_prc, prc_o1)
                                                            self.in_hit = 0
                                                            self.in_hit_touch = 0
                                                            self.init5_touched = 0


                        ###################################
                        # Conversion init 1(prc_s_limit)
                        ###################################

                        self.prc_per_10 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 10)
                        self.prc_per_20 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 10)
                        self.df.at[self.nf, "prc_per_10"] = self.prc_per_10

                        # self.gold1 = 0
                        # if price > self.prc_per_10:
                        #     self.gold1 = 1
                        # if price > self.prc_per_20:
                        #     self.gold1 = 2
                        # self.df.at[self.nf, "gold1"] = self.gold1
                        # self.gold1_avg = self.df.loc[self.nf - 10:self.nf - 1, "gold1"].mean()
                        # self.df.at[self.nf, "gold1_avg"] = self.gold1_avg

                        if self.nf > 800 and (
                                self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 1 and self.df_std_std[
                            self.df_std_std >= self.std_std_prc_cvol_m_limit * 1.25].count() > 1):
                            if (self.df.loc[self.nf - 30:self.nf - 10,
                                "std_prc_peak_1000"].mean() <= -1 and self.std_prc_peak == 0) or (self.nf > 2000 and self.rsi < 43):
                                if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.3:
                                    if self.std_std_prc_cvol_m < self.df.loc[self.nf - 100:self.nf - 1, "std_std_prc_cvol_m"].max()*0.3:
                                        self.bns_check = 3
                                        if self.bns_check_mode != 1 or (
                                                self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                            if (self.which_market == 1 or (
                                                    self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                # if self.df_bns_check_ss[self.df_bns_check_ss == 3].count() == 0:
                                                self.in_str = 2
                                                self.OrgMain = "b"
                                                self.nfset = self.nf
                                                self.inp = float(lblShoga1v)
                                                self.last_o = float(lblShoga1v)
                                                self.ave_prc = float(lblShoga1v)
                                                self.exed_qty += 1
                                                self.type = "conv 0"
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                              self.type,
                                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                                self.in_hit = 0
                                                self.in_hit_touch = 0
                                                self.init5_touched = 0

                            if (self.which_market != 4 and self.ai >= 0.1) or (
                                    self.which_market == 4 and self.ai_long >= 0.2):

                                if (self.which_market != 4 and self.df.loc[self.nf - 160:self.nf - 10,
                                                               "ai"].mean() < 0.1) or (
                                        self.which_market == 4 and self.df.loc[self.nf - 60:self.nf - 10,
                                                                   "ai_long"].mean() < 0.2):
                                    if self.prc_s != 0 and self.prc_s > (self.prc_s_limit * -1) and self.df_prc_s[
                                        self.df_prc_s < self.prc_s_limit * -1].count() > 1:
                                        if self.df_std_std[
                                            self.df_std_std >= self.std_std_prc_cvol_m_limit * 1].count() > 1:  # and (self.ai >= 0.9): # or self.ai_long >= 1.8):
                                            # self.test_signal = 5
                                            if self.dOrgMain_new_bns2 <= 0:
                                                self.bns_check = 2
                                                if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[
                                                    self.nf - 3, "OrgMain"] == 'n'):
                                                    if (self.which_market == 1 or (
                                                            self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                        self.in_str = 2
                                                        self.OrgMain = "b"
                                                        self.nfset = self.nf
                                                        self.inp = float(lblShoga1v)
                                                        self.last_o = float(lblShoga1v)
                                                        self.ave_prc = float(lblShoga1v)
                                                        self.exed_qty += 1
                                                        self.type = "conv 1"
                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                                      self.OrgMain, self.type,
                                                                      self.profit, self.profit_opt, self.mode,
                                                                      self.ave_prc, prc_o1)
                                                        self.in_hit = 0
                                                        self.in_hit_touch = 0
                                                        self.init5_touched = 0

                                ###################################
                                # Conversion init 2(long ai low -> prc_s_peak up)
                                ###################################
                                # if self.df.loc[self.nf - 250:self.nf - 25, "ai"].mean() <= 0.1 and self.ai_long >= 1:
                                #     if self.df.loc[self.nf - 100:self.nf - 25,
                                #        "prc_s_peak"].mean() < -0.8 and self.prc_s_peak == 0:
                                #         if self.dOrgMain_new_bns2 <= 0:
                                #             if self.df_bns_check2[self.df_bns_check2 <= -1].count() >= 1:
                                #                 self.bns_check = 2.3
                                #                 if self.bns_check_mode != 1 or (
                                #                         self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                #                     if (self.which_market == 1 or (
                                #                             self.stat_in_org == "000" and self.stat_out_org == "111")):
                                #                         self.in_str = 2
                                #                         self.OrgMain = "b"
                                #                         self.nfset = self.nf
                                #                         self.inp = float(lblShoga1v)
                                #                         self.last_o = float(lblShoga1v)
                                #                         self.ave_prc = float(lblShoga1v)
                                #                         self.exed_qty += 1
                                #                         self.type = "conv 2"
                                #                         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                #                                       self.OrgMain, self.type, self.profitf, self.profit_opt,
                                #                                       self.mode, self.ave_prc, prc_o1)
                                #                         self.in_hit = 0
                                #                         self.in_hit_touch = 0
                                #                         self.init5_touched = 0

                                ###################################
                                # Conversion init 3 (Test_signal_new)
                                ###################################
                                if self.df_test_signal[self.df_test_signal >= 3].count() >= 1 and self.test_signal < 3:
                                    if self.dOrgMain_new_bns2 <= 0:
                                        self.bns_check = 2.6
                                        if self.bns_check_mode != 1 or (
                                                self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                            if (self.which_market == 1 or (
                                                    self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                self.in_str = 2
                                                self.OrgMain = "b"
                                                self.nfset = self.nf
                                                self.inp = float(lblShoga1v)
                                                self.last_o = float(lblShoga1v)
                                                self.ave_prc = float(lblShoga1v)
                                                self.exed_qty += 1
                                                self.type = "conv 3"
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                              self.OrgMain, self.type,
                                                              self.profit, self.profit_opt, self.mode, self.ave_prc,
                                                              prc_o1)
                                                self.in_hit = 0
                                                self.in_hit_touch = 0

                        ###################################
                        # SELL
                        ###################################

                        ###################################
                        # Trend init 1(long ai_long high + not prc_s_peak peak)
                        ###################################
                        if self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 2 and self.df_bns2[
                            self.df_bns2 > 0].count() >= 10 and self.df_rsi_peak[self.df_rsi_peak > 0].count() >= 15:

                            # Original TR
                            if (self.which_market != 4 and self.ai <= 0.1 and self.ai_long <= 1.8) or (
                                    self.which_market == 4 and self.ai_long <= 0.2):
                                if self.ai <= 0.1 and self.df.loc[self.nf - 50:self.nf - 1,
                                                      "ai"].mean() <= 0.1:# and self.ai_long <= self.ai_long_low:
                                    if self.prc_s_peak != -2 and self.df_std_std[
                                        self.df_std_std >= self.std_std_prc_cvol_m_limit * 0.8].count() == 0:
                                        if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.2:
                                            # if self.df.loc[self.nf - 100:self.nf - 1, "PXY_l_ave"].mean() <= 5:
                                            if self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.6:
                                                if self.df.loc[self.nf - 10:self.nf - 1,"rsi"].mean() < self.df.loc[self.nf - 50:self.nf - 10,
                                                                        "rsi"].mean(): #self.dOrgMain_new_bns2 < 0 and self.rsi_peak <= 0 and
                                                    self.bns_check = -1
                                                    if self.which_market == 3 or self.which_market != 3:  # and self.df.loc[self.nf - 50:self.nf - 1,"count_m_per_80"].mean() <= 5
                                                        if (self.which_market == 1 or (
                                                                self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                            if self.df.loc[self.nf - 50:self.nf - 1,
                                                               "d_OMain"].mean() == 0:
                                                                if self.df_bns_check[
                                                                    self.df_bns_check <= -1].count() >= 5:
                                                                    self.in_str = -2
                                                                    self.OrgMain = "s"
                                                                    self.nfset = self.nf
                                                                    self.inp = float(lblBhoga1v)
                                                                    self.last_o = float(lblBhoga1v)
                                                                    self.ave_prc = float(lblBhoga1v)
                                                                    self.exed_qty += 1
                                                                    self.type = "tr 1"
                                                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v,
                                                                                  self.exed_qty, self.OrgMain,
                                                                                  self.type,
                                                                                  self.profit, self.profit_opt,
                                                                                  self.mode, self.ave_prc, prc_o1)
                                                                    self.in_hit = 0
                                                                    self.in_hit_touch = 0
                                                                    self.init5_touched = 0

                                ###################################
                                # Trend init 2(PXY_l_ave)
                                ###################################
                                if self.nf >= 500 and self.df.loc[self.nf - 200:self.nf - 1, "PXY_l_ave"].mean() <= 20:
                                    if self.prc_s_peak != -2 and self.std_prc_cvol_m < 5:
                                        if self.std_std_prc_cvol_m_peak == 0 and self.df.loc[self.nf - 500:self.nf - 1,
                                                                                 "std_std_prc_cvol_m_peak"].mean() > 0:
                                            if self.df.loc[self.nf - 100:self.nf - 1,
                                               "std_prc_peak_1000"].mean() <= 0 and self.std_prc_peak_1000 <= 0:
                                                if self.std_prc_cvol_m <= self.df.loc[self.nf - 250:self.nf - 1,
                                                                          "std_prc_cvol_m"].min() * 0.8:
                                                    self.bns_check = -1.3
                                                    if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[
                                                        self.nf - 3, "OrgMain"] == 'n'):
                                                        if (self.which_market == 1 or (
                                                                self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                            if self.dOrgMain_new_bns2 < 0 and self.rsi_peak <= 0 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.8:
                                                                if self.df.loc[self.nf - 50:self.nf - 1,
                                                                   "d_OMain"].mean() == 0:
                                                                    if self.df_bns_check[
                                                                        self.df_bns_check <= -1].count() >= 5:
                                                                        self.in_str = -2
                                                                        self.OrgMain = "s"
                                                                        self.nfset = self.nf
                                                                        self.inp = float(lblBhoga1v)
                                                                        self.last_o = float(lblBhoga1v)
                                                                        self.ave_prc = float(lblBhoga1v)
                                                                        self.exed_qty += 1
                                                                        self.type = "tr 2"
                                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v,
                                                                                      self.exed_qty, self.OrgMain,
                                                                                      self.type,
                                                                                      self.profit, self.profit_opt,
                                                                                      self.mode, self.ave_prc, prc_o1)
                                                                        self.in_hit = 0
                                                                        self.in_hit_touch = 0
                                                                        self.init5_touched = 0
                                                # else:
                                                #     self.bns_check = 1.3
                                                #     if self.bns_check_mode != 1 or self.OrgMain == 'n':
                                                #         self.in_str = 2
                                                #         self.OrgMain = "b"
                                                #         self.nfset = self.nf
                                                #         self.inp = float(lblShoga1v)
                                                #         self.last_o = float(lblShoga1v)
                                                #         self.ave_prc = float(lblShoga1v)
                                                #         self.exed_qty += 1
                                                #         self.type = "tr 2 conv"
                                                #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                #                       self.type,
                                                #                       self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                                #         self.in_hit = 0
                                                #         self.in_hit_touch = 0
                                                #         self.init5_touched = 0
                                ###################################
                                # Trend init 3(dOrgMain_new_bns2)
                                ###################################
                                if self.df.loc[self.nf - 50:self.nf - 1,
                                   "dOrgMain_new_bns2"].mean() >= 1 and self.dOrgMain_new_bns2 <= 0 and self.std_prc_cvol_m < 5:
                                    if self.std_prc_1000 < 0 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.4:
                                        self.bns_check = -1.5
                                        if self.bns_check_mode != 1 or (
                                                self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                            if (self.which_market == 1 or (
                                                    self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                if self.rsi_peak <= 1 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.4:
                                                    if self.df.loc[self.nf - 50:self.nf - 1, "d_OMain"].mean() == 0:
                                                        self.in_str = -2
                                                        self.OrgMain = "s"
                                                        self.nfset = self.nf
                                                        self.inp = float(lblShoga1v)
                                                        self.last_o = float(lblShoga1v)
                                                        self.ave_prc = float(lblShoga1v)
                                                        self.exed_qty += 1
                                                        self.type = "tr 3"
                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                                      self.OrgMain,
                                                                      self.type,
                                                                      self.profit, self.profit_opt, self.mode,
                                                                      self.ave_prc, prc_o1)
                                                        self.in_hit = 0
                                                        self.in_hit_touch = 0

                                if self.df.loc[self.nf - 50:self.nf - 1,"dOrgMain_new_bns2"].mean() <= 0 and self.dOrgMain_new_bns2 <= 0:
                                    if self.std_prc_1000 < 0 and self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.8:
                                        if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 0.8].count() == 0:
                                            self.bns_check = 1.8
                                            if self.bns_check_mode != 1 or (
                                                    self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                                if (self.which_market == 1 or (
                                                        self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                    if self.rsi_peak <= -1 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.8:
                                                        if self.df.loc[self.nf - 50:self.nf - 1, "d_OMain"].mean() == 0:
                                                            self.in_str = 2
                                                            self.OrgMain = "b"
                                                            self.nfset = self.nf
                                                            self.inp = float(lblBhoga1v)
                                                            self.last_o = float(lblBhoga1v)
                                                            self.ave_prc = float(lblBhoga1v)
                                                            self.exed_qty += 1
                                                            self.type = "tr 3_conv"
                                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                                          self.OrgMain,
                                                                          self.type,
                                                                          self.profit, self.profit_opt, self.mode,
                                                                          self.ave_prc, prc_o1)
                                                            self.in_hit = 0
                                                            self.in_hit_touch = 0
                                                            self.init5_touched = 0


                        ###################################
                        # Conversion init 1(prc_s_limit)
                        ###################################
                        self.prc_per_90 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 90)
                        self.prc_per_80 = np.percentile(np.array(self.df['price'], dtype=float)[self.nf - 500:self.nf - 2], 80)
                        self.df.at[self.nf, "prc_per_90"] = self.prc_per_90

                        # self.gold2 = 0
                        # if price < self.prc_per_90:
                        #     self.gold2 = -1
                        # if price < self.prc_per_80:
                        #     self.gold2 = -2
                        # self.df.at[self.nf, "gold2"] = self.gold2
                        # self.gold2_avg = self.df.loc[self.nf - 10:self.nf - 1, "gold2"].mean()
                        # self.df.at[self.nf, "gold2_avg"] = self.gold2_avg

                        if self.nf > 800 and (
                                self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 1.5 and self.df_std_std[
                            self.df_std_std >= self.std_std_prc_cvol_m_limit * 1.25].count() > 1):
                            if (self.df.loc[self.nf - 30:self.nf - 10,
                                "std_prc_peak_1000"].mean() >= 1 and self.std_prc_peak == 0) or (self.nf > 2000 and rsi > 58):
                                if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.3:
                                    if self.std_std_prc_cvol_m < self.df.loc[self.nf - 100:self.nf - 1, "std_std_prc_cvol_m"].max()*0.3:
                                        self.bns_check = -3
                                        if self.bns_check_mode != 1 or (
                                                self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                            if (self.which_market == 1 or (
                                                    self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                self.in_str = -2
                                                self.OrgMain = "s"
                                                self.nfset = self.nf
                                                self.inp = float(lblBhoga1v)
                                                self.last_o = float(lblBhoga1v)
                                                self.ave_prc = float(lblBhoga1v)
                                                self.exed_qty += 1
                                                self.type = "conv 0"
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                              self.type,
                                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                                self.in_hit = 0
                                                self.in_hit_touch = 0
                                                self.init5_touched = 0

                            if (self.which_market != 4 and self.ai <= 0.1) or (
                                    self.which_market == 4 and self.ai_long <= 0.2):

                                if (self.df.loc[self.nf - 160:self.nf - 10,
                                    "ai_long"].mean() > 1.8):  # not like 'conv 1 buy', no 'which_market condition' because ai's characteristic
                                    if self.prc_s != 0 and self.prc_s < (self.prc_s_limit * 1) and self.df_prc_s[
                                        self.df_prc_s > self.prc_s_limit * 1].count() > 1:
                                        if self.df_std_std[
                                            self.df_std_std >= self.std_std_prc_cvol_m_limit * 1].count() > 1:  # and (self.ai <= 0.1): # or self.ai_long <= self.ai_long_low):
                                            # self.test_signal = -5
                                            if self.dOrgMain_new_bns2 >= 0:
                                                self.bns_check = -2
                                                if self.bns_check_mode != 1 or (self.OrgMain == 'n' and self.df.at[
                                                    self.nf - 3, "OrgMain"] == 'n'):
                                                    if (self.which_market == 1 or (
                                                            self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                        self.in_str = -2
                                                        self.OrgMain = "s"
                                                        self.nfset = self.nf
                                                        self.inp = float(lblBhoga1v)
                                                        self.last_o = float(lblBhoga1v)
                                                        self.ave_prc = float(lblBhoga1v)
                                                        self.exed_qty += 1
                                                        self.type = "conv 1"
                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                                      self.OrgMain,
                                                                      self.type,
                                                                      self.profit, self.profit_opt, self.mode,
                                                                      self.ave_prc, prc_o1)
                                                        self.in_hit = 0
                                                        self.in_hit_touch = 0
                                                        self.init5_touched = 0

                                ###################################
                                # Conversion init 2(long ai low -> prc_s_peak up)
                                ###################################
                                # if self.df.loc[self.nf - 250:self.nf - 25, "ai_long"].mean() >= 1.5 and self.ai <= 0.2:
                                #     if self.df.loc[self.nf - 100:self.nf - 25,
                                #        "prc_s_peak"].mean() > 0.8 and self.prc_s_peak == 0:
                                #         if self.dOrgMain_new_bns2 >= 0:
                                #             if self.df_bns_check2[self.df_bns_check2 >= 1].count() >= 1:
                                #                 self.bns_check = -2.3
                                #                 if self.bns_check_mode != 1 or (
                                #                         self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                #                     if (self.which_market == 1 or (
                                #                             self.stat_in_org == "000" and self.stat_out_org == "111")):
                                #                         self.in_str = -2
                                #                         self.OrgMain = "s"
                                #                         self.nfset = self.nf
                                #                         self.inp = float(lblBhoga1v)
                                #                         self.last_o = float(lblBhoga1v)
                                #                         self.ave_prc = float(lblBhoga1v)
                                #                         self.exed_qty += 1
                                #                         self.type = "conv 2"
                                #                         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                #                                       self.OrgMain,
                                #                                       self.type,
                                #                                       self.profit, self.profit_opt, self.mode, self.ave_prc,
                                #                                       prc_o1)
                                #                         self.in_hit = 0
                                #                         self.in_hit_touch = 0

                                ###################################
                                # Conversion init 3 (Test_signal_new)
                                ###################################
                                if self.df_test_signal[
                                    self.df_test_signal <= -3].count() >= 1 and self.test_signal > -3:
                                    if self.dOrgMain_new_bns2 >= 0:
                                        self.bns_check = -2.6
                                        if self.bns_check_mode != 1 or (
                                                self.OrgMain == 'n' and self.df.at[self.nf - 3, "OrgMain"] == 'n'):
                                            if (self.which_market == 1 or (
                                                    self.stat_in_org == "000" and self.stat_out_org == "111")):
                                                self.in_str = -2
                                                self.OrgMain = "s"
                                                self.nfset = self.nf
                                                self.inp = float(lblBhoga1v)
                                                self.last_o = float(lblBhoga1v)
                                                self.ave_prc = float(lblBhoga1v)
                                                self.exed_qty += 1
                                                self.type = "conv 3"
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                              self.OrgMain,
                                                              self.type,

                                                              self.profit, self.profit_opt, self.mode, self.ave_prc,
                                                              prc_o1)
                                                self.in_hit = 0
                                                self.in_hit_touch = 0

            if self.bns_check_last != self.bns_check and self.bns_check != 0:
                self.bns_check_last = self.bns_check
            if self.bns_check_4_last != self.bns_check_4 and self.bns_check_4 != 0:
                self.bns_check_4_last = self.bns_check_4
            if self.gold_last != self.check_gold and self.check_gold != 0:
                self.gold_last = self.check_gold
            if self.rsi_last != self.check_rsi_gold and self.check_rsi_gold != 0:
                self.rsi_last = self.check_rsi_gold
            if self.pvol_last != self.check_pvol_gold and self.check_pvol_gold != 0:
                self.pvol_last = self.check_pvol_gold
            if self.p1000_last != self.check_p1000_gold and self.check_p1000_gold != 0:
                self.p1000_last = self.check_p1000_gold

            self.triple_last = 0
            if self.gold_last > 0 and self.rsi_last > 0 and self.pvol_last > 0:
                self.triple_last = 1
            if self.gold_last < 0 and self.rsi_last < 0 and self.pvol_last < 0:
                self.triple_last = -1
            if self.triple_last_last != self.triple_last and self.triple_last != 0:
                self.triple_last_last = self.triple_last


            self.df.at[self.nf, "bns_check_last"] = self.bns_check_last
            self.df.at[self.nf, "bns_check_4_last"] = self.bns_check_4_last

            self.df.at[self.nf, "gold_last"] = self.gold_last
            self.df.at[self.nf, "rsi_last"] = self.rsi_last
            self.df.at[self.nf, "pvol_last"] = self.pvol_last
            self.df.at[self.nf, "p1000_last"] = self.p1000_last

            self.df.at[self.nf, "triple_last"] = self.triple_last
            self.df.at[self.nf, "triple_last_last"] = self.triple_last_last

            # triple_gray (down -> up : 1)
            # if self.OrgMain == "s":
            self.t_gray = 0
            if self.triple_last == 0 and self.triple_last_last == -1:
                self.t_gray = 0.5
                if self.prc_s_peak > -1: # only in t_gray = 1
                    if self.cover_signal_2 > self.df.at[self.nf - 2, "cover_signal_2"]:
                        self.t_gray = 1
                    if self.dOrgMain_new_bns2 > self.df.at[self.nf - 2, "dOrgMain_new_bns2"]:
                        self.t_gray = 1
                    if self.sum_peak > self.df.at[self.nf - 2, "sum_peak"]:
                        self.t_gray = 1
                    if self.df.loc[self.nf - 10: self.nf - 1, "std_prc_cvol_m"].mean() > 0:
                        self.t_gray = 1

            if self.triple_last == 0 and self.triple_last_last == 1:
                self.t_gray = -0.5
                if self.cover_signal_2 < self.df.at[self.nf - 2, "cover_signal_2"]:
                    self.t_gray = -1
                if self.dOrgMain_new_bns2 > self.df.at[self.nf - 2, "dOrgMain_new_bns2"]:
                    self.t_gray = -1
                if self.which_market != 1 and self.sum_peak > self.df.at[self.nf - 2, "sum_peak"]:
                    self.t_gray = -1
                if self.df.loc[self.nf - 10: self.nf - 1, "std_prc_cvol_m"].mean() > 0:
                    self.t_gray = -1

            # triple_gray_strong (strogn up : 1, strong down :-1)
            self.t_gray_strong = 0
            if self.test_signal <= -2 and self.sum_peak >= 2 and self.dOrgMain_new_bns2 >= 1:
                self.t_gray_strong = 1
            if self.sum_peak >= 2 and self.prc_s_peak >= 2 and self.dOrgMain_new_bns2 >= 1:
                self.t_gray_strong = 2
            if self.test_signal >= 2 and self.sum_peak <= -2 and self.dOrgMain_new_bns2 <= -1:
                self.t_gray_strong = -1
            if self.sum_peak <= -2 and self.prc_s_peak <= -2 and self.dOrgMain_new_bns2 <= -1:
                self.t_gray_strong = -2

            # if self.which_market == 1:
            #     if self.dOrgMain_new_bns2 == 2:
            #         self.t_gray_strong = 2
            #     if self.dOrgMain_new_bns2 <= -1:
            #         self.t_gray_strong = -2

            self.df.at[self.nf, "gray"] = self.t_gray
            self.df.at[self.nf, "gray_strong"] = self.t_gray_strong

            self.df.at[self.nf, "last_cover_signal2"] = self.last_cover_signal2
            self.df.at[self.nf, "now_cover_signal2"] = self.now_cover_signal2
            self.df.at[self.nf, "last_prc_s_peak"] = self.last_prc_s_peak
            self.df.at[self.nf, "now_prc_s_peak"] = self.now_prc_s_peak
            self.df.at[self.nf, "long_prc_s_peak"] = self.long_prc_s_peak

            # NEW SIGNAL_IN
            if 1 == 1:
                if self.nf >= 501:

                    self.dOrgMain_new_s1 = 0
                    self.dOrgMain_new_s2 = 0
                    self.dOrgMain_new_b1 = 0
                    self.dOrgMain_new_b2 = 0
                    self.dOrgMain_new_b3 = 0
                    self.dOrgMain_new_b4 = 0
                    self.df_std_prc_cvol_m = self.df.loc[self.nf - 500: self.nf - 1, "std_prc_cvol_m"]

                    # if self.which_market != 3 or (self.which_market == 3 and now.hour < 13):
                    #     if self.OrgMain_new == 'n' and self.df.at[self.nf - 3, "OrgMain_new"] == "n":
                    if self.which_market == 3 or self.which_market == 1:
                        if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() <= 0.8:  # and self.cover_signal_2 == -1:
                            self.dOrgMain_new_s1 = -0.5
                            if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() <= 0:
                                # if self.df.at[self.nf - 2, "prc_s"] <= self.prc_s :
                                # self.OrgMain_new = "s"
                                self.dOrgMain_new_s1 = -1
                                # self.nfset_new = self.nf
                                # self.inp_new = float(lblBhoga1v)
                                # self.last_o_new = float(lblBhoga1v)
                                # self.ave_prc_new = float(lblBhoga1v)
                                # self.exed_qty_new += 1
                                # self.type_new = "in_S1"
                    if self.which_market == 4:
                        if self.df.loc[self.nf - 20:self.nf - 1, "ai_long"].mean() <= self.ai_long_low:
                            self.dOrgMain_new_s1 = -0.5
                            if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() <= 0:
                                # if self.df.at[self.nf - 2, "prc_s"] <= self.prc_s :
                                # self.OrgMain_new = "s"
                                self.dOrgMain_new_s1 = -1
                                # self.nfset_new = self.nf
                                # self.inp_new = float(lblBhoga1v)
                                # self.last_o_new = float(lblBhoga1v)
                                # self.ave_prc_new = float(lblBhoga1v)
                                # self.exed_qty_new += 1
                                # self.type_new = "in_S1"

                    if self.df_std_prc_cvol_m[self.df_std_prc_cvol_m >= self.std_prc_cvol_m_limit].count() >= 1:
                        if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() <= 2:
                            # self.OrgMain_new = "s"
                            self.dOrgMain_new_s2 = -1
                            # self.nfset_new = self.nf
                            # self.inp_new = float(lblBhoga1v)
                            # self.last_o_new = float(lblBhoga1v)
                            # self.ave_prc_new = float(lblBhoga1v)
                            # self.exed_qty_new += 1
                            # self.type_new = "in_S2"

                    # NEW SIGNAL_OUT
                    # if self.OrgMain_new == 's' and self.exed_qty_new != 0:

                    # if (price - self.ave_prc_new) / prc_std >= 0.5:

                    if self.df_std_prc_cvol_m[self.df_std_prc_cvol_m <= self.std_prc_cvol_m_limit * -1].count() >= 1:
                        if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() >= 0:
                            # self.Profit_new += ((float(lblBhoga1v) - self.inp_new))
                            # self.OrgMain_new = 'n'
                            self.dOrgMain_new_b1 = 1
                            # self.type_new = "out_S_spot1"
                            # self.last_o_new = 0
                            # self.inp_new = 0
                            # self.nfset_new = 0
                            # self.exed_qty_new = 0
                            # self.ave_prc_new = 0

                    # self.df_test_signal = self.df.loc[self.nf - 200: self.nf - 1, "test_signal"]
                    # self.df_prc_s = self.df.loc[self.nf - 100: self.nf - 1, "prc_s"]
                    if self.prc_s > (self.prc_s_limit * -1) and self.df_prc_s[
                        self.df_prc_s < self.prc_s_limit * -1].count() > 1:
                        if self.df_test_signal[self.df_test_signal >= 3].count() >= 1 and self.test_signal < 3:
                            # self.Profit_new += ((float(lblBhoga1v) - self.inp_new))
                            # self.OrgMain_new = 'n'
                            self.dOrgMain_new_b2 = 1
                            # self.type_new = "out_S_spot2"
                            # self.last_o_new = 0
                            # self.inp_new = 0
                            # self.nfset_new = 0
                            # self.exed_qty_new = 0
                            # self.ave_prc_new = 0

                    if self.which_market == 3:
                        if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() > -1:
                            self.dOrgMain_new_b3 = 1
                    if self.which_market != 3:
                        if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() > 1:
                            # self.Profit_new += ((float(lblBhoga1v) - self.inp_new))
                            # self.OrgMain_new = 'n'
                            self.dOrgMain_new_b3 = 1
                            # self.type_new = "out_S_late1"
                            # self.last_o_new = 0
                            # self.inp_new = 0
                            # self.nfset_new = 0
                            # self.exed_qty_new = 0
                            # self.ave_prc_new = 0

                    if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() > 0.1:  # and self.ai_2 == 1:
                        if self.which_market == 3:
                            if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() > -1:
                                # self.Profit_new += ((float(lblBhoga1v) - self.inp_new))
                                # self.OrgMain_new = 'n'
                                self.dOrgMain_new_b4 = 1
                                # self.type_new = "out_S_ late2"
                                # self.last_o_new = 0
                                # self.inp_new = 0
                                # self.nfset_new = 0
                                # self.exed_qty_new = 0
                                # self.ave_prc_new = 0
                        if self.which_market != 3:
                            if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() > 1:
                                self.dOrgMain_new_b4 = 1

                    #####################
                    # <1> new_bns(s2 and b1) : std_prc_cvol_m_limit.count
                    # after-peak trend
                    #####################
                    self.dOrgMain_new_bns = 0
                    if self.dOrgMain_new_s2 == -1 and self.dOrgMain_new_b1 == 0 and self.dOrgMain_new_b4 == 0:
                        if self.df.loc[self.nf - 20:self.nf - 1, "px1"].mean() < self.df.loc[self.nf - 40:self.nf - 20,
                                                                                 "px1"].mean():
                            if self.std_prc_slope < 0:
                                self.dOrgMain_new_bns = -1
                    if self.dOrgMain_new_s2 == 0 and self.dOrgMain_new_b1 == 1 and self.dOrgMain_new_b4 == 1:
                        if self.df.loc[self.nf - 20:self.nf - 1, "px1"].mean() > self.df.loc[self.nf - 40:self.nf - 20,
                                                                                 "px1"].mean():
                            if self.std_prc_slope > 0:
                                self.dOrgMain_new_bns = 1

                    # bns peak
                    self.df_new_bns = self.df.loc[self.nf - 100: self.nf - 1, "std_prc_cvol_m"]
                    if self.df_new_bns[self.df_new_bns <= -8].count() >= 1:
                        if self.std_prc_cvol_m > -5:
                            self.dOrgMain_new_bns = -2
                    if self.df_new_bns[self.df_new_bns >= 8].count() >= 1:
                        if self.std_prc_cvol_m < 5:
                            self.dOrgMain_new_bns = 2

                    #####################
                    # <2> new_bns(s1 and b3) : std_prc_cvol_m.mean() + std_std_prc_cvol_m
                    # simple trend
                    #####################
                    self.dOrgMain_new_bns2 = 0
                    if self.df.loc[self.nf - 50:self.nf - 1, "dOrgMain_new_s1"].mean() < -0.5:
                        if self.df.loc[self.nf - 50:self.nf - 1, "dOrgMain_new_b3"].mean() < 0.5:
                            self.dOrgMain_new_bns2 = -1
                            if self.std_prc_slope < 0:
                                self.dOrgMain_new_bns2 = -2
                    # if self.df.loc[self.nf - 50:self.nf - 1, "dOrgMain_new_s1"].mean() > -0.5:
                    if self.df.loc[self.nf - 50:self.nf - 1,
                       "dOrgMain_new_b4"].mean() > 0.5:  # or self.df.loc[self.nf - 50:self.nf - 1, "dOrgMain_new_b3"].mean() > 0.5:
                        if self.df.loc[self.nf - 50:self.nf - 1, "dOrgMain_new_s1"].mean() > -0.5:
                            self.dOrgMain_new_bns2 = 1
                            if self.std_prc_slope > 0:
                                self.dOrgMain_new_bns2 = 2

                    # bns2 peak
                    # self.df_new_bns2 = self.df.loc[self.nf - 100: self.nf - 1, "std_std_prc_cvol_m"]
                    # if self.df_new_bns2[self.df_new_bns2 >= self.std_std_prc_cvol_m_limit].count() >= 1:
                    if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit:
                        # if self.std_std_prc_cvol_m < 1:
                        if self.dOrgMain_new_bns2 <= -1:
                            self.dOrgMain_new_bns2 = -3
                        if self.dOrgMain_new_bns2 >= 1:
                            self.dOrgMain_new_bns2 = 3

                    #####################
                    # new_bns_order(in)
                    #####################
                    if self.dOrgMain_new_bns_order == 0 and self.nf >= self.last_o_new + 50:

                        if self.dOrgMain_new_s1 == -1 and self.dOrgMain_new_b4 == 0:  # and self.ai >= 0.8 and self.cover_signal_2 != -1:
                            self.dOrgMain_new_bns_order = -1
                            self.OrgMain_new = "s"
                            # self.dOrgMain_new_s1 = -1
                            self.nfset_new = self.nf
                            self.inp_new = float(lblBhoga1v)
                            self.last_o_new = float(lblBhoga1v)
                            self.ave_prc_new = float(lblBhoga1v)
                            self.exed_qty_new += 1
                            self.type_new = "in_order_s"

                        if self.dOrgMain_new_s1 == 0 and self.dOrgMain_new_b4 == 1:  # and self.ai >= 0.8 and self.cover_signal_2 != -1:
                            self.dOrgMain_new_bns_order = 1
                            self.OrgMain_new = "b"
                            # self.dOrgMain_new_s1 = -1
                            self.nfset_new = self.nf
                            self.inp_new = float(lblBhoga1v)
                            self.last_o_new = float(lblBhoga1v)
                            self.ave_prc_new = float(lblBhoga1v)
                            self.exed_qty_new += 1
                            self.type_new = "in_order_b"

                        # if self.df.loc[self.nf - 100:self.nf - 1, "dOrgMain_new_b1"].mean() == 0:
                        #     if self.df.loc[self.nf - 100:self.nf - 1, "dOrgMain_new_b3"].mean() == 1:
                        #         if self.df.loc[self.nf - 100:self.nf - 1, "dOrgMain_new_b4"].mean() == 1:
                        #             self.dOrgMain_new_bns_order = 1
                        #             self.OrgMain_new = "b"
                        #             # self.dOrgMain_new_s1 = -1
                        #             self.nfset_new = self.nf
                        #             self.inp_new = float(lblBhoga1v)
                        #             self.last_o_new = float(lblBhoga1v)
                        #             self.ave_prc_new = float(lblBhoga1v)
                        #             self.exed_qty_new += 1
                        #             self.type_new = "in_order_b2"

                        if self.dOrgMain_new_bns <= -1:  # and self.ai == 0 and self.ai_2 != 1:
                            self.dOrgMain_new_bns_order = -1
                            self.OrgMain_new = "s"
                            # self.dOrgMain_new_s1 = -1
                            self.nfset_new = self.nf
                            self.inp_new = float(lblBhoga1v)
                            self.last_o_new = float(lblBhoga1v)
                            self.ave_prc_new = float(lblBhoga1v)
                            self.exed_qty_new += 1
                            self.type_new = "in_order_s"

                    # new_bns_order(out)
                    if self.dOrgMain_new_bns_order == -1:

                        if (
                                self.dOrgMain_new_s1 == 0 and self.dOrgMain_new_b4 == 1) or self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit:
                            self.dOrgMain_new_bns_order = 0
                            self.Profit_new += (self.inp_new - (float(lblBhoga1v)))
                            self.OrgMain_new = 'n'
                            # self.dOrgMain_new_b4 = 1
                            self.type_new = "out_sb"
                            self.last_o_new = 0
                            self.inp_new = 0
                            self.nfset_new = 0
                            self.exed_qty_new = 0
                            self.ave_prc_new = 0

                    if self.dOrgMain_new_bns_order == 1:

                        if (
                                self.dOrgMain_new_s1 == -1 and self.dOrgMain_new_b4 == 0) or self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit:
                            self.dOrgMain_new_bns_order = 0
                            self.Profit_new += ((float(lblBhoga1v)) - self.inp_new)
                            self.OrgMain_new = 'n'
                            # self.dOrgMain_new_b4 = 1
                            self.type_new = "out_bs"
                            self.last_o_new = 0
                            self.inp_new = 0
                            self.nfset_new = 0
                            self.exed_qty_new = 0
                            self.ave_prc_new = 0

                    # if self.dOrgMain_new_bns_order == -1:
                    #     if self.df.loc[self.nf - 30:self.nf - 1, "dOrgMain_new_b3"].mean() > 0.5:
                    #         if self.df_std_prc_cvol_m[self.df_std_prc_cvol_m >= 8].count() == 0:
                    #             if (self.ai != 0 and self.dOrgMain_new_bns == 1) or self.std_prc_slope_200 > 0:
                    #                 self.dOrgMain_new_bns_order = 0
                    #                 self.Profit_new += (self.inp_new - float(lblBhoga1v))
                    #                 self.OrgMain_new = 'n'
                    #                 # self.dOrgMain_new_b4 = 1
                    #                 self.type_new = "out_sb"
                    #                 self.last_o_new = 0
                    #                 self.inp_new = 0
                    #                 self.nfset_new = 0
                    #                 self.exed_qty_new = 0
                    #                 self.ave_prc_new = 0
                    #         if self.df_std_prc_cvol_m[self.df_std_prc_cvol_m >= 8].count() >= 1:
                    #             if self.df.loc[self.nf - 10:self.nf - 1, "std_prc_cvol_m"].mean() >= 0 or self.dOrgMain_new_bns == 1:
                    #                 self.dOrgMain_new_bns_order = 0
                    #                 self.Profit_new += (self.inp_new - float(lblBhoga1v))
                    #                 self.OrgMain_new = 'n'
                    #                 # self.dOrgMain_new_b4 = 1
                    #                 self.type_new = "out_sb"
                    #                 self.last_o_new = 0
                    #                 self.inp_new = 0
                    #                 self.nfset_new = 0
                    #                 self.exed_qty_new = 0
                    #                 self.ave_prc_new = 0

                    if self.new_bns_mode == 0:
                        self.new_bns_mode = self.df.at[self.nf - 1, "dOrgMain_new_bns_order"]
                    if self.new_bns_mode != 0:
                        self.new_bns_mode = self.dOrgMain_new_bns_order

                self.df.at[self.nf, "OrgMain_new"] = self.OrgMain_new
                self.df.at[self.nf, "dOrgMain_new_s1"] = self.dOrgMain_new_s1
                self.df.at[self.nf, "dOrgMain_new_s2"] = self.dOrgMain_new_s2
                self.df.at[self.nf, "dOrgMain_new_b1"] = self.dOrgMain_new_b1
                self.df.at[self.nf, "dOrgMain_new_b2"] = self.dOrgMain_new_b2
                self.df.at[self.nf, "dOrgMain_new_b3"] = self.dOrgMain_new_b3
                self.df.at[self.nf, "dOrgMain_new_b4"] = self.dOrgMain_new_b4
                self.df.at[self.nf, "dOrgMain_new_bns"] = self.dOrgMain_new_bns
                self.df.at[self.nf, "dOrgMain_new_bns2"] = self.dOrgMain_new_bns2
                self.df.at[self.nf, "dOrgMain_new_bns_order"] = self.dOrgMain_new_bns_order
                self.df.at[self.nf, "new_bns_mode"] = self.new_bns_mode
                self.df.at[self.nf, "nfset_new"] = self.nfset_new
                self.df.at[self.nf, "inp_new"] = self.inp_new
                self.df.at[self.nf, "last_o_new"] = self.last_o_new
                self.df.at[self.nf, "ave_prc_new"] = self.ave_prc_new
                self.df.at[self.nf, "exed_qty_new"] = self.exed_qty_new
                self.df.at[self.nf, "Profit_new"] = self.Profit_new
                self.df.at[self.nf, "type_new"] = self.type_new
                self.df.at[self.nf, "bns_check"] = self.bns_check
                self.df.at[self.nf, "bns_check_2"] = self.bns_check_2
                self.df.at[self.nf, "bns_check_3"] = self.bns_check_3
                self.df.at[self.nf, "bns_check_2_lock"] = self.bns_check_2_lock
                self.df.at[self.nf, "bns_check_2_last"] = self.bns_check_2_last
                self.df.at[self.nf, "lock_price"] = self.lock_price

                #####################
                # ADD_ORDER
                #####################

                m_add = 1.5
                if self.which_market == 1:
                    m_add = 20
                if self.which_market == 4:
                    m_add = 6
                if self.nf > self.nfset + 20:
                    # if self.df.at[self.nf - 1, "std_prc_peak"] == -2 and self.df.at[self.nf, "std_prc_peak"] == -1:
                    #     # if self.OrgMain == 'b':
                    #         # if self.cvol_t_decay > self.cvol_t_cri_ave * -20:
                    #     if ema_25_count_m == -1 and self.df.loc[self.nf - 50:self.nf - 1, "std_prc_peak"].mean() == -2:
                    #         if self.which_market != 3 or now.hour<15:
                    #             if 1==1 and price <= self.ave_prc - self.tick * 3:
                    #                 if self.mode_spot != -2:
                    #                     self.add_touch = 1
                    #                     self.type = "init 6+"
                    #
                    #                         # self.ave_prc = (self.ave_prc * self.exed_qty + float(lblShoga1v)) / (self.exed_qty + 1)
                    #                         # self.exed_qty += 1
                    #                         #
                    #                         # self.add_signal = 1
                    #                         # self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                    #                         #               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                    #
                    # if self.df.at[self.nf - 1, "std_prc_peak"] == 2 and self.df.at[self.nf, "std_prc_peak"] == 1:
                    #     # if self.OrgMain == 's':
                    #         # if self.cvol_t_decay < self.cvol_t_cri_ave * 20:
                    #     if ema_25_count_m == -1 and self.df.loc[self.nf - 50:self.nf - 1, "std_prc_peak"].mean() == 2:
                    #         if self.which_market != 3 or now.hour<15:
                    #             if 1==1 and price >= self.ave_prc + self.tick * 3:
                    #                 if self.mode_spot != 2:
                    #                     self.add_touch = -1
                    #                     self.type = "init 6+"
                    #
                    #                         # self.ave_prc = (self.ave_prc * self.exed_qty + float(lblBhoga1v)) / (self.exed_qty + 1)
                    #                         # self.exed_qty += 1
                    #                         #
                    #                         # self.add_signal = -1
                    #                         # self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                    #                         #               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # add_1_b
                    if 1 == 0:
                        if self.df.at[self.nf - 1, "std_prc_peak"] == -1 and self.df.at[self.nf, "std_prc_peak"] == 0:
                            if self.df.loc[self.nf - 150:self.nf - 1, "sum_peak"].mean() <= -2:
                                if self.which_market != 3 or now.hour < 15:
                                    if price <= self.last_o - prc_std * 1 * m_add and self.exed_qty <= 4:  # self.tick * 3:
                                        if self.cvol_m >= self.cvol_m_start * 0.3 and count_m >= self.count_m_start * 0.3:
                                            # if self.mode_spot != -2:
                                            self.add_touch = 1
                                            self.type = "add 1+"
                                            if self.OrgMain == "s" and self.exed_qty >= 2:
                                                self.cover_by_opt = 1
                                        # self.last_o = float(lblShoga1v)
                                    # else:
                                    #     self.add_touch = 0

                    # add_2_b
                    if self.df.at[self.nf - 2, "test_signal"] != 3 and self.df.at[self.nf - 1, "test_signal"] == 3:
                        # if self.df.loc[self.nf - 10:self.nf - 1, "test_signal"].std() >= 1.2:
                        self.add_touch = 1
                        if self.ave_prc != 0 and self.ave_prc - price >= prc_std * 2 * max((self.exed_qty - 1), 1):
                            if self.which_market != 3 or now.hour < 15:
                                if price <= self.last_o - prc_std * 1 * m_add and (
                                        self.which_market != 3 or self.exed_qty <= self.max_e_qty):  # self.tick * 3:
                                    # if self.cvol_m >= self.cvol_m_start * 0.3 and count_m >= self.count_m_start * 0.3:
                                    # if self.mode_spot != -2:
                                    self.add_touch = 2
                                    self.type = "add 2+"
                                    if self.OrgMain == "s" and self.exed_qty >= 1:
                                        self.cover_by_opt = 1
                                    self.last_o = float(lblShoga1v)
                            # else:
                            #     self.add_touch = 0

                    # add_3_b
                    if self.df.at[self.nf - 2, "bns_check"] >= 2 and self.df.at[self.nf - 1, "bns_check"] < 2:
                        self.add_touch = 1.5
                        if self.ave_prc != 0 and self.ave_prc - price >= prc_std * 2 * max((self.exed_qty - 1), 1):
                            if self.which_market != 3 or now.hour < 15:
                                if price <= self.last_o - prc_std * 1 * m_add and (
                                        self.which_market != 3 or self.exed_qty <= self.max_e_qty):  # self.tick * 3:
                                    # if self.cvol_m >= self.cvol_m_start * 0.3 and count_m >= self.count_m_start * 0.3:
                                    # if self.mode_spot != -2:
                                    self.add_touch = 2
                                    self.type = "add 3+"
                                    if self.OrgMain == "s" and self.exed_qty >= 1:
                                        self.cover_by_opt = 1
                                    self.last_o = float(lblShoga1v)

                        if 1 == 0:
                            if self.ave_prc - price >= prc_std * 8 and self.exed_qty <= 4 and price <= self.last_o - prc_std * 1 * m_add:
                                self.add_touch = 1
                                self.type = "add 3+"

                    if 1 == 0:
                        if self.ave_prc != 0 and self.df.at[self.nf - 2, "sum_peak"] < 0 and self.df.at[
                            self.nf - 1, "sum_peak"] == 0:
                            if self.OrgMain == "s" and self.exed_qty >= 2:
                                self.cover_by_opt = 1

                    # add_8_b
                    if 1 == 0:
                        if self.OrgMain == 'b':
                            if self.df.at[self.nf - 1, "test_signal"] != 3 and self.df.at[self.nf, "test_signal"] == 3:
                                if self.which_market != 3 or now.hour < 15:
                                    if price <= self.ave_prc - self.tick * 5:
                                        self.add_touch = 1
                                        self.type = "init 8+"
                                        self.last_o = float(lblShoga1v)
                                    else:
                                        self.add_touch = 0

                    # add_9_b
                    if 1 == 0:
                        if self.in_hit == 2:
                            # if self.OrgMain == 's':
                            self.add_touch = 1
                            self.type = "init 9+"
                            # self.last_o = float(lblShoga1v)
                        if self.df.at[self.nf - 1, "in_hit"] == 2 and self.df.at[self.nf, "in_hit"] != 2:
                            self.add_touch = 0

                    # add_10_b
                    if 1 == 0:
                        if self.df.at[self.nf - 1, "ema_520"] <= -1 and self.df.at[self.nf, "ema_520"] >= 1:
                            if price > self.last_o + self.tick * 3:
                                self.add_touch = 1
                                self.type = "init 10+"
                        if self.add_touch == 1:
                            if self.df.at[self.nf - 1, "ema_520"] >= 1 and self.df.at[self.nf, "ema_520"] <= -1:
                                self.add_touch = 0

                    # add_1_s
                    if 1 == 0:
                        if self.df.at[self.nf - 1, "std_prc_peak"] == 1 and self.df.at[self.nf, "std_prc_peak"] == 0:
                            if self.df.loc[self.nf - 150:self.nf - 1, "sum_peak"].mean() >= 2:
                                if self.which_market != 3 or now.hour < 15:
                                    if price >= self.last_o + prc_std * 1 * m_add and self.exed_qty <= 4:  # self.tick * 3:
                                        if self.cvol_m >= self.cvol_m_start * 0.3 and count_m >= self.count_m_start * 0.3:
                                            # if self.mode_spot != 2:
                                            self.add_touch = -1
                                            self.type = "add 1-"
                                            if self.OrgMain == "b" and self.exed_qty >= 2:
                                                self.cover_by_opt = -1
                                            # self.last_o = float(lblBhoga1v)
                                    # else:
                                    #     self.add_touch = 0

                    # add_2_s
                    if self.df.at[self.nf - 2, "test_signal"] != -3 and self.df.at[self.nf - 1, "test_signal"] == -3:
                        # if self.df.loc[self.nf - 10:self.nf - 1, "test_signal"].std() >= 1.2:
                        self.add_touch = -1
                        if self.ave_prc != 0 and price - self.ave_prc >= prc_std * 2 * max((self.exed_qty - 1), 1):
                            if self.which_market != 3 or now.hour < 15:
                                if price >= self.last_o + prc_std * 1 * m_add and (
                                        self.which_market != 3 or self.exed_qty <= self.max_e_qty):  # self.tick * 3:
                                    # if self.cvol_m >= self.cvol_m_start * 0.3 and count_m >= self.count_m_start * 0.3:
                                    # if self.mode_spot != -2:
                                    self.add_touch = -2
                                    self.type = "add 2-"
                                    if self.OrgMain == "b" and self.exed_qty >= 1:
                                        self.cover_by_opt = -1
                                    self.last_o = float(lblBhoga1v)
                                # else:
                                #     self.add_touch = 0

                    # add_3_s
                    if self.df.at[self.nf - 2, "bns_check"] <= -2 and self.df.at[self.nf - 1, "bns_check"] > -2:
                        self.add_touch = -1.5
                        if self.ave_prc != 0 and price - self.ave_prc >= prc_std * 2 * max((self.exed_qty - 1), 1):
                            if self.which_market != 3 or now.hour < 15:
                                if price >= self.last_o + prc_std * 1 * m_add and (
                                        self.which_market != 3 or self.exed_qty <= self.max_e_qty):  # self.tick * 3:
                                    # if self.cvol_m >= self.cvol_m_start * 0.3 and count_m >= self.count_m_start * 0.3:
                                    # if self.mode_spot != -2:
                                    self.add_touch = -2
                                    self.type = "add 3-"
                                    if self.OrgMain == "b" and self.exed_qty >= 1:
                                        self.cover_by_opt = -1
                                    self.last_o = float(lblBhoga1v)

                    # add_2_s_short
                    # if self.df.at[self.nf - 2, "test_signal"] != -3 and self.df.at[self.nf - 1, "test_signal"] == -3:
                    if self.df.loc[self.nf - 150:self.nf - 1, "test_signal"].mean() <= -1:
                        if self.ave_prc != 0 and price - self.ave_prc >= prc_std * 2 * max((self.exed_qty - 1), 1):
                            if self.df.loc[self.nf - 50:self.nf - 1, "ai"].mean() <= 0.05:
                                if self.which_market != 3 or now.hour < 15:
                                    if (self.which_market != 3 or self.exed_qty <= 4):
                                        self.add_touch = -2
                                        self.type = "add 2-short"
                                        if self.OrgMain == "b" and self.exed_qty >= 1:
                                            self.cover_by_opt = -1
                                        self.last_o = float(lblBhoga1v)
                                    # else:
                                    #     self.add_touch = 0

                        if 1 == 0:
                            if price - self.ave_prc >= prc_std * 8 and self.exed_qty <= 4 and price >= self.last_o + prc_std * 1 * m_add:
                                self.add_touch = -1
                                self.type = "add 3-"

                    if 1 == 0:
                        if self.ave_prc != 0 and self.df.at[self.nf - 2, "sum_peak"] > 0 and self.df.at[
                            self.nf - 1, "sum_peak"] == 0:
                            if self.OrgMain == "b" and self.exed_qty >= 2:
                                self.cover_by_opt = -1

                    # add_8_s
                    if 1 == 0:
                        if self.OrgMain == 's':
                            if self.df.at[self.nf - 1, "test_signal"] != -3 and self.df.at[
                                self.nf, "test_signal"] == -3:
                                if self.which_market != 3 or now.hour < 15:
                                    if price >= self.ave_prc + self.tick * 5:
                                        self.add_touch = -1
                                        self.type = "init 8+"
                                        self.last_o = float(lblBhoga1v)
                                    else:
                                        self.add_touch = 0

                    # add_9_s
                    if 1 == 0:
                        if self.in_hit == -2:
                            # if self.OrgMain == 'b':
                            self.add_touch = -1
                            self.type = "init 9+"
                            # self.last_o = float(lblShoga1v)
                        if self.df.at[self.nf - 1, "in_hit"] == -2 and self.df.at[self.nf, "in_hit"] != -2:
                            self.add_touch = 0

                    # add_10_s
                    if 1 == 0:
                        if self.df.at[self.nf - 1, "ema_520"] >= 1 and self.df.at[self.nf, "ema_520"] <= -1:
                            if price < self.last_o - self.tick * 3:
                                self.add_touch = -1
                                self.type = "init 10+"
                        if self.add_touch == -1:
                            if self.df.at[self.nf - 1, "ema_520"] <= -1 and self.df.at[self.nf, "ema_520"] >= 1:
                                self.add_touch = 0

                self.df.at[self.nf, "add_touch"] = self.add_touch
                self.df.at[self.nf, "Indep"] = self.Indep

                # send add_signal from add_touch
                self.add_signal = 0
                self.df_prc_s = self.df.loc[self.nf - 100: self.nf - 1, "prc_s"]
                if self.which_market == 3:
                    self.std_diff = (
                                                prc_std ** 1.1) * 3  # (abs(prc_avg_1000 - prc_avg) / 0.2) ** 0.9   ave(prc_std) = 0.3
                if self.which_market == 4:
                    self.std_diff = (
                                                prc_std ** 0.5) / 2  # (abs(prc_avg_1000 - prc_avg) / 3.5) ** 0.9   ave(prc_std) = 4
                if self.which_market == 1:
                    self.std_diff = (prc_std ** 0.5) / 3  # ave(prc_std) = 30
                self.df.at[self.nf, "std_diff"] = self.std_diff
                if 1 == 1 and abs(self.add_touch) >= 1 and self.std_std_prc_cvol_m_peak >= 1:
                    if self.add_touch > 0 and (self.OrgMain == 'b' or (self.Indep == 1 and self.OrgMain == 's')):
                        if (self.OrgMain == 'b' and price <= self.last_o - self.tick * 10 * m_add * max(self.std_diff,
                                                                                                        1)) or self.OrgMain == 's':
                            if self.ai >= 0.1:  # or self.ai_long >= self.ai_long_low: # and (self.df.loc[self.nf - 200:self.nf - 1, "add_touch"] != 2).any() == 1:
                                if self.OrgMain == 'b' and self.exed_qty <= self.max_e_qty:
                                    # if (self.now_trend - self.last_trend) > 0 and self.price_trend == 1:
                                    # if (self.df.loc[self.nf - 100:self.nf - 1, "prc_s_peak"].mean()<0 and self.prc_s_peak > -2) or (price <= self.last_o - self.tick * 10 * m_add):
                                    if (
                                            self.force_on == 1 and self.force == 1) or self.force_on == 0:  # and self.cvol_c_ave >= 10:
                                        self.exed_qty += 1
                                        self.add_signal = 1
                                        self.type = "plus b1"
                                        self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(lblShoga1v)) / (
                                            self.exed_qty)
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                      self.type,
                                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                        self.add_touch = 0
                                        # self.prf_hit = 0
                                        self.last_o = float(lblShoga1v)

                                    if self.exed_qty <= 3:
                                        if self.df.at[self.nf - 1, "add_touch"] == 2 and self.df.at[
                                            self.nf - 1, "add_touch"] != 2:
                                            self.exed_qty += 1
                                            self.add_signal = 1
                                            self.type = "plus b2"
                                            self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(lblShoga1v)) / (
                                                self.exed_qty)
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.add_touch = 0
                                            # self.prf_hit = 0
                                            self.last_o = float(lblShoga1v)

                                if 1 == 1 and self.OrgMain == 's' and self.exed_qty > 1 and price <= self.last_o - self.tick * 15 * m_add:
                                    if 1 == 0 and ((
                                                           self.prf / self.exed_qty > 0.5 * prc_std and self.exed_qty == 1) or self.exed_qty >= 2):
                                        if (self.now_trend - self.last_trend) > 0 or self.price_trend == 1:
                                            self.exed_qty -= 1
                                            self.add_signal = 1
                                            self.type = "minus s_1"
                                            # self.ave_prc = (self.ave_prc * (self.exed_qty + 1) - float(lblShoga1v)) / (self.exed_qty)
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.add_touch = 0
                                            # self.prf_hit = 0
                                            self.last_o = float(lblShoga1v)

                                    self.df_prc_s = self.df.loc[self.nf - 100: self.nf - 1, "prc_s"]
                                    if self.prc_s > (self.prc_s_limit * -1) and self.df_prc_s[
                                        self.df_prc_s < self.prc_s_limit * -1].count() > 1:
                                        self.exed_qty -= 1
                                        self.add_signal = 1
                                        self.type = "minus s_2"
                                        # self.ave_prc = (self.ave_prc * (self.exed_qty + 1)- float(lblShoga1v)) / (self.exed_qty)
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                      self.type,
                                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                        self.add_touch = 0
                                        # self.prf_hit = 0
                                        self.last_o = float(lblShoga1v)

                                    if self.exed_qty >= 3:
                                        if self.df.at[self.nf - 1, "test_signal"] != 3 and self.df.at[
                                            self.nf - 2, "test_signal"] == 3:
                                            self.exed_qty -= 1
                                            self.add_signal = 1
                                            self.type = "minus s_3"
                                            # self.ave_prc = (self.ave_prc * (self.exed_qty + 1) - float(lblShoga1v)) / (self.exed_qty)
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.add_touch = 0
                                            # self.prf_hit = 0
                                            self.last_o = float(lblShoga1v)

                                # if 1==0 and self.exed_qty == 0:
                                #     self.Profit += ((self.inp - float(lblShoga1v)))
                                #     self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                #             float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                #     self.piox = -3
                                #     self.in_str = 0
                                #     self.OrgMain = 'n'
                                #     self.type = self.out_type
                                #     self.turnover += 1
                                #     self.exed_qty = 0
                                #     self.exed_qty_adj = 0
                                #     self.ave_prc = 0
                                #     self.peak_touch = 0
                                #     self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                #                   self.type,
                                #                   self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.add_touch < 0 and (self.OrgMain == 's' or (self.Indep == 1 and self.OrgMain == 'b')):
                        if (self.OrgMain == 's' and price >= self.last_o + self.tick * 10 * m_add * max(self.std_diff,
                                                                                                        1)) or self.OrgMain == 'b':
                            if self.ai <= 0.9 or self.ai_long <= 1.8:  # and (self.df.loc[self.nf - 200:self.nf - 1, "add_touch"] != -2).any() == 1:
                                if self.OrgMain == 's' and self.exed_qty <= self.max_e_qty:
                                    if self.df.loc[self.nf - 100:self.nf - 1, "ai"].mean() <= 0.05:
                                        # if (self.now_trend - self.last_trend) < 0 and self.price_trend == 0:
                                        #     if (self.df.loc[self.nf - 100:self.nf - 1, "prc_s_peak"].mean()>0 and self.prc_s_peak < 2) or (price >= self.last_o + self.tick * 10 * m_add):
                                        if (
                                                self.force_on == 1 and self.force == -1) or self.force_on == 0:  # and self.cvol_c_ave <= 10:
                                            self.exed_qty += 1
                                            self.add_signal = -1
                                            self.type = "plus s1"
                                            self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(lblBhoga1v)) / (
                                                self.exed_qty)
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.add_touch = 0
                                            # self.prf_hit = 0
                                            self.last_o = float(lblBhoga1v)

                                        if self.exed_qty <= 3:
                                            if self.df.at[self.nf - 1, "add_touch"] == -2 and self.df.at[
                                                self.nf - 1, "add_touch"] != -2:
                                                self.exed_qty += 1
                                                self.add_signal = -1
                                                self.type = "plus s2"
                                                self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(
                                                    lblBhoga1v)) / (self.exed_qty)
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                              self.OrgMain, self.type,
                                                              self.profit, self.profit_opt, self.mode, self.ave_prc,
                                                              prc_o1)
                                                self.add_touch = 0
                                                # self.prf_hit = 0
                                                self.last_o = float(lblBhoga1v)

                                if 1 == 1 and self.OrgMain == 'b' and self.exed_qty > 1 and price >= self.last_o + self.tick * 15 * m_add:
                                    if 1 == 0 and ((
                                                           self.prf / self.exed_qty > 0.5 * prc_std and self.exed_qty == 1) or self.exed_qty >= 2):
                                        if (self.now_trend - self.last_trend) < 0 or self.price_trend == 0:
                                            self.exed_qty -= 1
                                            self.add_signal = -1
                                            self.type = "minus b_1"
                                            # self.ave_prc = (self.ave_prc * (self.exed_qty + 1) - float(lblShoga1v)) / (self.exed_qty)
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.add_touch = 0
                                            # self.prf_hit = 0
                                            self.last_o = float(lblBhoga1v)

                                    self.df_prc_s = self.df.loc[self.nf - 100: self.nf - 1, "prc_s"]
                                    if self.prc_s < (self.prc_s_limit * 1) and self.df_prc_s[
                                        self.df_prc_s > self.prc_s_limit * 1].count() > 1:
                                        self.exed_qty -= 1
                                        self.add_signal = -1
                                        self.type = "minus b_2"
                                        # self.ave_prc = (self.ave_prc * (self.exed_qty + 1) - float(lblShoga1v)) / (self.exed_qty)
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                      self.type,
                                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                        self.add_touch = 0
                                        # self.prf_hit = 0
                                        self.last_o = float(lblBhoga1v)

                                    if self.exed_qty >= 3:
                                        if self.df.at[self.nf - 1, "test_signal"] != -3 and self.df.at[
                                            self.nf - 2, "test_signal"] == -3:
                                            self.exed_qty -= 1
                                            self.add_signal = -1
                                            self.type = "minus b_3"
                                            # self.ave_prc = (self.ave_prc * (self.exed_qty + 1) - float(lblShoga1v)) / (self.exed_qty)
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                            self.add_touch = 0
                                            # self.prf_hit = 0
                                            self.last_o = float(lblBhoga1v)

                                # if 1==0 and self.exed_qty == 0:
                                #     self.Profit += ((float(lblBhoga1v) - self.inp))
                                #     self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                #             float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                #     self.piox = 3
                                #     self.in_str = 0
                                #     self.OrgMain = 'n'
                                #     self.type = self.out_type
                                #     self.turnover += 1
                                #     self.exed_qty = 0
                                #     self.exed_qty_adj = 0
                                #     self.ave_prc = 0
                                #     self.peak_touch = 0
                                #     self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                #                   self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # 1 per plus
                if self.OrgMain == 'b' and self.exed_qty == 1:
                    if (
                            self.std_std_prc_cvol_m_peak >= 1 and price <= self.ave_prc * 0.9925) or price <= self.ave_prc * 0.9875:
                        self.exed_qty += 1
                        self.add_signal = 1
                        self.type = "plus b3_1 per"
                        self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(lblShoga1v)) / (
                            self.exed_qty)
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                      self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.add_touch = 0
                        # self.prf_hit = 0
                        self.last_o = float(lblShoga1v)

                if self.OrgMain == 's' and self.exed_qty == 1:
                    if (
                            self.std_std_prc_cvol_m_peak >= 1 and price >= self.ave_prc * 1.0075) or price >= self.ave_prc * 1.0125:
                        self.exed_qty += 1
                        self.add_signal = -1
                        self.type = "plus s3_1 per"
                        self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(lblBhoga1v)) / (self.exed_qty)
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.add_touch = 0
                        # self.prf_hit = 0
                        self.last_o = float(lblBhoga1v)

                # add in profit
                if 1==0:
                    if self.Add_Prf == 1 and self.exed_qty < 3:
                        if self.OrgMain == 'b' and price >= self.inp + self.tick * self.exed_qty and price <= self.inp + self.tick * 8:
                            self.ave_prc = (self.ave_prc * self.exed_qty + float(lblShoga1v)) / (
                                    self.exed_qty + 1)
                            self.exed_qty += 1
                            self.add_signal = 1
                            self.type = "add_prf"
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                          self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            self.last_o = float(lblShoga1v)

                        if self.OrgMain == 's' and price <= self.inp - self.tick * self.exed_qty and price >= self.inp - self.tick * 8:
                            self.ave_prc = (self.ave_prc * self.exed_qty + float(lblBhoga1v)) / (self.exed_qty + 1)
                            self.exed_qty += 1
                            self.add_signal = -1
                            self.type = "add_prf"
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            self.last_o = float(lblBhoga1v)

                self.df.at[self.nf, "add_signal"] = self.add_signal
                # self.df.at[self.nf, "test_signal"] = self.test_signal
                self.df.at[self.nf, "new_signal"] = self.new_signal

        self.df.at[self.nf, "inp"] = self.inp
        self.df.at[self.nf, "in_str_1"] = self.in_str_1
        self.df.at[self.nf, "in_str"] = self.in_str
        self.df.at[self.nf, "cover_by_opt"] = self.cover_by_opt
        self.df.at[self.nf, "nfset"] = self.nfset
        self.df.at[self.nf, "Add_Prf"] = self.Add_Prf

        ###############################
        # hit_peak setting
        ###############################

        # prf_able
        self.prf_able = 0

        profit_band = self.profit_min_tick * self.cvol_m / self.cvol_m_cri_ave * 2  # * ee_s ** 0.6 # * ee_s
        loss_band = self.loss_max_tick  # * ee_s

        if profit_band > self.profit_min_tick * 2:
            profit_band = self.profit_min_tick * 2
        if profit_band < self.profit_min_tick:
            profit_band = self.profit_min_tick
        if loss_band > self.loss_max_tick:
            loss_band = self.loss_max_tick
        if loss_band < self.loss_max_tick / 2:
            loss_band = self.loss_max_tick / 2
        self.df.at[self.nf, "profit_band"] = profit_band

        # if self.AvePrc==0:
        #     self.AvePrc=self.inp

        m = 1
        if self.OrgMain == "b":
            # if self.mode == 1:
            # if price > self.inp + self.tick * profit_band: #self.inp + prc_std * 1:  # self.tick * profit_band:  # or price >= self.AvePrc + self.tick * profit_band:
            #     self.prf_able = 1
            #     self.prf_hit = 1
            # if price < self.inp - self.tick * loss_band: #self.inp - prc_std * 1:  # self.tick * loss_band:
            #     self.prf_able = -1
            # if self.mode == 0:

            self.nowprf = (price - self.ave_prc) / prc_std
            self.prf = (price - self.ave_prc) * self.exed_qty

            if self.which_market == 1:
                m = 4
            if self.which_market == 4:
                m = 2
            if (
                    price > self.ave_prc + prc_std * 1.5 * m and price > self.ave_prc + price * 0.08 / 100 * m) or price > self.ave_prc + price * 0.12 / 100 * m:
                self.prf_able = 1
                self.prf_hit = 1
            if (price > self.ave_prc + prc_std * 2.5 * m) or price > self.ave_prc + price * 0.25 / 100 * m:
                self.prf_able = 2
            if (price > self.ave_prc + prc_std * 5 * m) or price > self.ave_prc + price * 0.5 / 100 * m:
                self.prf_able = 3
            if price < self.ave_prc - prc_std * 1.5 * m:
                self.prf_able = -1
                self.prf_hit_inv = 1

        if self.OrgMain == "s":
            # if self.mode == 1:
            # if price < self.inp - self.tick * profit_band: #self.inp + prc_std * 1:  # self.tick * profit_band:  # or price >= self.AvePrc + self.tick * profit_band:
            #     self.prf_able = 1
            #     self.prf_hit = 1
            # if price > self.inp + self.tick * loss_band: #self.inp - prc_std * 1:  # self.tick * loss_band:
            #     self.prf_able = -1
            # if self.mode == 0:

            self.nowprf = (self.ave_prc - price) / prc_std
            self.prf = (self.ave_prc - price) * self.exed_qty

            # if self.which_market == 1:
            #     m = 4
            # if self.which_market == 4:
            #     m = 2
            if (
                    price < self.ave_prc - prc_std * 1.5 * m and price < self.ave_prc - price * 0.08 / 100 * m) or price < self.ave_prc - price * 0.12 / 100 * m:  # self.tick * profit_band:  # or price <= self.AvePrc - self.tick * profit_band:
                self.prf_able = 1
                self.prf_hit = 1
            if (price < self.ave_prc - prc_std * 2.5 * m) or price < self.ave_prc - price * 0.25 / 100 * m:
                self.prf_able = 2
            if (price < self.ave_prc - prc_std * 10 * m) or price < self.ave_prc - price * 1 / 100 * m:
                self.prf_able = 3
            if price > self.ave_prc + prc_std * 1.5 * m:  # self.tick * loss_band:
                self.prf_able = -1
                self.prf_hit_inv = 1

        self.df.at[self.nf, "prf"] = self.prf
        self.df.at[self.nf, "prf_able"] = self.prf_able
        self.df.at[self.nf, "prf_hit_inv"] = self.prf_hit_inv
        self.df.at[self.nf, "nowprf"] = self.nowprf
        self.prf_sum = (self.profit + self.prf) + (self.profit_opt + self.prf_cover * prc_std)
        self.df.at[self.nf, "prf_sum"] = self.prf_sum

        #### check_gold ####
        self.check_gold = 0
        if self.gold1 == 2 and self.gold2 != -2:
            self.check_gold = 1
        if self.gold2 == -2 and self.gold1 != 2:
            self.check_gold = -1
        self.df.at[self.nf, "check_gold"] = self.check_gold

        #### check_rsi_gold ####
        self.check_rsi_gold = 0
        if self.rsi_gold1 == 2 and self.rsi_gold2 != -2:
            self.check_rsi_gold = 1
        if self.rsi_gold2 == -2 and self.rsi_gold1 != 2:
            self.check_rsi_gold = -1
        self.df.at[self.nf, "check_rsi_gold"] = self.check_rsi_gold

        #### check_pvol_gold ####
        self.check_pvol_gold = 0
        if self.pvol_gold1 == 2 and self.pvol_gold2 != -2:
            self.check_pvol_gold = 1
        if self.pvol_gold2 == -2 and self.pvol_gold1 != 2:
            self.check_pvol_gold = -1
        self.df.at[self.nf, "check_pvol_gold"] = self.check_pvol_gold

        #### std_prc_1000 ####
        self.check_p1000_gold = 0
        if self.p1000_gold1 == 2 and self.p1000_gold2 != -2:
            self.check_p1000_gold = 1
        if self.p1000_gold2 == -2 and self.p1000_gold1 != 2:
            self.check_p1000_gold = -1
        self.df.at[self.nf, "check_p1000_gold"] = self.check_p1000_gold

        ##### std_prc, ai ######
        self.bns_check_4 = 0
        if self.nf > 610:
            # plus
            if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.loc[self.nf - 35:self.nf - 6, "std_prc_cvol_m"].mean() * 0.95:
                if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() >= self.df.loc[self.nf - 65:self.nf - 15, "ai"].mean():
                    if (self.df_ai_s[self.df_ai_s < 0.1].count() >= 50 and self.ai < 0.5) or self.std_std_prc_cvol_m <= self.std_std_prc_cvol_m_limit * 1.25:
                        if self.cover_signal_2 != -1 and self.test_signal != 3 and self.ai < 0.8:
                            # if self.check_gold == 1:
                            if (self.which_market == 3 and self.ai <= 0.75) or self.which_market != 3:
                            # if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit].count() == 0:
                                self.bns_check_4 = 1
            if self.df.loc[self.nf - 5:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.loc[self.nf - 100:self.nf - 1, "std_prc_cvol_m"].max() * 0.95:
                if self.std_prc_cvol_m < self.std_prc_cvol_m_limit * 1.25:  # self.cover_signal_2 == -1 and
                    if self.which_market != 4 and self.df.loc[self.nf - 90:self.nf - 1, "ai"].mean() < 0.6:
                        if self.rsi_peak != 3:
                            # if self.check_gold == 1:
                            self.bns_check_4 = 1.4
                    if self.which_market == 4 and self.df.loc[self.nf - 90:self.nf - 1, "ai_long"].mean() < 0.6:
                        if self.rsi_peak != 3:
                            # if self.check_gold == 1:
                            self.bns_check_4 = 1.4
            if self.bns_check_4 >= 1:
                if self.std_std_prc_cvol_m_peak >= 1 and self.ai < 0.05:
                    self.bns_check_4 == 0
            # minus
            if self.df.loc[self.nf - 20:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 35:self.nf - 15, "std_prc_cvol_m"].mean() * 1.05:
                if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() <= self.df.loc[self.nf - 65:self.nf - 15, "ai"].mean():
                    # if self.df_std_prc[self.df_std_prc >= self.std_prc_cvol_m_limit < -0.8].count() == 1 and self.std_prc_cvol_m > self.std_prc_cvol_m_limit * -0.5:
                    if (self.df_ai_s[self.df_ai_s > 0.5].count() >= 50 and self.ai > 0.05) or self.std_std_prc_cvol_m >= self.std_std_prc_cvol_m_limit * -1.25:
                        if self.cover_signal_2 != 1 and self.test_signal != -3 and self.ai_long > 0.25:
                            # if self.check_gold == -1:
                            # if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit].count() == 0:
                            self.bns_check_4 = -1
            if self.df.loc[self.nf - 5:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 100:self.nf - 1, "std_prc_cvol_m"].min() * 1.1 :
                if self.std_prc_cvol_m > self.std_prc_cvol_m_limit * -1.5:# self.cover_signal_2 == -1 and
                    if self.which_market != 4 and self.df.loc[self.nf - 90:self.nf - 1, "ai"].mean() < 0.25:
                        # if self.check_gold == -1:
                        self.bns_check_4 = -1.4
                    if self.which_market == 4 and self.df.loc[self.nf - 90:self.nf - 1, "ai_long"].mean() < 0.35:
                        # if self.check_gold == -1:
                        self.bns_check_4 = -1.4
        self.df.at[self.nf, "bns_check_4"] = self.bns_check_4

        ###############################
        #  // Out Decision //
        ###############################

        # Peak_Out
        if 1 == 1 and self.auto_cover == 0 and self.nf > self.nfset + 3:  # and count_m<count_m_cri:

            if self.nfset != 0:
                self.df.at[self.nf, "set_count"] = self.nf - self.nfset + 1
                # print("nfset: ", self.nfset)

            if self.OrgMain == "b" and ((
                                                self.which_market == 3 and self.stat_in_org == "111") or self.which_market != 3 or self.chkForb == 1):  # and self.nfset + 5 < self.nf:

                #  small_out
                if 1 == 1 and self.prf_able == 0 and self.prf_hit_inv == 1 and price > self.ave_prc + self.tick * (
                        12 / self.which_market):
                    if self.df.loc[self.nf - 10:self.nf - 1, "std_prc"].mean() < 0:
                        if prc_s < 0 and self.nf > self.nfset + 200:  # self.mode_spot != 2 and
                            if (self.ai <= 0.1):  # and self.ai_long <= self.ai_long_low):
                                if self.which_market != 1 and self.which_market != 2:
                                    self.Profit += ((float(
                                        lblBhoga1v) - self.inp))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                            float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = 1
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    self.out_type = "b_small_out"
                                    self.type = self.out_type
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                # prf

                if 1 == 0:
                    # change_test_signal
                    if self.at_test_changed == 0:
                        if (self.test_signal == -3 and self.df.loc[self.nf - 50:self.nf - 1,
                                                       "test_signal"].mean() <= -2) == -1:
                            if (price - self.ave_prc) / prc_std >= -0.25 and self.exed_qty <= 1:
                                self.peak_touch = 3
                                self.out_type = "b_at_test_sig"
                                self.in_hit_touch = -2
                                self.at_test_changed = 1

                                self.Profit += ((float(
                                    lblBhoga1v) - self.inp))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                        float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = 2
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = self.out_type
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                # time.sleep(0.2)
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price
                    if self.at_test_changed == 1:
                        if self.test_signal != -3:
                            self.at_test_changed = 0

                    # change_cover_signal_2
                    if self.chkCoverSig_2 == 1 and self.at_cover_changed == 0:
                        if self.df.loc[self.nf - 25:self.nf - 1, "cover_signal_2"].mean() == -1:
                            if (price - self.ave_prc) / prc_std >= -0.25 and self.exed_qty <= 1:
                                self.peak_touch = 3
                                self.out_type = "b_at_cover_sig"
                                self.in_hit_touch = -2
                                self.at_cover_changed = 1

                                self.Profit += ((float(
                                    lblBhoga1v) - self.inp))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                        float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = 2
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = self.out_type
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                # time.sleep(0.2)
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price
                    if self.at_cover_changed == 1:
                        if self.cover_signal_2 != -1:
                            self.at_cover_changed = 0

                # profit_out
                if 1 == 1 and (self.prf_able + self.prf_hit) >= 1 and (price - self.ave_prc) / prc_std >= 0.5:

                    if self.std_prc_peak == 2 and cvol_t > 0 and (
                            ema_25_count_m == 1 or prc_s < 0):  # and dxy_med_std !=0 cvol_t_ave > self.cvol_t_cri_ave_p * 100
                        self.peak_touch = 1
                        self.out_type = "b_at_peak"

                    if (self.std_prc_peak == 2 and self.prc_s_peak == 2) and self.prf_able == 2:
                        self.peak_touch = 2
                        self.out_type = "b_at_slope"

                    if (self.cvol_c_sig_sum >= 15):
                        self.peak_touch = 3
                        self.out_type = "b_at_sum"

                    if (prc_s > self.prc_s_limit):
                        self.peak_touch = 4
                        self.out_type = "b_at_prc_s"

                    if (self.df.at[self.nf - 3, "cvol_s_peak"] == 2 and self.df.at[self.nf - 2, "cvol_s_peak"] != 2):
                        self.peak_touch = 5
                        self.out_type = "b_at_cvol_s"

                    # if (self.df.at[self.nf-1, "cvol_s_peak"] == 2 and self.df.at[self.nf, "cvol_s_peak"] != 2):
                    #     self.peak_touch = 3
                    #     self.type = "b_at_cvol"

                    if (self.peak_touch >= 1 and self.std_prc_peak >= 0) or (
                            self.prf_able >= 2 and self.prc_s_peak >= 0):
                        # if (self.ai <= 0.1 and self.ai_long <= 0.2):
                        if (
                                self.now_trend - self.last_trend) < 0 or self.price_trend == 0 or self.prf_able >= 2 or self.dxy_decay < 0:
                            if self.ai_long <= 1.8 or self.prf_able == 3:
                                if self.nf <= 1000 or (self.nf > 1000 and self.std_std_prc_cvol_m_peak == 0):
                                    self.Profit += ((float(
                                        lblBhoga1v) - self.inp))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                            float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = 3
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    if self.prf_able == 2:
                                        self.out_type = "prf_2"
                                    self.type = self.out_type
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                    if self.peak_touch >= 2:
                        if (self.ai_long <= 1.8 or self.ai <= 0.9) or self.prf_able == 3:
                            if self.ai == 0 or (self.df.loc[self.nf - 100:self.nf - 1,
                                                "prc_s_peak"].mean() > 0 and self.prc_s_peak < 2):
                                if self.exed_qty >= 2 or (self.exed_qty == 1 and (
                                        self.std_prc_peak_cvol_m == -1 or self.prc_s_peak == 0)):
                                    self.Profit += ((float(
                                        lblBhoga1v) - self.inp))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                            float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = 3
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    self.type = self.out_type
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                    if self.peak_touch >= 1:  # and ((self.std_prc_peak < 2 and self.mode_spot != 2) or self.mode_spot == -2):
                        if (prc_s < 0 and std_prc < 0) or (self.df.at[self.nf - 1, "test_signal"] == -3 and self.df.at[
                            self.nf, "test_signal"] != -3):
                            if (self.ai <= 0.1):  # and self.ai_long <= 0.2):
                                self.Profit += ((float(
                                    lblBhoga1v) - self.inp))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                        float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = 4
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = self.out_type + "+test"
                                # self.type = "b_at_slope"
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                self.no += 1
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price

                    if 1 == 0 and cvol_t < 0 and self.prf_hit == 1:  # self.prf_able == 1:
                        if ema_25 == -1 and ema_25_count_m == 1:  # self.count_m_act:
                            if cvol_s < 0 and cvol_c <= 15 and dxy_20_medi_s < 0 and self.std_prc_peak <= -1:
                                if self.peak_touch != 1:
                                    self.Profit += ((float(
                                        lblBhoga1v) - self.inp))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                            float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = 5
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    self.type = "b_after_peak"
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                    # prc_s_out
                    self.df_prc_s_peak = self.df.loc[self.nf - 100: self.nf - 1, "prc_s_peak"]
                    if self.df.loc[self.nf - 100:self.nf - 1, "piox"].max() != 9:
                        if self.prc_s_peak < 2 and self.df_prc_s_peak[self.df_prc_s_peak >= 2].count() > 1:
                            if self.nf <= 1000 or (self.nf > 1000 and self.std_std_prc_cvol_m_peak == 0):
                                self.Profit += ((float(lblBhoga1v) - self.inp))
                                self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                        float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = 9
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = "b_prc_s_out"
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                self.no += 1
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price

                    # std_std_prc_cvol_m out
                    if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit:
                        self.Profit += ((float(lblBhoga1v) - self.inp))
                        self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                        self.piox = 9
                        self.in_str = 0
                        self.OrgMain = 'n'
                        self.type = "std_std_out_1"
                        self.turnover += 1
                        self.exed_qty = 0
                        self.exed_qty_adj = 0
                        self.ave_prc = 0
                        self.peak_touch = 0
                        self.in_hit_touch = 0
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.no += 1
                        # self.last_cover_prc = 0
                        # self.cover_out_prc = price

                    # std_std_peak_out
                    if self.df.at[self.nf - 1, "std_std_prc_cvol_m_peak"] == 3 and self.std_std_prc_cvol_m_peak != 3:
                        self.Profit += ((float(lblBhoga1v) - self.inp))
                        self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                float(lblBhoga1v) + self.ave_fprc) * self.fee_rate) * self.exed_qty
                        self.piox = 9
                        self.in_str = 0
                        self.OrgMain = 'n'
                        self.type = "b_std_std_out_3"
                        self.turnover += 1
                        self.exed_qty = 0
                        self.exed_qty_adj = 0
                        self.ave_prc = 0
                        self.peak_touch = 0
                        self.in_hit_touch = 0
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.no += 1
                        # self.last_cover_prc = 0
                        # self.cover_out_prc = price

                    # trend_out
                    if ((self.now_trend - self.last_trend) < 0 and self.df.loc[self.nf - 20:self.nf - 1,
                                                                   "cvol_m_sig"].mean() != 0) or self.now_trend == 0:
                        self.Profit += ((float(lblBhoga1v) - self.inp))
                        self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                    float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                        self.piox = 6
                        self.in_str = 0
                        self.OrgMain = 'n'
                        self.type = "b_trend_out"
                        self.turnover += 1
                        self.exed_qty = 0
                        self.exed_qty_adj = 0
                        self.ave_prc = 0
                        # self.time_out = 1
                        self.peak_touch = 0
                        self.in_hit_touch = 0
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.no += 1
                        # self.last_cover_prc = 0

                # trend2 out
                if self.which_market != 3:
                    if self.dOrgMain_new_bns2 == -2 and (price - self.ave_prc) / prc_std <= -0.15:
                        if self.std_prc_peak_cvol_m < self.df.loc[self.nf - 50:self.nf - 5,
                                                      "std_prc_peak_cvol_m"].mean():
                            if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 1:
                                self.Profit += ((float(lblBhoga1v) - self.inp))
                                self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                        float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = 9
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = "b_trend_out2"
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                self.no += 1
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price

                # time_out
                if self.which_market == 3 and now.hour == 15 and now.minute >= 30 and self.exed_qty > 0 and self.time_out == 0:
                    self.Profit += ((float(lblBhoga1v) - self.inp))
                    self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                    self.piox = 6
                    self.in_str = 0
                    self.OrgMain = 'n'
                    self.type = "b_time_out"
                    self.turnover += 1
                    self.exed_qty = 0
                    self.exed_qty_adj = 0
                    self.ave_prc = 0
                    self.time_out = 1
                    self.peak_touch = 0
                    self.in_hit_touch = 0
                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                    self.no += 1
                    # self.last_cover_prc = 0
                    # self.cover_out_prc = price

                # bns_check3_out
                if self.df.at[self.nf - 2, "bns_check"] == -3 and self.exed_qty >= 1:
                    if self.bns_check > -3 and self.std_prc_peak < 1:
                        if (price - self.ave_prc) / prc_std >= 0.2:
                            self.Profit += ((float(lblBhoga1v) - self.inp))
                            self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                        float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                            self.piox = 7
                            self.in_str = 0
                            self.OrgMain = 'n'
                            self.type = "b_bns_check3_out"
                            self.turnover += 1
                            self.exed_qty = 0
                            self.exed_qty_adj = 0
                            self.ave_prc = 0
                            self.peak_touch = 0
                            self.in_hit_touch = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            self.no += 1
                            # self.last_cover_prc = 0
                            # self.cover_out_prc = price

                # AI_out b에서는 비활성화, s에서는 활성화
                # if 1==0 and self.type != "init 7" and self.ai == 0 and self.df.loc[self.nf - 200:self.nf - 1, "ai"].mean() <= 0.01:
                if 1 == 0 and self.ai == 0 and self.df.loc[self.nf - 200:self.nf - 1, "ai"].mean() <= 0.01:
                    if self.prc_s_peak >= 0:  # and self.df_conv.mean() <= 0.05:
                        if self.df.loc[self.nf - 50:self.nf - 1, "prc_s_peak"].mean() >= 0:
                            self.Profit += ((float(lblBhoga1v) - self.inp))
                            self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                                    float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                            self.piox = 9
                            self.in_str = 0
                            self.OrgMain = 'n'
                            self.type = "b_AI_out"
                            self.turnover += 1
                            self.exed_qty = 0
                            self.exed_qty_adj = 0
                            self.ave_prc = 0
                            self.peak_touch = 0
                            self.in_hit_touch = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            self.no += 1
                            # self.last_cover_prc = 0
                            # self.cover_out_prc = price

                # # AI_out + short_out
                # if self.nf>=300 and self.which_market == 3 and self.ai == 0 and self.df.loc[self.nf - 200:self.nf - 1, "ai"].mean() <= 0.01:
                #     if self.force == -1 and self.ai == 0:
                #         df_prc_s = self.df.loc[self.nf - 100: self.nf - 1, "prc_s"]
                #         if self.prc_s < (self.prc_s_limit * 1) and df_prc_s[df_prc_s > self.prc_s_limit * 1].count() > 1:
                #             self.Profit += ((float(lblBhoga1v) - self.inp))
                #             self.profit += ((float(lblBhoga1v) - self.ave_prc) - (
                #                         float(lblBhoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                #             self.piox = 8
                #             self.in_str = 0
                #             self.OrgMain = 'n'
                #             self.type = "b_AI_out"
                #             self.turnover += 1
                #             self.exed_qty = 0
                #             self.exed_qty_adj = 0
                #             self.ave_prc = 0
                #             self.peak_touch = 0
                #             self.in_hit_touch = 0
                #             self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                #                           self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                #             self.no += 1
                #             # self.last_cover_prc = 0
                #             # self.cover_out_prc = price
                #

            if self.OrgMain == "s" and ((
                                                self.which_market == 3 and self.stat_in_org == "111") or self.which_market != 3 or self.chkForb == 1):  # and self.nfset < self.nf:

                #  small_out
                if 1 == 1 and self.prf_able == 0 and self.prf_hit_inv == 1 and price < self.ave_prc - self.tick * (
                        12 / self.which_market):
                    if self.df.loc[self.nf - 10:self.nf - 1, "std_prc"].mean() > 0:
                        if prc_s > 0 and self.nf > self.nfset + 200:  # self.mode_spot != -2 and
                            if (self.ai >= 0.9 and self.ai_long >= 1.8):
                                if self.which_market != 1 and self.which_market != 2:
                                    self.Profit += ((self.inp - float(
                                        lblShoga1v)))  # - (float(lblBhoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                            float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = -1
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    self.type = "s_small_out"
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                # prf

                if 1 == 0:
                    # change_test_signal
                    if self.at_test_changed == 0:
                        if (self.test_signal == 3 and self.df.loc[self.nf - 50:self.nf - 1, "test_signal"].mean() >= 2):
                            if (self.ave_prc - price) / prc_std >= -0.25 and self.exed_qty <= 1:
                                self.peak_touch = -3
                                self.out_type = "s_at_test_sig"
                                self.in_hit_touch = 2
                                self.at_test_changed = 1

                                self.Profit += ((self.inp - float(
                                    lblShoga1v)))  # - (float(lblShoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                        float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = -2
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = self.out_type
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                self.no += 1
                                # time.sleep(0.2)
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price
                    if self.at_test_changed == 1:
                        if self.test_signal != 3 or self.cover_signal_2 != 1:
                            self.at_test_changed = 0

                    # change_cover_signal_2
                    if self.chkCoverSig_2 == 1 and self.at_cover_changed == 0:
                        if self.df.loc[self.nf - 25:self.nf - 1, "cover_signal_2"].mean() == 1:
                            if (self.ave_prc - price) / prc_std >= -0.25 and self.exed_qty <= 1:
                                self.peak_touch = -3
                                self.out_type = "s_at_cover_sig"
                                self.in_hit_touch = 2
                                self.at_cover_changed = 1

                                self.Profit += ((self.inp - float(
                                    lblShoga1v)))  # - (float(lblShoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                        float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = -2
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = self.out_type
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                self.no += 1
                                time.sleep(0.2)
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price
                    if self.at_cover_changed == 1:
                        if self.cover_signal_2 != 1:
                            self.at_cover_changed = 0

                # profit_out
                if 1 == 1 and (self.prf_able + self.prf_hit) >= 1 and (self.ave_prc - price) / prc_std >= 0.5:

                    if self.std_prc_peak == -2 and cvol_t < 0 and (
                            ema_25_count_m == 1 or prc_s > 0):  # and cvol_s_sig == 1
                        self.peak_touch = -1
                        self.out_type = "s_at_peak"

                    if (self.std_prc_peak == -2 and self.prc_s_peak == -2) and self.prf_able == 2:
                        self.peak_touch = -2
                        self.out_type = "s_at_slope"

                    if (self.cvol_c_sig_sum <= -15):
                        self.peak_touch = -3
                        self.out_type = "s_at_sum"

                    if (prc_s < self.prc_s_limit * -1):
                        self.peak_touch = -4
                        self.out_type = "s_at_prc_s"

                    if (self.df.at[self.nf - 3, "cvol_s_peak"] == 2 and self.df.at[self.nf - 2, "cvol_s_peak"] != 2):
                        self.peak_touch = -5
                        self.out_type = "s_at_cvol_s"

                    # if (self.df.at[self.nf-1, "cvol_s_peak"] == 2 and self.df.at[self.nf, "cvol_s_peak"] != 2):
                    #     self.peak_touch = -3
                    #     self.type = "s_at_cvol"

                    if (self.peak_touch <= -1 and self.std_prc_peak == 0) or (
                            self.prf_able >= 2 and self.prc_s_peak >= 0):
                        if (
                                self.now_trend - self.last_trend) > 0 or self.price_trend == 1 or self.prf_able >= 2 or self.dxy_decay > 0:
                            if (wc_sXY_ave != 0 and self.ai_long >= self.ai_long_low) or self.prf_able == 3:
                                if self.nf <= 1000 or (self.nf > 1000 and self.std_std_prc_cvol_m_peak == 0):
                                    self.Profit += ((self.inp - float(
                                        lblShoga1v)))  # - (float(lblShoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                            float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = -3
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    if self.prf_able == 2:
                                        self.out_type = "prf_2"
                                    self.type = self.out_type
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                    if self.peak_touch <= -2:
                        if (self.ai_long >= self.ai_long_low and self.ai >= 0.1) or self.prf_able == 3:
                            if self.ai_long >= 1.8 or (self.df.loc[self.nf - 100:self.nf - 1,
                                                       "prc_s_peak"].mean() < 0 and self.prc_s_peak > -2):
                                if self.exed_qty >= 2 or (self.exed_qty == 1 and (
                                        self.std_prc_peak_cvol_m == -1 or self.prc_s_peak == 0)):
                                    self.Profit += ((self.inp - float(
                                        lblShoga1v)))  # - (float(lblShoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                            float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = -3
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    self.type = self.out_type
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                    if self.peak_touch <= -1:  # and ((self.std_prc_peak > -2 and self.mode_spot != -2) or self.mode_spot == 2):
                        if (prc_s > 0 and std_prc > 0) or (self.df.at[self.nf - 1, "test_signal"] == 3 and self.df.at[
                            self.nf, "test_signal"] != 3):
                            if (self.ai >= 0.9 and self.ai_long >= 1.8):
                                self.Profit += ((self.inp - float(
                                    lblShoga1v)))  # - (float(lblShoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                        float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = -4
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = self.out_type + "+test"
                                # self.type = "s_at_slope"
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                self.no += 1
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price

                    if 1 == 0 and cvol_t > 0 and self.prf_hit == 1:  #:and self.prf_able == 1
                        if ema_25 == -1 and ema_25_count_m == 1:  # self.count_m_act:
                            if cvol_s < 0 and cvol_c >= 5 and dxy_20_medi_s > 0 and self.std_prc_peak >= 1:
                                if self.peak_touch != -1:
                                    self.Profit += ((self.inp - float(
                                        lblShoga1v)))  # - (float(lblShoga1v) + self.inp) * self.fee_rate) #* abs(self.ExedQty)
                                    self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                            float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = -5
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    self.type = "s_after_peak"
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                    # prc_s_out
                    self.df_prc_s_peak = self.df.loc[self.nf - 100: self.nf - 1, "prc_s_peak"]
                    if self.df.loc[self.nf - 100:self.nf - 1, "piox"].min() != -9:
                        if self.prc_s_peak > -2 and self.df_prc_s_peak[self.df_prc_s_peak <= -2].count() > 1:
                            if self.nf <= 1000 or (self.nf > 1000 and self.std_std_prc_cvol_m_peak == 0):
                                self.Profit += ((self.inp - float(lblShoga1v)))
                                self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                        float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                self.piox = -9
                                self.in_str = 0
                                self.OrgMain = 'n'
                                self.type = "s_prc_s_out"
                                self.turnover += 1
                                self.exed_qty = 0
                                self.exed_qty_adj = 0
                                self.ave_prc = 0
                                self.peak_touch = 0
                                self.in_hit_touch = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                              self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                self.no += 1
                                # self.last_cover_prc = 0
                                # self.cover_out_prc = price

                    # std_std_prc_cvol_m out
                    if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit:
                        self.Profit += ((self.inp - float(lblShoga1v)))
                        self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                        self.piox = -9
                        self.in_str = 0
                        self.OrgMain = 'n'
                        self.type = "std_std_out_1"
                        self.turnover += 1
                        self.exed_qty = 0
                        self.exed_qty_adj = 0
                        self.ave_prc = 0
                        self.peak_touch = 0
                        self.in_hit_touch = 0
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                      self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.no += 1
                        # self.last_cover_prc = 0
                        # self.cover_out_prc = price

                    # std_std_peak_out
                    if self.df.at[self.nf - 1, "std_std_prc_cvol_m_peak"] == 3 and self.std_std_prc_cvol_m_peak != 3:
                        self.Profit += ((self.inp - float(lblShoga1v)))
                        self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                        self.piox = -9
                        self.in_str = 0
                        self.OrgMain = 'n'
                        self.type = "s_std_std_out_3"
                        self.turnover += 1
                        self.exed_qty = 0
                        self.exed_qty_adj = 0
                        self.ave_prc = 0
                        self.peak_touch = 0
                        self.in_hit_touch = 0
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.no += 1
                        # self.last_cover_prc = 0
                        # self.cover_out_prc = price

                    # trend_out
                    if ((self.now_trend - self.last_trend) > 0 and self.df.loc[self.nf - 20:self.nf - 1,
                                                                   "cvol_m_sig"].mean() != 0) or self.now_trend == 1:
                        self.Profit += ((self.inp - float(lblShoga1v)))
                        self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                    float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                        self.piox = -6
                        self.in_str = 0
                        self.OrgMain = 'n'
                        self.type = "s_trend_out"
                        self.turnover += 1
                        self.exed_qty = 0
                        self.exed_qty_adj = 0
                        self.ave_prc = 0
                        # self.time_out = 1
                        self.peak_touch = 0
                        self.in_hit_touch = 0
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                      self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.no += 1
                        # self.last_cover_prc = 0
                        # self.cover_out_prc = price

                    # trend2 out
                    if self.which_market != 3:
                        if self.dOrgMain_new_bns2 == 2 and (self.ave_prc - price) / prc_std <= -0.15:
                            if self.std_prc_peak_cvol_m > self.df.loc[self.nf - 50:self.nf - 5,
                                                          "std_prc_peak_cvol_m"].mean():
                                if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 1:
                                    self.Profit += ((self.inp - float(lblShoga1v)))
                                    self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                            float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                                    self.piox = -9
                                    self.in_str = 0
                                    self.OrgMain = 'n'
                                    self.type = "s_trend_out2"
                                    self.turnover += 1
                                    self.exed_qty = 0
                                    self.exed_qty_adj = 0
                                    self.ave_prc = 0
                                    self.peak_touch = 0
                                    self.in_hit_touch = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    self.no += 1
                                    # self.last_cover_prc = 0
                                    # self.cover_out_prc = price

                # time_out
                if self.which_market == 3 and now.hour == 15 and now.minute >= 30 and self.exed_qty > 0 and self.time_out == 0:
                    self.Profit += ((self.inp - float(lblShoga1v)))
                    self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                    self.piox = -6
                    self.in_str = 0
                    self.OrgMain = 'n'
                    self.type = "s_time_out"
                    self.turnover += 1
                    self.exed_qty = 0
                    self.exed_qty_adj = 0
                    self.ave_prc = 0
                    self.time_out = 1
                    self.peak_touch = 0
                    self.in_hit_touch = 0
                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                  self.type,
                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                    self.no += 1
                    # self.last_cover_prc = 0
                    # self.cover_out_prc = price

                # bns_check3_out
                if self.df.at[self.nf - 2, "bns_check"] == 3 and self.exed_qty >= 1:
                    if self.bns_check < 3 and self.std_prc_peak > -1:
                        if (price - self.ave_prc) / prc_std <= -0.2:
                            self.Profit += ((self.inp - float(lblShoga1v)))
                            self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                    float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                            self.piox = -7
                            self.in_str = 0
                            self.OrgMain = 'n'
                            self.type = "s_check3_out"
                            self.turnover += 1
                            self.exed_qty = 0
                            self.exed_qty_adj = 0
                            self.ave_prc = 0
                            self.peak_touch = 0
                            self.in_hit_touch = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                          self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            self.no += 1
                            # self.last_cover_prc = 0
                            # self.cover_out_prc = price

                # AI_out
                if 1 == 0 and self.nf >= 300 and self.type != "init 7":
                    if self.ai_long >= 1.95 and self.df.loc[self.nf - 200:self.nf - 1,
                                                "ai_long"].mean() >= 1.95:  # self.which_market == 3 and
                        self.Profit += ((self.inp - float(lblShoga1v)))
                        self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                                float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                        self.piox = -8
                        self.in_str = 0
                        self.OrgMain = 'n'
                        self.type = "s_AI_out"
                        self.turnover += 1
                        self.exed_qty = 0
                        self.exed_qty_adj = 0
                        self.ave_prc = 0
                        self.peak_touch = 0
                        self.in_hit_touch = 0
                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                      self.type,
                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                        self.no += 1
                        # self.last_cover_prc = 0
                        # self.cover_out_prc = price

                # if self.nf>=300 and self.which_market != 3 and self.ai_long == 2 and self.df.loc[self.nf - 200:self.nf - 1, "ai_long"].mean() >= 1.95:
                #     # if self.ai_long == 2: #self.force == 1 and
                #     df_prc_s = self.df.loc[self.nf - 100: self.nf - 1, "prc_s"]
                #     if self.prc_s > (self.prc_s_limit * -1) and df_prc_s[df_prc_s < self.prc_s_limit * -1].count() > 1:
                #         self.Profit += ((self.inp - float(lblShoga1v)))
                #         self.profit += ((self.ave_prc - float(lblShoga1v)) - (
                #                 float(lblShoga1v) + self.ave_prc) * self.fee_rate) * self.exed_qty
                #         self.piox = -8
                #         self.in_str = 0
                #         self.OrgMain = 'n'
                #         self.type = "s_AI_out"
                #         self.turnover += 1
                #         self.exed_qty = 0
                #         self.exed_qty_adj = 0
                #         self.ave_prc = 0
                #         self.peak_touch = 0
                #         self.in_hit_touch = 0
                #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                #                       self.type,
                #                       self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                #         self.no += 1

        self.df.at[self.nf, "peak_touch"] = self.peak_touch
        self.df.at[self.nf, "piox"] = self.piox
        self.df.at[self.nf, "profit"] = self.profit
        self.df.at[self.nf, "Profit"] = self.Profit
        self.df.at[self.nf, "type"] = self.type
        self.df.at[self.nf, "add_count"] = self.add_count
        self.df.at[self.nf, "minus_count"] = self.minus_count
        self.df.at[self.nf, "at_test_changed"] = self.at_test_changed
        self.df.at[self.nf, "at_cover_changed"] = self.at_cover_changed
        # print 'piox = ', self.piox

        ###############################
        #  // RESET & ETC //
        ###############################

        if self.OrgMain == "b":
            self.d_OMain = 1
            # self.piox = 0
            print('test_sig %d  /add_count %d  /minus_count %d  /profit %0.2f /Profit %0.2f' % (
                self.test_signal, self.add_count - 1, self.minus_count - 1, self.profit, self.Profit))
        elif self.OrgMain == "s":
            self.d_OMain = -1
            # self.piox = 0
            print('test_sig %d  /add_count %d  /minus_count %d  /profit %0.2f /Profit %0.2f' % (
                self.test_signal, self.add_count - 1, self.minus_count - 1, self.profit, self.Profit))
        elif self.OrgMain == "n":
            if self.d_OMain != 5:  # in case Cover was not released due to add_signal at the same time
                self.d_OMain = 0
            self.add = 0
            self.minus = 0
            self.add_count = 1
            self.minus_count = 1
            self.prf = 0
            self.inp = 0
            self.last_o = 0
            self.exed_qty = 0
            self.ave_prc = 0
            # self.AvePrc = 0
            # self.ExedQty = 0
            self.nfset = 0
            self.prf_hit = 0
            self.add_touch = 0
            self.cover_by_opt = 0
            self.in_hit = 0
            self.reorder_msg_done = 0
            self.reorder_msg_done_cov = 0
            # self.in_hit_touch = 0
            # self.cover_signal = 0
            # self.cover_ordered = 0
            # self.last_cover_prc = 0
            # self.cover_out_prc = 0
            # print('/profit %0.2f' % (self.profit))
            # print('/Profit %0.2f' % (self.Profit))

        # # minus
        # if self.minus == 1:
        #     self.d_OMain = 2
        # if self.minus == -1:
        #     self.d_OMain = -2

        # # add
        # if self.add == 1:
        #     self.d_OMain = 3
        # if self.add == -1:
        #     self.d_OMain = -3

        ############
        # add_signal
        ############
        if self.OrgMain != "n" and abs(self.exed_qty) < self.add_limit:  # and self.exed_qty % 2 == 0:
            if self.add_signal == 1:
                self.d_OMain = 3
            if self.add_signal == -1:
                self.d_OMain = -3

            if 1 == 1 and self.nf >= 200:
                if self.df.at[self.nf - 1, "add_signal"] == 1 and self.df.at[self.nf - 1, "d_OMain"] != 3:
                    self.d_OMain = 3
                if self.df.at[self.nf - 1, "add_signal"] == -1 and self.df.at[self.nf - 1, "d_OMain"] != -3:
                    self.d_OMain = -3

        ############
        # ReOrder_signal
        ############

        # main
        if 1 == 1 and self.nf > 200 and self.reorder_msg_done == 0 and self.chkForb != 1:
            if (self.stat_in_org == "110" and self.stat_out_org == "111") or (
                    self.stat_in_org == "111" and self.stat_out_org == "110"):
                df_ord = self.df.loc[self.nf - 30:self.nf - 1, "OrdNo"]
                if self.reorder_msg == 0 and df_ord[df_ord != str(0)].count() >= 25:
                    self.d_OMain = 5 # REORDER to Main
                    self.reorder_msg = 1
                    self.reorder_msg_done = 1

        # cover
        if 1 == 1 and self.nf > 300 and self.reorder_msg_done_cov == 0 and self.chkForb != 1 and self.acc_uninfied != 1:
            if abs(self.df.loc[self.nf - 30:self.nf - 1, "cover_ordered"].mean()) == 1:
                if self.df.loc[self.nf - 30:self.nf - 1, "cover_order_exed"].mean() == 0:
                    self.reorder_msg = 2
                    self.d_OMain = 7 # REORDER to Cover'
                    self.reorder_msg_done_cov = 1
                    bot1.sendMessage(chat_id="322233222", text="cover_in_not_exed")
                    # self.reorder_msg_done_cov = 1
            if self.df.loc[self.nf - 30:self.nf - 1, "cover_ordered"].mean() == 0:
                if abs(self.df.loc[self.nf - 30:self.nf - 1, "cover_order_exed"].mean()) == 1:
                    self.reorder_msg = 2
                    self.d_OMain = -7  # REORDER to Cover'
                    self.reorder_msg_done_cov = 1
                    bot1.sendMessage(chat_id="322233222", text="cover_out_not_exed")
                    # self.reorder_msg_done_cov = 1

        self.df.at[self.nf, "reorder_msg"] = self.reorder_msg
        self.df.at[self.nf, "reorder_msg_done"] = self.reorder_msg_done
        self.df.at[self.nf, "reorder_msg_done_cov"] = self.reorder_msg_done_cov

        ############
        # order through cover_signal
        ############

        # term
        self.cover_term = 0
        if self.nf >= 200:
            if self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0:
                # if (self.which_market == 3 and now.hour != 15) or self.which_market != 3:
                self.cover_term = 1

        # cover
        if self.nf >= 600 and ((self.dynamic_cover == 1 and self.chkCover == 1 and self.nf >= 600 and self.cover_term == 1 and self.exed_qty >= 1) or self.auto_cover != 0):

            # bns_check0
            # if 1==0 and self.df_bns_check_ss[self.df_bns_check_ss > 0].count() >= 1 and self.ai_long > 1.5 and self.ai > 0.5:
            #     if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 35:self.nf - 6,"std_prc_cvol_m"].mean() * 1.02:
            #         if self.df.loc[self.nf - 15:self.nf - 1, "ai"].mean() <= self.df.loc[self.nf - 65:self.nf - 15,"ai"].mean():
            #             if self.df_std_prc[self.df_std_prc >= self.std_prc_cvol_m_limit * 0.8].count() >= 1 and self.std_prc_cvol_m < self.std_prc_cvol_m_limit * 0.5:
            #                 if self.cover_signal_2 != 1 and self.test_signal != 3:
            #                     self.bns_check = -0.4
            #                     self.bns_check_3 = -0.4
            #                     if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1, "cover_ordered"].mean() == 0 and (
            #                             self.stat_in_org == "111" or self.chkForb == 1):
            #                         if self.type != "s-d-out_peaklimit":
            #                             self.d_OMain = -4
            #                             self.cover_ordered = -1
            #                             self.cover_in_prc = price
            #                             self.cover_in_nf = self.nf
            #                             self.cover_in_time = now.minute
            #                             self.type = "cover_s_bns_check0_new"
            #                             # self.sys_force_out = 0
            #                             # self.last_cover_prc = price
            #                             self.cover_out_prc = 0
            #                             self.cover_by_opt = 0
            #                             # self.type = self.cover_tyfpe
            #                             self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
            #                                           self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            if self.t_gray <= 0 and self.df_bns_check_ss[self.df_bns_check_ss < 0].count() >= 1:
                if (self.t_gray < 0 or self.triple_last < 0) and self.t_gray_strong <= 0:
                # if self.pvol_last < 0 and self.gold_last < 0 and self.rsi_last < 0 and self.p1000_last < 0:# and self.dOrgMain_new_bns2 < 3:
                    if self.df_bns_check_s[self.df_bns_check_s >= 3].count() == 0:
                        if self.df_prc_s[self.df_prc_s > self.prc_s_limit * 1].count() == 0:
                        # if self.df.loc[self.nf - 10:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 30:self.nf - 1, "std_prc_cvol_m"].mean() * 1.02:
                        #     if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() <= self.df.loc[self.nf - 70:self.nf - 20,"ai"].mean():
                        #         # if (self.df_ai_s[self.df_ai_s > 0.5].count() >= 10 and self.ai > 0.05) or self.df.loc[self.nf - 100:self.nf - 1, "ai"].mean() <= 0.1:
                        #         if self.df_bns_check_s[self.df_bns_check_s <= -2].count() == 0 and self.cover_signal_2 != 1:
                        #             # if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit].count() == 0:
                        #             if self.test_signal != 3:
                            self.bns_check_3 = -0.5
                            self.bns_check_5 = -0.5
                            if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                                # if self.type != "s-d-out_peaklimit":
                                self.d_OMain = -4
                                self.cover_ordered = -1
                                self.cover_in_prc = price
                                self.cover_in_nf = self.nf
                                self.cover_in_time = now.minute
                                self.type = "cover_s_bns_check0 : " + str(self.bns_check_last)
                                # self.sys_force_out = 0
                                # self.last_cover_prc = price
                                self.cover_out_prc = 0
                                self.cover_by_opt = 0
                                # self.type = self.cover_tyfpe
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            # if 1==1 and self.df.loc[self.nf - 90:self.nf - 1, "ai"].mean() < 0.05:
                            #     if self.cover_signal_2 == -1 and self.std_prc_cvol_m < self.std_prc_cvol_m_limit:
                            #         if self.df.loc[self.nf - 5:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 100:self.nf - 1, "std_prc_cvol_m"].min() * 0.9 :
                            #             if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                            #                 if self.type != "s-d-out_peaklimit":
                            #                     self.d_OMain = -4
                            #                     self.cover_ordered = -1
                            #                     self.cover_in_prc = price
                            #                     self.cover_in_nf = self.nf
                            #                     self.cover_in_time = now.minute
                            #                     self.type = "cover_s_bns_check0- : " + str(self.bns_check_last)
                            #                     # self.sys_force_out = 0
                            #                     # self.last_cover_prc = price
                            #                     self.cover_out_prc = 0
                            #                     self.cover_by_opt = 0
                            #                     # self.type = self.cover_tyfpe
                            #                     self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                            #                                   self.type, self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # if self.df_bns_check_ss[self.df_bns_check_ss < 0].count() >= 1 and self.ai < 0.05: #
            #     if self.std_prc_cvol_m < self.std_prc_cvol_m_limit * -0.8:
            #         if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.8:
            #             if self.cover_signal_2 == -1 and self.test_signal >= 2:
            #                 # self.bns_check_4 = -2
            #                 if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
            #                     if 1==0 and self.type != "s-d-out_peaklimit":
            #                         self.d_OMain = -4
            #                         self.cover_ordered = -1
            #                         self.cover_in_prc = price
            #                         self.cover_in_nf = self.nf
            #                         self.cover_in_time = now.minute
            #                         self.type = "cover_s_bns_check0_strong"
            #                         # self.sys_force_out = 0
            #                         # self.last_cover_prc = price
            #                         self.cover_out_prc = 0
            #                         self.cover_by_opt = 0
            #                         # self.type = self.cover_type
            #                         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
            #                                       self.type,self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # if 1==0 and self.df_bns_check_ss[self.df_bns_check_ss < 0].count() >= 1 and self.ai < 0.2:
            #     if self.df.loc[self.nf - 10:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.loc[self.nf - 30:self.nf - 1, "std_prc_cvol_m"].mean() * 0.98:
            #         if self.df.loc[self.nf - 15:self.nf - 1, "ai"].mean() >= self.df.loc[self.nf - 65:self.nf - 15,"ai"].mean():
            #             if self.df_std_prc[self.df_std_prc <= self.std_prc_cvol_m_limit * -0.8].count() >= 1 and self.std_prc_cvol_m > self.std_prc_cvol_m_limit * -0.5:
            #                 if self.cover_signal_2 != -1 and self.test_signal != -3:
            #                     self.bns_check = 0.4
            #                     self.bns_check_3 = 0.4
            #                     if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
            #                         if self.type != "b-d-out_peaklimit":
            #                             self.d_OMain = 4
            #                             self.cover_ordered = 1
            #                             self.cover_in_prc = price
            #                             self.cover_in_nf = self.nf
            #                             self.cover_in_time = now.minute
            #                             self.type = "cover_b_bns_check0_new"
            #                             # self.sys_force_out = 0
            #                             # self.last_cover_prc = price
            #                             self.cover_out_prc = 0
            #                             self.cover_by_opt = 0
            #                             # self.type = self.cover_type
            #                             self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
            #                                           self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            if self.t_gray >= 0 and self.df_bns_check_ss[self.df_bns_check_ss > 0].count() >= 1:# and self.ai < 0.8:# and self.bns_check != 1:
                if (self.t_gray > 0 or self.triple_last > 0) and self.t_gray_strong >= 0:
                # if self.pvol_last > 0 and self.gold_last > 0 and self.rsi_last > 0 and self.p1000_last > 0:# and self.dOrgMain_new_bns2 < 3:
                # if self.bns_check_last > 0 and self.check_gold > 0 and self.check_rsi_gold > 0 and self.dOrgMain_new_bns2 > -3:
                    if self.df_bns_check_s[self.df_bns_check_s <= -3].count() == 0:
                        if self.df_prc_s[self.df_prc_s < self.prc_s_limit * -1].count() == 0:
                        # if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.loc[self.nf - 35:self.nf - 6, "std_prc_cvol_m"].mean() * 0.98:
                        #     if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() >= self.df.loc[self.nf - 70:self.nf - 20,"ai"].mean():
                        #         # if (self.df_ai_s[self.df_ai_s < 0.1].count() >= 10 and self.ai < 0.5) or self.df.loc[self.nf - 100:self.nf - 1, "ai"].mean() >= 0.4:
                        #         if self.df_bns_check_s[self.df_bns_check_s >= 2].count() == 0 and self.cover_signal_2 != -1:
                        #             # if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit].count() == 0:
                        #             if self.test_signal != -3:
                            self.bns_check_3 = 0.5
                            self.bns_check_5 = 0.5
                            if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                                if self.type != "b-d-out_peaklimit":
                                    self.d_OMain = 4
                                    self.cover_ordered = 1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_b_bns_check0 : " + str(self.bns_check_last)
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # if self.df_bns_check_ss[self.df_bns_check_ss > 0].count() >= 1 and self.ai > 0.2:
            #     if self.std_prc_cvol_m > self.std_prc_cvol_m_limit * 0.8:
            #         if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 0.8:
            #             if self.cover_signal_2 == 1 and self.test_signal <= -2:
            #                 # self.bns_check_4 = 2
            #                 if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
            #                     if 1==0 and self.type != "b-d-out_peaklimit":
            #                         self.d_OMain = 4
            #                         self.cover_ordered = 1
            #                         self.cover_in_prc = price
            #                         self.cover_in_nf = self.nf
            #                         self.cover_in_time = now.minute
            #                         self.type = "cover_b_bns_check0_strong"
            #                         # self.sys_force_out = 0
            #                         # self.last_cover_prc = price
            #                         self.cover_out_prc = 0
            #                         self.cover_by_opt = 0
            #                         # self.type = self.cover_type
            #                         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
            #                                       self.type,self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # triple + p1000
            self.bns_triple = 0
            # self.df_bns_triple = self.df.loc[self.nf - 25: self.nf - 1, "bns_triple"]

            if self.std_prc_peak != 2:
                if self.p1000_last == 1:
                    if self.df.at[self.nf - 2, "p1000_last"] != 1:
                        if self.std_prc_cvol_m > 0:
                            self.bns_triple = 1
                    elif self.df.at[self.nf - 2, "p1000_last"] == 1:
                        if self.triple_last == 1 and self.df.at[self.nf - 2, "triple_last"] != 1:
                            self.bns_triple = 1
                    if self.std_prc_cvol_m > 0:
                            self.bns_triple = 1

            if self.std_prc_peak != -2:
                if self.p1000_last == -1:
                    if self.df.at[self.nf - 2, "p1000_last"] != -1:
                        if self.std_prc_cvol_m:
                            self.bns_triple = -1
                    elif self.df.at[self.nf - 2, "p1000_last"] == -1:
                        if self.triple_last == -1 and self.df.at[self.nf - 2, "triple_last"] != -1:
                            self.bns_triple = -1
                    if self.std_prc_cvol_m < 0:
                            self.bns_triple = -1

            self.df.at[self.nf, "bns_triple"] = self.bns_triple

            if (self.bns_triple == 1 and self.df.at[self.nf - 2, "bns_triple"] != 1) or self.df.loc[self.nf - 15:self.nf - 1, "bns_triple"].mean() == 1:
                if self.sum_peak >= -2 and self.df.loc[self.nf - 15:self.nf - 1, "bns_triple"].mean() == 1:
                    if self.cover_signal_2 >= 1 or self.df.loc[self.nf - 15:self.nf - 1, "cover_signal_2"].mean() >= 0:
                        if self.triple_last_last >= 0 and self.t_gray > -0.5: #!= -1:
                            if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit].count() == 0:
                                # if self.type !=  "b-d-out-prc_out_g3":
                                self.bns_check_3 = 1
                                self.bns_check_6 = 1
                                if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                                    self.d_OMain = 4
                                    self.cover_ordered = 1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_b_triple"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            if self.df_gray_strong[self.df_gray_strong == -2].count() >= 5 and self.t_gray_strong == 0:
                if self.t_gray >= 0.5:
                    if self.df.loc[self.nf - 10:self.nf - 1, "ai_long"].mean() >= 1.5 and self.df.loc[self.nf - 10:self.nf - 1, "ai"].mean() >= 0.9:
                        self.bns_check_3 = 1.5
                        if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1, "cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                            self.d_OMain = 4
                            self.cover_ordered = 1
                            self.cover_in_prc = price
                            self.cover_in_nf = self.nf
                            self.cover_in_time = now.minute
                            self.type = "cover_b_gray"
                            # self.sys_force_out = 0
                            # self.last_cover_prc = price
                            self.cover_out_prc = 0
                            self.cover_by_opt = 0
                            # self.type = self.cover_type
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            if (self.bns_triple == -1 and self.df.at[self.nf - 2, "bns_triple"] != -1) or self.df.loc[self.nf - 15:self.nf - 1, "bns_triple"].mean() == -1:
                if self.sum_peak <= 2 and self.df.loc[self.nf - 15:self.nf - 1, "bns_triple"].mean() == -1:
                    if self.cover_signal_2 <= -1 or self.df.loc[self.nf - 15:self.nf - 1, "cover_signal_2"].mean() <= 0:
                        if self.triple_last_last <= 0 and self.t_gray < 0.5:
                            if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit].count() == 0:
                                # if self.type != "s-d-out-prc_out_g3":
                                self.bns_check_3 = -1
                                self.bns_check_6 = -1
                                if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1, "cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                                    self.d_OMain = -4
                                    self.cover_ordered = -1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_s_triple"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type, self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            if self.df_gray_strong[self.df_gray_strong == 2].count() >= 5 and self.t_gray_strong == 0:
                if self.t_gray <= -0.5:
                    if self.df.loc[self.nf - 10:self.nf - 1, "ai"].mean() <= 0.1 and self.ai < 0.2:
                        self.bns_check_3 = -1.5
                        if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1, "cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                            self.d_OMain = -4
                            self.cover_ordered = -1
                            self.cover_in_prc = price
                            self.cover_in_nf = self.nf
                            self.cover_in_time = now.minute
                            self.type = "cover_s_gray"
                            # self.sys_force_out = 0
                            # self.last_cover_prc = price
                            self.cover_out_prc = 0
                            self.cover_by_opt = 0
                            # self.type = self.cover_type
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                          self.type, self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # bns_check3 (bns_check == 3)
            # if self.df_bns_check[self.df_bns_check >= 3].count() >= 10 and self.df.loc[self.nf - 10:self.nf - 1,"bns_check"].mean() == 0:
            if self.cvol_m_peak[self.cvol_m_peak >= 3].count() >= 5 and self.df.loc[self.nf - 10:self.nf - 1,"std_std_prc_cvol_m_peak"].mean() <= 2:
                if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.loc[self.nf - 35:self.nf - 6, "std_prc_cvol_m"].mean() * 1.02:
                    if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() >= self.df.loc[self.nf - 100:self.nf - 20,"ai"].mean():
                        if self.ai < 0.2: #self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.8 and
                            if self.cover_signal_2 != -1:# and self.test_signal != 2:
                                if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                                    if 1==1 and self.type != "b-d-out_peaklimit" and self.cover_ordered != 1:
                                        self.d_OMain = 4
                                        self.cover_ordered = 1
                                        self.cover_in_prc = price
                                        self.cover_in_nf = self.nf
                                        self.cover_in_time = now.minute
                                        self.type = "cover_b_bns_check3"
                                        # self.sys_force_out = 0
                                        # self.last_cover_prc = price
                                        self.cover_out_prc = 0
                                        self.cover_by_opt = 0
                                        # self.type = self.cover_type
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                      self.type,self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # if self.df_bns_check[self.df_bns_check <= -3].count() >= 10 and self.df.loc[self.nf - 10:self.nf - 1,"bns_check"].mean() == 0:
            if self.cvol_m_peak[self.cvol_m_peak >= 3].count() >= 5 and self.df.loc[self.nf - 10:self.nf - 1,"std_std_prc_cvol_m_peak"].mean() <= 2:
                if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 35:self.nf - 6,"std_prc_cvol_m"].mean() * 1.02:
                    if self.df.loc[self.nf - 20:self.nf - 1, "ai"].mean() <= self.df.loc[self.nf - 100:self.nf - 20,"ai"].mean():
                        if self.ai_long > 0.5: #self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.8 and
                            if self.cover_signal_2 != 1:# and self.test_signal != -2:
                                if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                                    if 1 == 1 and self.type != "b-d-out_peaklimit" and self.cover_ordered != -1:
                                        self.d_OMain = -4
                                        self.cover_ordered = -1
                                        self.cover_in_prc = price
                                        self.cover_in_nf = self.nf
                                        self.cover_in_time = now.minute
                                        self.type = "cover_s_bns_check3"
                                        # self.sys_force_out = 0
                                        # self.last_cover_prc = price
                                        self.cover_out_prc = 0
                                        self.cover_by_opt = 0
                                        # self.type = self.cover_type
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                      self.type, self.profit, self.profit_opt, self.mode, self.ave_prc,prc_o1)

            # # bns_check_2
            # if 1==0 and self.df_bns_check2s[self.df_bns_check2s <= -1.7].count() >= 1 and self.df.at[self.nf - 1, "bns_check_2"] > -1.7:
            #     if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 35:self.nf - 6,"std_prc_cvol_m"].mean() * 1.02:
            #         if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
            #             if self.type != "s-d-out_peaklimit" and self.cover_ordered != -1:
            #                 self.d_OMain = -4
            #                 self.cover_ordered = -1
            #                 self.cover_in_prc = price
            #                 self.cover_in_nf = self.nf
            #                 self.cover_in_time = now.minute
            #                 self.type = "cover_s_bns_check_2"
            #                 # self.sys_force_out = 0
            #                 # self.last_cover_prc = price
            #                 self.cover_out_prc = 0
            #                 self.cover_by_opt = 0
            #                 # self.type = self.cover_tyfpe
            #                 self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
            #                               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # if 1==0 and self.df_bns_check2s[self.df_bns_check2s >= 1.7].count() >= 1 and self.df.at[self.nf - 1, "bns_check_2"] < 1.7:
            #     if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.loc[self.nf - 35:self.nf - 6, "std_prc_cvol_m"].mean() * 0.98:
            #         if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
            #             if self.type != "b-d-out_peaklimit" and self.cover_ordered != 1:
            #                 self.d_OMain = 4
            #                 self.cover_ordered = 1
            #                 self.cover_in_prc = price
            #                 self.cover_in_nf = self.nf
            #                 self.cover_in_time = now.minute
            #                 self.type = "cover_b_bns_check_2"
            #                 # self.sys_force_out = 0
            #                 # self.last_cover_prc = price
            #                 self.cover_out_prc = 0
            #                 self.cover_by_opt = 0
            #                 # self.type = self.cover_type
            #                 self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
            #                               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # std_prc_cvol_m cover (hold)
            # if self.df_std_prc[self.df_std_prc > self.std_prc_cvol_m_limit * 1.5].count() >= 1 and self.ai_long > 0.25:
            #     if self.df.loc[self.nf - 5:self.nf - 1,"std_prc_cvol_m"].mean() < self.std_prc_cvol_m_limit * 0.1:
            #         if self.test_signal > -2:
            #             if (self.which_market == 3 and now.hour != 9) or (self.which_market == 4 and now.hour != 22) or self.which_market == 1:
            #                 self.bns_check_2 = -0.5
            #                 if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
            #                     if self.type != "s-d-out-peaklimit" and self.cover_ordered != -1:
            #                         self.d_OMain = -4
            #                         self.cover_ordered = -1
            #                         self.cover_in_prc = price
            #                         self.cover_in_nf = self.nf
            #                         self.cover_in_time = now.minute
            #                         self.type = "cover_s_std_prc"
            #                         # self.sys_force_out = 0
            #                         # self.last_cover_prc = price
            #                         self.cover_out_prc = 0
            #                         self.cover_by_opt = 0
            #                         # self.type = self.cover_type
            #                         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
            #                                       self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # if self.df_std_prc[self.df_std_prc < self.std_prc_cvol_m_limit * -1.5].count() >= 1 and self.ai < 0.8:
            #     if self.df.loc[self.nf - 5:self.nf - 1,"std_prc_cvol_m"].mean() > self.std_prc_cvol_m_limit * -0.1:
            #         if self.test_signal < 2:
            #             if (self.which_market == 3 and now.hour != 9) or (self.which_market == 4 and now.hour != 22) or self.which_market == 1:
            #                 self.bns_check_2 = 0.5
            #                 if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (
            #                         self.stat_in_org == "111" or self.chkForb == 1):
            #                     if self.type != "b-d-out-peaklimit" and self.cover_ordered != 1:
            #                         self.d_OMain = 4
            #                         self.cover_ordered = 1
            #                         self.cover_in_prc = price
            #                         self.cover_in_nf = self.nf
            #                         self.cover_in_time = now.minute
            #                         self.type = "cover_b_std_prc"
            #                         # self.sys_force_out = 0
            #                         # self.last_cover_prc = price
            #                         self.cover_out_prc = 0
            #                         self.cover_by_opt = 0
            #                         # self.type = self.cover_type
            #                         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
            #                                       self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            self.df.at[self.nf, "bns_check_2"] = self.bns_check_2
            self.df.at[self.nf, "bns_check_3"] = self.bns_check_3
            self.df.at[self.nf, "bns_check_5"] = self.bns_check_5
            self.df.at[self.nf, "bns_check_6"] = self.bns_check_6
            #
            # # (bns2 + bns_check) s (hold)
            # if self.type != "conv 0" and self.type != "conv 1" and self.type != "conv 2" and self.type != "conv 3":
            #     if self.OrgMain == "b" and self.df.loc[self.nf - 100:self.nf - 1,
            #                                                                 "exed_qty"].mean() >= 1:
            #         if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() <= self.df.loc[self.nf - 35:self.nf - 6,"std_prc_cvol_m"].mean() * 1.02:
            #             if self.dOrgMain_new_bns2 <= 0 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] > 0:
            #                 if ((self.bns_check < 0 and self.bns_check > -2) or self.bns_check == -3):
            #                     if self.df_std_std[self.df_std_std >= self.std_std_prc_cvol_m_limit].count() == 0:
            #                         if self.stat_in_org == "111" and self.cover_ordered != -1:  # and price <= self.inp - self.tick * 2:
            #                             if self.bns_check <= 0:
            #                                 self.d_OMain = -4
            #                                 self.cover_ordered = -1
            #                                 self.cover_in_prc = price
            #                                 self.cover_in_nf = self.nf
            #                                 self.cover_in_time = now.minute
            #                                 self.type = "cover_s_bns2_check"
            #                                 # self.last_cover_prc = price
            #                                 self.cover_out_prc = 0
            #                                 self.cover_by_opt = 0
            #                                 # self.type = self.cover_type
            #                                 self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,self.type,
            #                                               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
            #                                 self.last_o = float(lblShoga1v)

                # # (bns2 + bns_check) b (hold)
                # if self.OrgMain == "s" and self.df.loc[self.nf - 3:self.nf - 1,
                #                                                             "exed_qty"].mean() >= 1:
                #     if self.df.loc[self.nf - 15:self.nf - 1, "std_prc_cvol_m"].mean() >= self.df.loc[self.nf - 35:self.nf - 6, "std_prc_cvol_m"].mean() * 0.98:
                #         if self.dOrgMain_new_bns2 >= 0 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] < 0:
                #             if ((self.bns_check > 0 and self.bns_check < 2) or self.bns_check == 3):
                #                 if self.df_std_std[self.df_std_std >= self.std_std_prc_cvol_m_limit].count() == 0:
                #                     if self.stat_in_org == "111" and self.cover_ordered != 1:  # and price >= self.inp - self.tick * 2:
                #                         if self.bns_check >= 0:
                #                             self.d_OMain = 4
                #                             self.cover_ordered = 1
                #                             self.cover_in_prc = price
                #                             self.cover_in_nf = self.nf
                #                             self.cover_in_time = now.minute
                #                             self.type = "cover_b_bns2_check"
                #                             # self.last_cover_prc = price
                #                             self.cover_out_prc = 0
                #                             self.cover_by_opt = 0
                #                             # self.type = self.cover_type
                #                             self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,self.type,
                #                                           self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                #                             self.last_o = float(lblShoga1v)

            # (short-in)
            # if 1 == 0:  # block after analysis -> recalled(only when 's')
            #     if self.OrgMain == "b" and self.df.loc[self.nf - 100:self.nf - 1,
            #                                "cover_ordered"].mean() == 0 and self.df.loc[self.nf - 3:self.nf - 1,
            #                                                                 "exed_qty"].mean() >= 1:
            #         if self.nf >= 501 and self.ai <= 0.1 and self.ai_long <= self.ai_long_low and self.df.loc[
            #                                                                                       self.nf - 200:self.nf - 1,
            #                                                                                       "ai"].mean() <= 0.01:
            #             if self.prc_s_peak >= 0:  # and self.df_conv.mean() <= 0.05:
            #                 if self.df.loc[self.nf - 50:self.nf - 1, "prc_s_peak"].mean() >= 0:
            #                     if self.stat_in_org == "111" and self.cover_ordered != -1:
            #                         if self.bns_check <= 0:
            #                             self.d_OMain = -4
            #                             self.cover_ordered = -1
            #                             self.cover_in_prc = price
            #                             self.cover_in_nf = self.nf
            #                             self.cover_in_time = now.minute
            #                             self.type = "cover_s_short"
            #                             # self.sys_force_out = 0
            #                             # self.last_cover_prc = price
            #                             self.cover_out_prc = 0
            #                             self.cover_by_opt = 0
            #                             # self.type = self.cover_type
            #                             self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
            #                                           self.type,
            #                                           self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # (short-in => long-in)
            # if 1 == 0:  # block after analysis
            #     if self.OrgMain != "n" and self.df.loc[self.nf - 3:self.nf - 1,
            #                                                                 "exed_qty"].mean() >= 1:
            #         if self.nf >= 501 and self.ai >= 0.5 and self.ai_long >= 1.8 and self.df.loc[
            #                                                                          self.nf - 220:self.nf - 20,
            #                                                                          "ai"].mean() <= 0.05:
            #             if self.prc_s_peak >= 0:  # and self.df_conv.mean() <= 0.05:
            #                 if (self.df.at[self.nf - 2, "test_signal"] != 3 and self.df.at[
            #                     self.nf - 1, "test_signal"] == 3):
            #                     if self.df.loc[self.nf - 50:self.nf - 1, "prc_s_peak"].mean() >= 0:
            #                         if self.stat_in_org == "111" and self.cover_ordered != -1:
            #                             if price <= self.last_o - self.tick * 15 * m_add:
            #                                 if self.bns_check >= 0:
            #                                     self.d_OMain = 4
            #                                     self.cover_ordered = 1
            #                                     self.cover_in_prc = price
            #                                     self.cover_in_nf = self.nf
            #                                     self.cover_in_time = now.minute
            #                                     self.type = "cover_b3-long"
            #                                     # self.sys_force_out = 0
            #                                     # self.last_cover_prc = price
            #                                     self.cover_out_prc = 0
            #                                     self.cover_by_opt = 0
            #                                     self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
            #                                                   self.OrgMain,
            #                                                   self.type,
            #                                                   self.profit, self.profit_opt, self.mode, self.ave_prc,
            #                                                   prc_o1)
            #                                     self.last_o = float(lblShoga1v)

            # # std_std_prc_cvol_m_peak_s
            # if 1==0 and self.OrgMain == "b" and self.df.loc[self.nf - 3:self.nf - 1,
            #                                                             "exed_qty"].mean() >= 1:
            #     if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 1 and self.std_prc_peak_1000 == -2:
            #         if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 1.1].count() == 1:
            #             if self.nf >= 1000 and self.ai <= 0.1 and self.dOrgMain_new_bns2 < 0:
            #                 if (self.stat_in_org == "111" and self.cover_ordered != -1) or self.chkForb == 1:
            #                     if self.bns_check <= 0 and self.df_bns_check_s[self.df_bns_check_s == 3].count() == 0:
            #                         if self.type == "s-d-out_peaklimit" and self.cover_ordered != -1:
            #                             if self.df.loc[self.nf - 50:self.nf - 1, "cover_ordered"].mean() == 0:
            #                                 self.d_OMain = -4
            #                                 self.cover_ordered = -1
            #                                 self.cover_in_prc = price
            #                                 self.cover_in_nf = self.nf
            #                                 self.cover_in_time = now.minute
            #                                 self.type = "cover_s_std_std"
            #                                 # self.sys_force_out = 0
            #                                 # self.last_cover_prc = price
            #                                 self.cover_out_prc = 0
            #                                 self.cover_by_opt = 0
            #                                 # self.type = self.cover_type
            #                                 self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
            #                                               self.type,
            #                                               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # # std_std_prc_cvol_m_peak_b
            # if 1==0 and self.OrgMain == "s" and self.df.loc[self.nf - 3:self.nf - 1,
            #                                                             "exed_qty"].mean() >= 1:
            #     if self.std_std_prc_cvol_m > self.std_std_prc_cvol_m_limit * 1 and self.std_prc_peak_1000 == 2:
            #         if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 1.1].count() == 0:
            #             if self.nf >= 1000 and self.ai_long >= 1.8 and self.dOrgMain_new_bns2 > 0:
            #                 if (self.stat_in_org == "111" and self.cover_ordered != 1) or self.chkForb == 1:
            #                     if self.bns_check >= 0 and self.df_bns_check_s[self.df_bns_check_s == -3].count() == 0:
            #                         if self.df.loc[self.nf - 50:self.nf - 1, "cover_ordered"].mean() == 0:
            #                             if self.type != "b-d-out_peaklimit" and self.cover_ordered != 1:
            #                                 self.d_OMain = 4
            #                                 self.cover_ordered = 1
            #                                 self.cover_in_prc = price
            #                                 self.cover_in_nf = self.nf
            #                                 self.cover_in_time = now.minute
            #                                 self.type = "cover_b_std_std"
            #                                 # self.sys_force_out = 0
            #                                 # self.last_cover_prc = price
            #                                 self.cover_out_prc = 0
            #                                 self.cover_by_opt = 0
            #                                 # self.type = self.cover_type
            #                                 self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
            #                                               self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # bit only
            if 1==0 and self.which_market == 1:
                if self.df.at[self.nf - 2, "test_signal"] == 0 and self.df.at[self.nf - 1, "test_signal"] >= 2:
                    if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                        if 1 == 1 and self.type != "b-d-out_peaklimit" and self.cover_ordered != -1:
                            self.d_OMain = -4
                            self.cover_ordered = -1
                            self.cover_in_prc = price
                            self.cover_in_nf = self.nf
                            self.cover_in_time = now.minute
                            self.type = "cover_s_bit_test"
                            # self.sys_force_out = 0
                            # self.last_cover_prc = price
                            self.cover_out_prc = 0
                            self.cover_by_opt = 0
                            # self.type = self.cover_type
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                          self.type, self.profit, self.profit_opt, self.mode, self.ave_prc,prc_o1)

                if self.df.at[self.nf - 2, "test_signal"] == 0 and self.df.at[self.nf - 1, "test_signal"] <= -2:
                    if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                        if 1 == 1 and self.type != "s-d-out_peaklimit" and self.cover_ordered != 1:
                            self.d_OMain = 4
                            self.cover_ordered = 1
                            self.cover_in_prc = price
                            self.cover_in_nf = self.nf
                            self.cover_in_time = now.minute
                            self.type = "cover_b_bit_test"
                            # self.sys_force_out = 0
                            # self.last_cover_prc = price
                            self.cover_out_prc = 0
                            self.cover_by_opt = 0
                            # self.type = self.cover_type
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                          self.type, self.profit, self.profit_opt, self.mode, self.ave_prc,prc_o1)

            # test_signal in (cover_s2, b2)
            if 1==1 and self.df_test_signal[self.df_test_signal <= -3].count() >= 1 and self.test_signal > -3:
                if self.t_gray >= 0:
                    if self.OrgMain == "s" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                        if self.triple_last >= 0 and self.cover_ordered != 1:
                            self.d_OMain = 4
                            self.cover_ordered = 1
                            self.cover_in_prc = price
                            self.cover_in_nf = self.nf
                            self.cover_in_time = now.minute
                            self.type = "cover_b_test_signal"
                            # self.sys_force_out = 0
                            # self.last_cover_prc = price
                            self.cover_out_prc = 0
                            self.cover_by_opt = 0
                            # self.type = self.cover_type
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            if 1==1 and self.df_test_signal[self.df_test_signal >= 3].count() >= 1 and self.test_signal < 3:
                if self.t_gray <= 0:
                    if self.OrgMain == "b" and self.df.loc[self.nf - 25:self.nf - 1,"cover_ordered"].mean() == 0 and (self.stat_in_org == "111" or self.chkForb == 1):
                        if self.triple_last <= 0 and self.cover_ordered != -1:
                            self.d_OMain = -4
                            self.cover_ordered = -1
                            self.cover_in_prc = price
                            self.cover_in_nf = self.nf
                            self.cover_in_time = now.minute
                            self.type = "cover_s_test_signal"
                            # self.sys_force_out = 0
                            # self.last_cover_prc = price
                            self.cover_out_prc = 0
                            self.cover_by_opt = 0
                            # self.type = self.cover_type
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # (in)
            if self.exed_qty >= 5 and self.df.loc[self.nf - 3:self.nf - 1,"exed_qty"].mean() >= 1 and self.nowprf < 0 and self.cvol_m >= self.cvol_m_limit:

                # cover with b in S state
                if self.OrgMain == "s" and (self.exed_qty == 1 or price >= self.cover_out_prc + self.tick * 3):
                    if self.cvol_m >= self.cvol_m_start * 0.5 and count_m >= self.count_m_start * 0.5:
                        if self.df.loc[self.nf - 20:self.nf - 10,"cvol_m"].mean() < self.df.loc[self.nf - 10:self.nf - 1,"cvol_m"].mean():  # or 1==1:
                            if (self.cover_by_opt == 1) or (self.df.at[self.nf - 1, "exed_qty"] <= self.max_e_qty):  # and self.prf_able == -1):  # @ s-mode ( 4, -4) self.cover_signal_2 == 1 or
                                if (self.ai >= 0.9 or self.ai_long >= 1.8) and self.now_trend != 0 and self.cover_ordered != 1:
                                    if self.df.loc[self.nf - 50: self.nf - 3, "test_signal"].mean() <= 0:
                                        if self.bns_check >= 0:
                                            self.d_OMain = 4
                                            self.cover_ordered = 1
                                            self.cover_in_prc = price
                                            self.cover_in_nf = self.nf
                                            self.cover_in_time = now.minute
                                            self.type = "cover_b1"
                                            # self.sys_force_out = 0
                                            # self.last_cover_prc = price
                                            self.cover_out_prc = 0
                                            self.cover_by_opt = 0
                                            # self.type = self.cover_type
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # if (self.df.at[self.nf-3, "test_signal"] == 0 and self.df.at[self.nf-2, "test_signal"] <= -2) or (self.df.at[self.nf-3, "test_signal"] == 3 and self.df.at[self.nf-2, "test_signal"] <= 2):
                        if (self.df.loc[self.nf - 100: self.nf - 2, "test_signal"].mean() == 2 and self.df.at[
                            self.nf - 1, "test_signal"] == 3):
                            if self.cover_ordered != 1:
                                # if self.df.loc[self.nf - 50: self.nf - 3, "test_signal"].mean() <= 0:
                                if self.bns_check >= 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_b2"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price795563c
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if (self.ai >= 0.9 and self.ai_long >= 1.8) and self.df.loc[self.nf - 10:self.nf - 1,
                                                                        "ai_spot"].mean() == 1:
                            if self.cvol_m >= self.cvol_m_start * 0.5:  # and cvol_m_mean>= self.cvol_m_limit: #self.df.loc[self.nf - 20:self.nf - 1, "d_OMain"].mean() == 0:
                                if self.df.loc[self.nf - 20:self.nf - 10, "cvol_m"].mean() < self.df.loc[
                                                                                             self.nf - 10:self.nf - 1,
                                                                                             "cvol_m"].mean():
                                    if self.cover_ordered != 1:
                                        if self.df.loc[self.nf - 50: self.nf - 3, "test_signal"].mean() <= 0:
                                            if self.bns_check >= 0:
                                                self.d_OMain = 4
                                                self.cover_ordered = 1
                                                self.cover_in_prc = price
                                                self.cover_in_nf = self.nf
                                                self.cover_in_time = now.minute
                                                self.type = "cover_b3"
                                                # self.sys_force_out = 0
                                                # self.last_cover_prc = price
                                                self.cover_out_prc = 0
                                                self.cover_by_opt = 0
                                                # self.type = self.cover_type
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                              self.OrgMain, self.type,
                                                              self.profit, self.profit_opt, self.mode, self.ave_prc,
                                                              prc_o1)

                        if self.ai_long >= 1.95 and self.df.loc[self.nf - 100:self.nf - 1, "ai_long"].mean() >= 1.95:
                            if self.cover_ordered != 1:
                                if self.bns_check >= 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_b7"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if self.nf >= 1000 and self.df.loc[self.nf - 200:self.nf - 1, "PXY_l_ave"].mean() >= 99:
                            if self.cover_ordered != 1:
                                if self.bns_check >= 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_b8"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 0 and ((self.now_trend - self.last_trend) > 0) and (
                                self.ai_spot == 1 and self.ai_long_spot == 2):
                            if self.test_signal != -3 and prc_s > 0 and self.dxy_decay > 0:
                                if self.df.loc[self.nf - 20:self.nf - 10, "cvol_m"].mean() < self.df.loc[
                                                                                             self.nf - 10:self.nf - 1,
                                                                                             "cvol_m"].mean():
                                    if 1 == 1:
                                        if self.df.loc[self.nf - 20:self.nf - 1,
                                           "cvol_m_sig"].mean() != 0 or self.now_trend == 1:
                                            if self.cvol_m >= self.cvol_m_limit:
                                                if self.cover_ordered != 1:
                                                    if self.bns_check >= 0:
                                                        self.d_OMain = 4
                                                        self.cover_ordered = 1
                                                        self.cover_in_prc = price
                                                        self.cover_in_nf = self.nf
                                                        self.cover_in_time = now.minute
                                                        self.type = "cover_b4"
                                                        # self.sys_force_out = 0
                                                        # self.last_cover_prc = price
                                                        self.cover_out_prc = 0
                                                        self.cover_by_opt = 0
                                                        # self.type = self.cover_type
                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                                      self.OrgMain, self.type,
                                                                      self.profit, self.profit_opt, self.mode,
                                                                      self.ave_prc, prc_o1)

                        # if self.std_prc_peak == -1 and (self.std_prc_peak > self.df.at[self.nf - 1, "std_prc_peak"]):
                        if self.df.loc[self.nf - 200: self.nf - 2, "std_prc_peak"].mean() == -1 and self.df.at[
                            self.nf - 1, "std_prc_peak"] == 0:
                            if self.cover_ordered != 1:
                                if self.bns_check >= 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_b5"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 0:
                            if (self.df.at[self.nf - 3, "cvol_s_peak"] == 2 and self.df.at[
                                self.nf - 2, "cvol_s_peak"] != 2):
                                if self.cover_ordered != 1:
                                    if self.bns_check >= 0:
                                        self.d_OMain = 4
                                        self.cover_ordered = 1
                                        self.cover_in_prc = price
                                        self.cover_in_nf = self.nf
                                        self.cover_in_time = now.minute
                                        self.type = "cover_b6"
                                        # self.sys_force_out = 0
                                        # self.last_cover_prc = price
                                        self.cover_out_prc = 0
                                        self.cover_by_opt = 0
                                        # self.type = self.cover_type
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                      self.type,
                                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # cover with s in B state
                if self.OrgMain == "b" and (self.exed_qty == 1 or price <= self.cover_out_prc - self.tick * 3):
                    if self.cvol_m >= self.cvol_m_start * 0.5 and count_m >= self.count_m_start * 0.5:
                        if self.df.loc[self.nf - 20:self.nf - 10,"cvol_m"].mean() < self.df.loc[self.nf - 10:self.nf - 1,"cvol_m"].mean():  # or 1==1:
                            if (self.cover_by_opt == -1) or (self.df.at[self.nf - 1, "exed_qty"] <= self.max_e_qty):
                                if (self.ai_spot == 0 or self.ai_long_spot == 0) and self.now_trend != 1 and self.cover_ordered != -1:
                                    if self.df.loc[self.nf - 50: self.nf - 3, "test_signal"].mean() >= 0:
                                        if self.bns_check <= 0:
                                            self.d_OMain = -4
                                            self.cover_ordered = -1
                                            self.cover_in_prc = price
                                            self.cover_in_nf = self.nf
                                            self.cover_in_time = now.minute
                                            self.type = "cover_s1"
                                            # self.sys_force_out = 0
                                            # self.last_cover_prc = price
                                            self.cover_out_prc = 0
                                            self.cover_by_opt = 0
                                            # self.type = self.cover_type
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # if (self.df.at[self.nf-3, "test_signal"] == 0 and self.df.at[self.nf-2, "test_signal"] >= 2) or (self.df.at[self.nf-3, "test_signal"] == -3 and self.df.at[self.nf-2, "test_signal"] >= -2):
                        if (self.df.loc[self.nf - 100: self.nf - 2, "test_signal"].mean() == -2 and self.df.at[
                            self.nf - 1, "test_signal"] == -3):
                            if self.cover_ordered != -1:
                                if self.bns_check <= 0:
                                    # if self.df.loc[self.nf - 50: self.nf - 3, "test_signal"].mean() >= 0:
                                    self.d_OMain = -4
                                    self.cover_ordered = -1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_s2"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if (self.ai <= 0.1 and self.ai_long <= self.ai_long_low) and self.df.loc[
                                                                                     self.nf - 10:self.nf - 1,
                                                                                     "ai_spot"].mean() == 0:
                            if self.cvol_m >= self.cvol_m_start * 0.5:
                                if self.df.loc[self.nf - 20:self.nf - 10, "cvol_m"].mean() < self.df.loc[
                                                                                             self.nf - 10:self.nf - 1,
                                                                                             "cvol_m"].mean():
                                    if self.cover_ordered != -1:
                                        if self.df.loc[self.nf - 50: self.nf - 3, "test_signal"].mean() >= 0:
                                            if self.bns_check <= 0:
                                                self.d_OMain = -4
                                                self.cover_ordered = -1
                                                self.cover_in_prc = price
                                                self.cover_in_nf = self.nf
                                                self.cover_in_time = now.minute
                                                self.type = "cover_s3"
                                                # self.sys_force_out = 0
                                                # self.last_cover_prc = price
                                                self.cover_out_prc = 0
                                                self.cover_by_opt = 0
                                                # self.type = self.cover_type
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                              self.OrgMain, self.type,
                                                              self.profit, self.profit_opt, self.mode, self.ave_prc,
                                                              prc_o1)

                        if self.ai == 0 and self.df.loc[self.nf - 200:self.nf - 1, "ai"].mean() <= 0.01:
                            if self.cover_ordered != -1:
                                if self.bns_check <= 0:
                                    self.d_OMain = -4
                                    self.cover_ordered = -1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_s7"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 0 and ((self.now_trend - self.last_trend) < 0) and (
                                self.ai_spot == 0):  # and self.ai_long_spot == 0):
                            if self.test_signal != 3 and prc_s < 0 and self.dxy_decay < 0:
                                if self.df.loc[self.nf - 20:self.nf - 10, "cvol_m"].mean() < self.df.loc[
                                                                                             self.nf - 10:self.nf - 1,
                                                                                             "cvol_m"].mean():
                                    if 1 == 1:
                                        if self.df.loc[self.nf - 20:self.nf - 1,
                                           "cvol_m_sig"].mean() != 0 or self.now_trend == 0:
                                            if self.cvol_m >= self.cvol_m_limit:
                                                if self.cover_ordered != -1:
                                                    if self.bns_check <= 0:
                                                        self.d_OMain = -4
                                                        self.cover_ordered = -1
                                                        self.cover_in_prc = price
                                                        self.cover_in_nf = self.nf
                                                        self.cover_in_time = now.minute
                                                        self.type = "cover_s4"
                                                        # self.sys_force_out = 0
                                                        # self.last_cover_prc = price
                                                        self.cover_out_prc = 0
                                                        self.cover_by_opt = 0
                                                        # self.type = self.cover_type
                                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty,
                                                                      self.OrgMain, self.type,
                                                                      self.profit, self.profit_opt, self.mode,
                                                                      self.ave_prc, prc_o1)

                        # if self.std_prc_peak == 1 and (self.std_prc_peak < self.df.at[self.nf - 1, "std_prc_peak"]):
                        if self.df.loc[self.nf - 200: self.nf - 2, "std_prc_peak"].mean() == 1 and self.df.at[
                            self.nf - 1, "std_prc_peak"] == 0:
                            if self.cover_ordered != -1:
                                if self.bns_check <= 0:
                                    self.d_OMain = -4
                                    self.cover_ordered = -1
                                    self.cover_in_prc = price
                                    self.cover_in_nf = self.nf
                                    self.cover_in_time = now.minute
                                    self.type = "cover_s5"
                                    # self.sys_force_out = 0
                                    # self.last_cover_prc = price
                                    self.cover_out_prc = 0
                                    self.cover_by_opt = 0
                                    # self.type = self.cover_type
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 0:
                            if (self.df.at[self.nf - 3, "cvol_s_peak"] == 2 and self.df.at[
                                self.nf - 2, "cvol_s_peak"] != 2):
                                if self.test_signal != 2:
                                    if self.cover_ordered != -1:
                                        if self.bns_check <= 0:
                                            self.d_OMain = -4
                                            self.cover_ordered = -1
                                            self.cover_in_prc = price
                                            self.cover_in_nf = self.nf
                                            self.cover_in_time = now.minute
                                            self.type = "cover_s6"
                                            # self.sys_force_out = 0
                                            # self.last_cover_prc = price
                                            self.cover_out_prc = 0
                                            self.cover_by_opt = 0
                                            # self.type = self.cover_type
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # test_signal_trade

                # test_sig_df = self.df.loc[self.nf - 500: self.nf - 2, "test_signal"]

                if self.exed_qty >= 1 and self.exed_qty <= self.max_e_qty - 1:

                    if self.OrgMain == "b":
                        # 통상적인 spot b
                        if test_sig_df[test_sig_df >= 2].count() >= 400 and self.df.at[
                            self.nf - 2, "test_signal"] != 0 and self.df.at[self.nf - 1, "test_signal"] == 0:
                            self.d_OMain = 5
                            self.exed_qty += 1
                            self.type = "spot b"
                            self.last_o = float(lblShoga1v)
                            self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(lblShoga1v)) / (self.exed_qty)
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # 통상적인 spot s
                        if test_sig_df[test_sig_df <= -2].count() >= 400 and self.df.at[
                            self.nf - 2, "test_signal"] != 0 and self.df.at[self.nf - 1, "test_signal"] == 0:
                            self.d_OMain = -5
                            self.exed_qty -= 1
                            self.type = "spot s"
                            self.last_o = float(lblShoga1v)
                            self.ave_prc = (self.ave_prc * (self.exed_qty + 1) - float(lblShoga1v)) / (self.exed_qty)
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.OrgMain == "s":
                        # 통상적인 spot s
                        if test_sig_df[test_sig_df <= -2].count() >= 400 and self.df.at[
                            self.nf - 2, "test_signal"] != 0 and self.df.at[self.nf - 1, "test_signal"] == 0:
                            self.d_OMain = 5
                            self.exed_qty += 1
                            self.type = "spot s"
                            self.last_o = float(lblBhoga1v)
                            self.ave_prc = (self.ave_prc * (self.exed_qty - 1) + float(lblShoga1v)) / (self.exed_qty)
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # 통상적인 spot b
                        if test_sig_df[test_sig_df >= 2].count() >= 400 and self.df.at[
                            self.nf - 2, "test_signal"] != 0 and self.df.at[self.nf - 1, "test_signal"] == 0:
                            self.d_OMain = -5
                            self.exed_qty -= 1
                            self.type = "spot b"
                            self.last_o = float(lblBhoga1v)
                            self.ave_prc = (self.ave_prc * (self.exed_qty + 1) - float(lblShoga1v)) / (self.exed_qty)
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            # if self.cover_signal == 1:
            #     self.cover_signal = 0
            # if self.cover_signal == -1:
            #     self.cover_signal = 0

            # (out)
        if self.cover_ordered != 0 and (self.which_market != 3 or self.cover_order_exed != 0 or self.chkForb == 1 or self.acc_uninfied == 1):
            self.prf_cover = (self.cover_in_prc - price) / prc_std

            if self.df.loc[self.nf - 25: self.nf - 1, "cover_ordered"].mean() == -1:# and (self.which_market==3 or abs(self.prf_cover)>=1.5):

                self.prf_cover = (self.cover_in_prc - price) / prc_std

                # np2_out : (b_in) -> (s_out)
                if 1==0 and self.np1 == 1 and self.np2 == -1:
                    if self.df.at[self.nf - 84, "np1"] != 1 and self.df.at[self.nf - 83, "np1"] == 1:
                        if self.df.loc[self.nf - 80: self.nf - 1, "np1"].mean() == 1 and (self.prc_s_peak >= 2 or self.bns_check_3 >= 0.5):
                            if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-np2_out"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # NEW
                if 1 == 1:
                    # if 1==0 and self.bns_check > 0:
                    #     if (self.bns_check <= 0.9 or self.bns_check == 3) and (self.OrgMain == "b" or self.OrgMain == "n"): #self.prf_cover >= 0
                    #         self.d_OMain = 4
                    #         self.cover_ordered = 0
                    #         # self.cover_order_exed = 0
                    #         # self.last_cover_prc = 0
                    #         self.zx += self.cover_in_prc - price
                    #         self.cover_in_prc = 0
                    #         self.cover_in_nf = 0
                    #         self.cover_out_prc = price
                    #         self.type = "s-d-out-bns_check"
                    #         self.dyna_out_touch = 0
                    #         self.prf_cover = 0
                    #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                    #         self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # test_signal
                    if 1==1:
                        if self.df.at[self.nf - 2, "test_signal"] >= 3 and self.df.at[self.nf - 1, "test_signal"] < 3: #self.prf_cover >= 0
                            if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-test_sig"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # check_out
                    if 1==0:
                        if self.df_bns_check_s[self.df_bns_check_s >= 0.9].count() >= 1 and self.bns_check > 0.9 and self.bns_check_4 == 1:
                            if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out_check"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # gold_out
                    if 1==1:
                        if (self.rsi_last == 2 or self.pvol_last == 2) and self.nf > self.cover_in_nf + 50:
                            if self.prf_cover < 0:
                                if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out_gold"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                    self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # triple_out
                    if 1==1:
                        if abs(self.prf_cover) >= 2.5 and (self.cover_in_prc - price) <= -1.5:
                            if self.triple_last == 1 or self.triple_last > self.df.at[self.nf - 2, "triple_last"]:
                                if self.nf > self.cover_in_nf + 100 or self.df.loc[self.nf - 50: self.nf - 1, "triple_last"].mean() == 1:
                                    if self.df_triple[self.df_triple <= 0].count() >= 25 or self.prc_s_peak >= 1:
                                        if self.cover_signal_2 >= 0 or self.t_gray_strong >= 0:
                                            if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                                self.d_OMain = 4
                                                self.cover_ordered = 0
                                                # self.cover_order_exed = 0
                                                # self.last_cover_prc = 0
                                                self.profit_opt += self.cover_in_prc - price
                                                self.cover_in_prc = 0
                                                self.cover_in_nf = 0
                                                self.cover_out_prc = price
                                                self.type = "s-d-out_triple1"
                                                self.dyna_out_touch = 0
                                                self.prf_cover = 0
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                            if (self.triple_last == 1 and self.cover_signal_2 == 1) or self.t_gray_strong >= 0:
                                if self.df_triple[self.df_triple == 1].count() >= 50:
                                    if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                        self.d_OMain = 4
                                        self.cover_ordered = 0
                                        # self.cover_order_exed = 0
                                        # self.last_cover_prc = 0
                                        self.profit_opt += self.cover_in_prc - price
                                        self.cover_in_prc = 0
                                        self.cover_in_nf = 0
                                        self.cover_out_prc = price
                                        self.type = "s-d-out_triple2"
                                        self.dyna_out_touch = 0
                                        self.prf_cover = 0
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if self.triple_last >= 0 and self.pvol_last == 1: # and self.triple_last_last == -1:
                            if self.t_gray == 1 and self.t_gray_strong >= 0:
                                if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                    if self.which_market == 3 or abs(self.prf_cover) >= 2:
                                        self.d_OMain = 4
                                        self.cover_ordered = 0
                                        # self.cover_order_exed = 0
                                        # self.last_cover_prc = 0
                                        self.profit_opt += self.cover_in_prc - price
                                        self.cover_in_prc = 0
                                        self.cover_in_nf = 0
                                        self.cover_out_prc = price
                                        self.type = "s-d-out_gray"
                                        self.dyna_out_touch = 0
                                        self.prf_cover = 0
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                      self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if self.prf_cover < -2 and self.triple_last == 1:
                            if self.t_gray == 1 or self.t_gray_strong > 0:
                                if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out_gray_strong"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)


                    # dbns2_out
                    if 1==1:
                        if self.prf_cover >= 2.5:
                            if self.df.at[self.nf - 2, "dOrgMain_new_bns2"] < 2 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] >= 2: #self.prf_cover >= 0
                                if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-dbns2"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                    self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.nf > 1000 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.5:
                        if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 2].count() >= 1:
                            if self.df_std_prc[self.df_std_prc <= self.std_prc_cvol_m_limit * -1.5].count() >= 1:
                                if self.df_bns_check_ss[self.df_bns_check_ss >= 2].count() >= 1:
                                    if self.test_signal == 0 and self.cover_signal_2 != -1:
                                        if (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0: #self.prf_cover >= 0
                                            self.d_OMain = 4
                                            self.cover_ordered = 0
                                            # self.cover_order_exed = 0
                                            # self.last_cover_prc = 0 #
                                            self.profit_opt += self.cover_in_prc - price
                                            self.cover_in_prc = 0
                                            self.cover_in_nf = 0
                                            self.cover_out_prc = price
                                            self.type = "s-d-out-peaklimit"
                                            self.dyna_out_touch = 0
                                            self.prf_cover = 0
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                            self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if 1==1 and self.which_market != 1:
                        if (price > self.df.loc[self.cover_in_nf: self.nf - 1, "price"].min() + self.tick * 15 * m or self.prf_cover <= -2.5):
                            if self.prf_cover <= -3 or self.sum_peak <= 2:
                                if self.prc_s > 0 and self.std_prc > 0 and self.ai >= 0.1:
                                    if self.std_std_prc_cvol_m_peak >= 1:
                                        if price > self.df.loc[self.cover_in_nf: self.nf - 1,"price"].min() + self.tick * 8 * m:
                                            if self.df.loc[self.nf - 50: self.nf - 1, "cover_ordered"].mean() == -1:
                                                if self.cover_ordered != 0:
                                                    self.d_OMain = 4
                                                    self.cover_ordered = 0
                                                    # self.cover_order_exed = 0
                                                    # self.last_cover_prc = 0
                                                    self.profit_opt += self.cover_in_prc - price
                                                    self.cover_in_prc = 0
                                                    self.cover_in_nf = 0
                                                    self.cover_out_prc = price
                                                    self.type = "s-d-out-prc_out"
                                                    self.dyna_out_touch = 0
                                                    self.prf_cover = 0
                                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if 1==0 and self.which_market == 1:
                        if (price > self.df.loc[self.cover_in_nf: self.nf - 1, "price"].min() + self.tick * 15 * m or self.prf_cover <= -2.5 * m):
                            if self.prf_cover <= -3 * m or self.sum_peak <= 2:
                                if self.prc_s > 0 and self.std_prc > 0 and self.ai >= 0.1:
                                    if self.std_std_prc_cvol_m_peak >= 1:
                                        if price > self.df.loc[self.cover_in_nf: self.nf - 1, "price"].min() + self.tick * 8 * m:
                                            if self.df.loc[self.nf - 50: self.nf - 1, "cover_ordered"].mean() == -1:
                                                if self.cover_ordered != 0:
                                                    self.d_OMain = 4
                                                    self.cover_ordered = 0
                                                    # self.cover_order_exed = 0
                                                    # self.last_cover_prc = 0
                                                    self.profit_opt += self.cover_in_prc - price
                                                    self.cover_in_prc = 0
                                                    self.cover_in_nf = 0
                                                    self.cover_out_prc = price
                                                    self.type = "s-d-out-prc_out"
                                                    self.dyna_out_touch = 0
                                                    self.prf_cover = 0
                                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if price < self.df.loc[self.cover_in_nf: self.nf - 1, "price"].max() - self.tick * 10 * m or self.prf_cover >= 1.5:
                        if self.prc_s > 0 and self.std_prc > 0 and self.ai > 0.1:
                            if self.std_std_prc_cvol_m_peak >= 1:
                                if self.cover_ordered != 0 and self.which_market == 3:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-prc_out_g1"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if price < self.df.loc[self.cover_in_nf: self.nf - 1, "price"].min() - self.tick * 5 * m or self.prf_cover >= 1.5:
                        if (self.prf_cover >= 3) or (self.df.at[self.nf - 1, "sum_peak"] <= -3 and self.sum_peak > -3):
                            if self.cover_ordered != 0:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-prc_out_g2"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if 1==1 and self.which_market != 1:
                        if 1 == 0 and self.sum_peak >= -2: #self.sum_peak == -3 or
                            if self.df_sum_peak[self.df_sum_peak <= -4].count() >= 1:
                                if self.t_gray_strong >= 0 and self.triple_last >= 0 and self.type != "s-d-out-prc_out_g3":
                                # if self.sum_peak > self.df.at[self.nf - 1, "sum_peak"]:
                                #     if self.dOrgMain_new_bns2 > -2 and self.test_signal < 3: #self.gray_strong != -2 and
                                    if self.cover_ordered != 0:
                                        self.d_OMain = 4
                                        self.cover_ordered = 0
                                        # self.cover_order_exed = 0
                                        # self.last_cover_prc = 0
                                        self.profit_opt += self.cover_in_prc - price
                                        self.cover_in_prc = 0
                                        self.cover_in_nf = 0
                                        self.cover_out_prc = price
                                        self.type = "s-d-out-prc_out_g3"
                                        self.dyna_out_touch = 0
                                        self.prf_cover = 0
                                        self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                      self.type,self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    elif 1==0 and self.which_market == 1:
                        if 1 == 1 and self.sum_peak == -2: #self.sum_peak == -3 or
                            if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 0.5].count() >= 1:
                                if self.df_sum_peak[self.df_sum_peak <= -4].count() >= 1:
                                    if self.t_gray_strong >= 0 and self.triple_last >= 0:
                                    # if self.sum_peak > self.df.at[self.nf - 1, "sum_peak"]:
                                    #     if self.dOrgMain_new_bns2 > -2 and self.test_signal < 3:
                                        if self.cover_ordered != 0:
                                            self.d_OMain = 4
                                            self.cover_ordered = 0
                                            # self.cover_order_exed = 0
                                            # self.last_cover_prc = 0
                                            self.profit_opt += self.cover_in_prc - price
                                            self.cover_in_prc = 0
                                            self.cover_in_nf = 0
                                            self.cover_out_prc = price
                                            self.type = "s-d-out-prc_out_g3"
                                            self.dyna_out_touch = 0
                                            self.prf_cover = 0
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                          self.type,self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if 1 == 0 and self.dOrgMain_new_bns2 > -3 and self.sum_peak > -3:
                        if self.df_bns2[self.df_bns2 <= -3].count() >= 1:
                            if self.sum_peak > self.df.at[self.nf - 1, "sum_peak"]:
                                if self.cover_ordered != 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-prc_out_g4"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.cover_signal_2 > self.df.at[self.nf - 1, "cover_signal_2"] and self.last_cover_signal2 == -1:
                        if self.df.loc[self.nf - 30:self.nf - 1, "cover_signal_2"].mean() == 0:
                            if (self.OrgMain == "b" or self.OrgMain == "n"): #self.prf_cover >= 0
                                if self.cover_ordered != 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-cov_sig_2"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                    self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # if self.which_market == 1 and self.d_OMain == 4:
                #     if self.test_signal >= 2 or self.sum_peak <= -2:
                #         self.d_OMain == 0

                if self.which_market != 1 and self.cover_ordered == 1 and self.d_OMain == -4:
                    if self.dOrgMain_new_bns2 <= -3 or self.sum_peak <= -3:
                        self.d_OMain = 0
                        self.peak_block = 1

                if self.peak_block == 1:
                    if self.dOrgMain_new_bns2 > -3 and self.sum_peak > -3:
                        if self.cover_ordered != 0:
                            self.d_OMain = 4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += self.cover_in_prc - price
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "s-d-out-peak_block"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                            self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            self.peak_block = 0

                # OLD
                if 1 == 1:
                    # out in b-mode
                    if self.prf_cover >= 0 and (self.OrgMain == "b" or self.OrgMain == "n") and self.cover_ordered != 0:

                        if 1 == 1 and self.type == "cover_s_std_std":
                            if (self.df.loc[self.nf - 25: self.nf - 5,
                                "dxy_decay"].mean() > 200) or self.std_prc_peak_1000 > self.df.at[
                                self.nf - 2, "std_prc_peak_1000"]:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-std_std"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if (self.df.at[self.nf - 3, "test_signal"] != 3 and self.df.at[
                            self.nf - 2, "test_signal"] == 3):  # or (self.df.at[self.nf - 2, "test_signal"] <= -2):
                            # if price > self.last_cover_prc + prc_std*1:
                            if self.d_OMain != 3:
                                self.dyna_out_touch = 1
                            elif self.d_OMain == 3:
                                self.d_OMain = 5

                        if self.cover_ordered == -1 and (
                                self.df.loc[self.nf - 10: self.nf - 5, "test_signal"].mean() > 2 and self.df.at[
                            self.nf - 4, "test_signal"] <= 2):
                            if self.df.loc[self.nf - 30: self.nf - 1, "cover_ordered"].mean() == -1:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-peak"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if self.cover_ordered == -1 and (self.df.at[self.nf - 3, "sum_peak"] == 0 and self.df.at[
                            self.nf - 2, "sum_peak"] >= 3) and self.exed_qty >= 1:
                            self.d_OMain = 4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += self.cover_in_prc - price
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "s-d-out-s2"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # (잘 안맞음)
                        if 1 == 0 and self.cover_ordered == -1 and self.cover_in_prc != 0 and (
                                self.now_trend - self.last_trend) > 0 or self.now_trend == 1 and self.cover_in_prc - price >= prc_std * 0.4 * m_add:
                            if self.df.loc[self.nf - 20:self.nf - 1, "cvol_m_sig"].mean() != 0 and self.exed_qty >= 2:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-s3"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 1 and self.cover_ordered == -1 and self.cover_in_prc != 0:
                            if self.ai > 0.1 and self.df.at[self.nf - 3, "cvol_s_peak"] == 2 and self.df.at[
                                self.nf - 2, "cvol_s_peak"] != 2 and self.exed_qty >= 1 and self.cover_in_prc - price >= prc_std * 0.4 * m_add:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-s4"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if self.cover_ordered == -1 and self.cover_in_prc != 0 and self.prf_cover >= 0 and self.d_OMain == 5:  # and self.df.at[self.nf - 1, "test_signal"] != 3:
                            self.d_OMain = 4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += self.cover_in_prc - price
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "s-d-out-3"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 0 and self.cover_ordered == -1 and (self.force_on == 1 and self.force == 1):
                            if self.ai == 1 or self.prc_s > 0:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-force"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # if self.cover_ordered == -1 and self.nf > 50:# and self.cover_ordered == 0:# and self.sys_force_out != 1:
                        #     if self.df.at[self.nf - 2, "cover_ordered"] != 0:# and self.OrgMain == "n":

                        if 1 == 1 and self.cover_ordered == -1 and self.exed_qty <= 2:  # and abs(self.prf_cover) >= 0.1:
                            if (self.df.at[self.nf - 2, "std_prc_peak"] < -1 and self.df.at[
                                self.nf - 1, "std_prc_peak"] >= -1):  # or self.std_prc_peak == 2:#self.dyna_out_touch == 1 and
                                if self.std_std_prc_cvol_m_peak >= 1:
                                    # if std_prc > 0.5:
                                    # if 1==1 or (prc_s > 0 and self.dxy_decay > 0) or (self.ai_spot == 1 and self.ai_long_spot == 2):# and self.exed_qty >= 1:
                                    #     if self.df.loc[self.nf - 20:self.nf - 10, "cvol_m"].mean() < self.df.loc[self.nf - 10:self.nf - 1,"cvol_m"].mean():
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-s1"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # when s_covered in series
                        if 1 == 1:
                            dfs = self.df.loc[self.nf - 100:self.nf - 1, "d_OMain"]
                            if self.cover_ordered == -1 and dfs[dfs == 4].count() >= 1:
                                if self.df.loc[self.nf - 100:self.nf - 20, "ai"].mean() < 0.1 and self.df.loc[
                                                                                                  self.nf - 20:self.nf - 1,
                                                                                                  "ai"].mean() >= 0.8:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-ss"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

            if self.df.loc[self.nf - 25: self.nf - 1, "cover_ordered"].mean() == -1:
                if self.OrgMain == "b" or self.OrgMain == "n":

                    # big-out
                    if self.cover_in_prc != 0:
                        self.prf_cover = (self.cover_in_prc - price) / prc_std

                    if self.cover_ordered == -1 and self.cover_in_prc != 0 and (self.prf_cover >= 3.5 or (
                            self.prf_cover >= 2 and abs(prc_s) >= self.prc_s_limit * 0.9)) and self.exed_qty <= 5:
                        if (self.df_sum_peak[self.df_sum_peak <= -3].count() >= 1 and self.sum_peak > -3) or self.sum_peak == -4 or self.t_gray == 1:
                            if self.test_signal != 2:
                                if (self.ai_spot == 1 or self.ai_long_spot == 2):
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-b"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # in cover profit state - out
                    if self.cover_ordered == -1 and self.cover_in_prc != 0:
                        df_test_sig = self.df.loc[self.nf - 300:self.nf - 1, "test_signal"]
                        if (df_test_sig[
                                df_test_sig == 3].count() >= 4 and self.prf_cover>=2) and self.exed_qty >= 2:
                            if self.ai >= 0.9 and self.ai_long >= 1.8:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-cp"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # in minus-qty out
                    if self.cover_ordered == -1 and self.cover_in_prc != 0:
                        if self.df.at[self.nf - 2, "exed_qty"] > self.df.at[
                            self.nf - 1, "exed_qty"] and self.exed_qty >= 1:
                            if self.ai >= 0.9 and self.ai_long >= 1.8:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-minus"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # term-out
                    if self.cover_ordered == -1:  # and self.chkForb != 1:
                        if self.df.loc[self.nf - 10: self.nf - 2, "d_OMain"].mean() != 1 and self.OrgMain == "n":
                            if self.df.at[self.nf - 1, "d_OMain"] != 4:
                                self.d_OMain = 4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "s-d-out-n"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                try:
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    bot1.sendMessage(chat_id="322233222", text="s-d-out-n")
                                except:
                                    pass

                    if self.df.loc[self.nf - 25: self.nf - 1, "cover_signal_2"].mean() == 1:
                        # if self.df.loc[self.nf - 5: self.nf - 1, "cover_signal_2"].mean() <= -0.5:
                        if self.prf_cover <= -2.5:
                            if self.test_signal <= -2 or self.std_std_prc_cvol_m_peak >= 2:
                                if self.cover_ordered != 0:
                                    self.d_OMain = 4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += self.cover_in_prc - price
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "s-d-out-cov_sig2"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # time-out
                    now = datetime.now()
                    if self.cover_ordered == -1 and (now.hour == 15 and now.minute == 31):
                        self.d_OMain = 4
                        self.cover_ordered = 0
                        # self.cover_order_exed = 0
                        # self.last_cover_prc = 0
                        self.profit_opt += self.cover_in_prc - price
                        self.cover_in_prc = 0
                        self.cover_in_nf = 0
                        self.cover_out_prc = price
                        self.type = "s-d-out-t-n"
                        self.dyna_out_touch = 0
                        self.prf_cover = 0
                        try:
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            bot1.sendMessage(chat_id="322233222", text="s-d-out-t-n")
                        except:
                            pass
                # if self.OrgMain == "n":
                #     if self.which_market == 3 and self.cover_signal != -2:
                #         if 1 == 1:
                #             try:
                #                 now = datetime.now()
                #                 account_sid = "AC3e42fb9a1717412229fa66d1182f5891"
                #                 auth_token = "808f66d57f585c4c16212ba52ecfdf13"
                #                 client = Client(account_sid, auth_token)
                #                 client.messages.create(
                #                     to="+821033165109",
                #                     from_="+18587710343",
                #                     body=str(demo) + "Error! at " + str(now.hour) + ':' + str(now.minute) + ':' + str(
                #                         now.second) + 'COVER not released ')
                #             except:
                #                 print("Sending MSG ERROR!!")

                # if self.OrgMain == "n":
                #     if self.which_market == 3:
                #         if self.cvol_m <= 0.03:
                #             if ema_520 != -2 or ema_520_prc_std != 2:
                #                 self.d_OMain = 4
                #                 self.cover_ordered = 0
                #                 self.cover_order_exed = 0
                #                 # self.last_cover_prc = 0
                #                 self.type = "s-d-out"
                #                 self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                #                               self.profit, self.profit_opt, self.mode,  self.ave_prc, prc_o1)
                #     if self.which_market != 3:
                #         self.d_OMain = 4
                #         self.cover_ordered = 0
                #         self.cover_order_exed = 0
                #         # self.last_cover_prc = 0
                #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                #                       self.profit, self.profit_opt, self.mode,  self.ave_prc, prc_o1)

            if self.df.loc[self.nf - 25: self.nf - 1, "cover_ordered"].mean() == 1:# and (self.which_market==3 or abs(self.prf_cover)>=1.5):

                self.prf_cover = (price - self.cover_in_prc) / prc_std

                # np2_out : (s_in) -> (b_out)
                if self.np1 == 1 and self.np2 == -1:
                    if self.df.at[self.nf - 14, "np2"] != -1 and self.df.at[self.nf - 13, "np2"] == -1:
                        if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out-np2_out"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                # NEW
                if 1 == 1:
                    if 1==0 and self.bns_check < 0:
                        if (self.bns_check <= -0.9 or self.bns_check == -3) and (self.OrgMain == "s" or self.OrgMain == "n"):
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out-bns_check"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # test_signal
                if 1 == 1:
                    if self.df.at[self.nf - 2, "test_signal"] <= -3 and self.df.at[self.nf - 1, "test_signal"] > -3:
                        if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out-test_sig"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # check_out
                if 1==0:
                    if self.df_bns_check_s[self.df_bns_check_s <= -0.9].count() >= 1 and self.bns_check < -0.9 and self.bns_check_4 == -1:
                        if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out_check"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # gold_out
                if 1==1:
                    if (self.rsi_last == -2 or self.pvol_last == -2) and self.nf > self.cover_in_nf + 50:
                        if self.prf_cover < 0:
                            if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out_gold"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # triple_out
                if 1==1:
                    if abs(self.prf_cover) >= 2.5 and (self.cover_in_prc - price) >= 1.5:
                        if self.triple_last == -1 or self.triple_last < self.df.at[self.nf - 2, "triple_last"]:
                            if self.nf > self.cover_in_nf + 100 or self.df.loc[self.nf - 50: self.nf - 1, "triple_last"].mean() == -1:
                                if self.df_triple[self.df_triple >= 0].count() >= 25 or self.prc_s_peak <= -1:
                                    if self.cover_signal_2 <= 0 or self.t_gray_strong <= 0:
                                        if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                                            self.d_OMain = -4
                                            self.cover_ordered = 0
                                            # self.cover_order_exed = 0
                                            # self.last_cover_prc = 0
                                            self.profit_opt += price - self.cover_in_prc
                                            self.cover_in_prc = 0
                                            self.cover_in_nf = 0
                                            self.cover_out_prc = price
                                            self.type = "b-d-out_triple1"
                                            self.dyna_out_touch = 0
                                            self.prf_cover = 0
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if (self.triple_last == -1 and self.cover_signal_2 == -1) or self.t_gray_strong <= 0:
                            if self.df_triple[self.df_triple == -1].count() >= 50:
                                if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out_triple2"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.triple_last <= 0 and self.pvol_last == -1: # and self.triple_last_last == 1):
                        if self.t_gray == -1 and self.t_gray_strong <= 0:
                            if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                                if self.which_market == 3 or abs(self.prf_cover) >= 2:
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out_gray"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.prf_cover < -2 and self.triple_last == -1:
                        if self.t_gray == -1 or self.t_gray_strong < 0:
                            if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out_gray_strong"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # dbns2_out
                if 1==1:
                    if self.prf_cover >= 2.5:
                        if self.df.at[self.nf - 2, "dOrgMain_new_bns2"] > -2 and self.df.at[self.nf - 1, "dOrgMain_new_bns2"] <= -2: #self.prf_cover >= 0
                            if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.zx += self.cover_in_prc - price
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-dbns2"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.nf > 1000 and self.std_std_prc_cvol_m < self.std_std_prc_cvol_m_limit * 0.6:
                        if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 2].count() >= 1:
                            if self.df_std_prc[self.df_std_prc >= self.std_prc_cvol_m_limit * 1].count() >= 1:
                                if self.df_bns_check_ss[self.df_bns_check_ss <= -2].count() >= 1:
                                    if self.test_signal == 0 and self.cover_signal_2 != 1:
                                        if (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0: #self.prf_cover >= 0
                                            self.d_OMain = -4
                                            self.cover_ordered = 0
                                            # self.cover_order_exed = 0
                                            # self.last_cover_prc = 0
                                            self.profit_opt += price - self.cover_in_prc
                                            self.cover_in_prc = 0
                                            self.cover_in_nf = 0
                                            self.cover_out_prc = price
                                            self.type = "b-d-out-peaklimit"
                                            self.dyna_out_touch = 0
                                            self.prf_cover = 0
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if 1==1 and  self.which_market != 1:
                        if (price < self.df.loc[self.cover_in_nf: self.nf - 1, "price"].max() - self.tick * 15 * m or self.prf_cover <= -2.5):
                            if self.prf_cover <= -3 or self.sum_peak <= -2:
                                if self.prc_s < 0 and self.std_prc < 0 and self.ai < 0.2:
                                    if self.std_std_prc_cvol_m_peak >= 1:
                                        if price < self.df.loc[self.cover_in_nf: self.nf - 1, "price"].max() - self.tick * 8 * m:
                                            if self.df.loc[self.nf - 50: self.nf - 1, "cover_ordered"].mean() == 1:
                                                if self.cover_ordered != 0:
                                                    self.d_OMain = -4
                                                    self.cover_ordered = 0
                                                    # self.cover_order_exed = 0
                                                    # self.last_cover_prc = 0
                                                    self.profit_opt += price - self.cover_in_prc
                                                    self.cover_in_prc = 0
                                                    self.cover_in_nf = 0
                                                    self.cover_out_prc = price
                                                    self.type = "b-d-out-prc_out"
                                                    self.dyna_out_touch = 0
                                                    self.prf_cover = 0
                                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if 1==0 and self.which_market == 1:
                        if (price < self.df.loc[self.cover_in_nf: self.nf - 1, "price"].max() - self.tick * 15 * m or self.prf_cover <= -2.5 * m):
                            if self.prf_cover <= -3 * m or self.sum_peak <= -2:
                                if self.prc_s < 0 and self.std_prc < 0 and self.ai < 0.2:
                                    if self.std_std_prc_cvol_m_peak >= 1:
                                        if price < self.df.loc[self.cover_in_nf: self.nf - 1, "price"].max() - self.tick * 8 * m:
                                            if self.df.loc[self.nf - 50: self.nf - 1, "cover_ordered"].mean() == 1:
                                                if self.cover_ordered != 0:
                                                    self.d_OMain = -4
                                                    self.cover_ordered = 0
                                                    # self.cover_order_exed = 0
                                                    # self.last_cover_prc = 0
                                                    self.profit_opt += price - self.cover_in_prc
                                                    self.cover_in_prc = 0
                                                    self.cover_in_nf = 0
                                                    self.cover_out_prc = price
                                                    self.type = "b-d-out-prc_out"
                                                    self.dyna_out_touch = 0
                                                    self.prf_cover = 0
                                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                    if price > self.df.loc[self.cover_in_nf: self.nf - 1, "price"].min() + self.tick * 10 * m or self.prf_cover >= 1.5:
                        if self.prc_s < 0 and self.std_prc < 0 and self.ai < 0.1:
                            if self.std_std_prc_cvol_m_peak >= 1:
                                if self.cover_ordered != 0 and self.which_market == 3:
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out-prc_out_g1"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if price > self.df.loc[self.cover_in_nf: self.nf - 1, "price"].max() + self.tick * 5 * m or self.prf_cover >= 1.5:
                        if (self.prf_cover >= 3) or (self.df.at[self.nf - 1, "sum_peak"] >= 3 and self.sum_peak < 3):
                            if self.cover_ordered != 0:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-prc_out_g2"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # g3
                    if 1==1 and self.which_market != 1:
                        if 1 == 1 and self.sum_peak <= 2: #self.sum_peak == 3 or
                            if self.df_sum_peak[self.df_sum_peak >= 4].count() >= 1:
                                if self.t_gray_strong <= 0 and self.triple_last <= 0 and self.type != "b-d-out-prc_out_g3":
                                # if self.sum_peak < self.df.at[self.nf - 1, "sum_peak"]:
                                #     if self.dOrgMain_new_bns2 < 2 and self.test_signal > -3: #self.gray_strong != 2 and
                                        if self.cover_ordered != 0:
                                            self.d_OMain = -4
                                            self.cover_ordered = 0
                                            # self.cover_order_exed = 0
                                            # self.last_cover_prc = 0
                                            self.profit_opt += price - self.cover_in_prc
                                            self.cover_in_prc = 0
                                            self.cover_in_nf = 0
                                            self.cover_out_prc = price
                                            self.type = "b-d-out-prc_out_g3"
                                            self.dyna_out_touch = 0
                                            self.prf_cover = 0
                                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    elif 1==0 and self.which_market == 1:
                        if 1 == 1 and self.sum_peak <= 2: #self.sum_peak == 3 or s
                            if self.df_std_std_s[self.df_std_std_s >= self.std_std_prc_cvol_m_limit * 0.5].count() >= 1:
                                if self.df_sum_peak[self.df_sum_peak >= 4].count() >= 1:
                                    if self.t_gray_strong <= 0 and self.triple_last <= 0:
                                    # if self.sum_peak < self.df.at[self.nf - 1, "sum_peak"]:
                                    #     if self.dOrgMain_new_bns2 < 2 and self.test_signal > -3:
                                            if self.cover_ordered != 0:
                                                self.d_OMain = -4
                                                self.cover_ordered = 0
                                                # self.cover_order_exed = 0
                                                # self.last_cover_prc = 0
                                                self.profit_opt += price - self.cover_in_prc
                                                self.cover_in_prc = 0
                                                self.cover_in_nf = 0
                                                self.cover_out_prc = price
                                                self.type = "b-d-out-prc_out_g3"
                                                self.dyna_out_touch = 0
                                                self.prf_cover = 0
                                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if 1 == 0 and self.dOrgMain_new_bns2 < 3 and self.sum_peak < 3:
                        if self.df_bns2[self.df_bns2 >= 3].count() >= 1:
                            if self.sum_peak < self.df.at[self.nf - 1, "sum_peak"]:
                                if self.cover_ordered != 0:
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out-prc_out_g4"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.cover_signal_2 < self.df.at[self.nf - 1, "cover_signal_2"] and self.last_cover_signal2 == 1:
                        if self.df.loc[self.nf - 30:self.nf - 1, "cover_signal_2"].mean() == 0:
                            if (self.OrgMain == "s" or self.OrgMain == "n"):
                                if self.cover_ordered != 0:
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out-cov_sig_2"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                # if self.which_market == 1 and self.d_OMain == -4:
                #     if self.test_signal <= -2 or self.sum_peak >= 2:
                #         self.d_OMain == 0

                if self.which_market != 1 and self.cover_ordered == -1 and self.d_OMain == 4:
                    if self.dOrgMain_new_bns2 >= 3 or self.sum_peak >= 3:
                        self.d_OMain = 0
                        self.peak_block = -1

                if self.peak_block == -1:
                    if self.dOrgMain_new_bns2 < 3 and self.sum_peak < 3:
                        if self.cover_ordered != 0:
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out-peak_block"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                            self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            self.peak_block = 0

                # OLD
                if 1 == 1:

                    # out in s-mode
                    if self.prf_cover >= 0 and (self.OrgMain == "s" or self.OrgMain == "n") and self.cover_ordered != 0:

                        if 1 == 1 and self.type == "cover_b_std_std":
                            if self.df.loc[self.nf - 25: self.nf - 5,
                               "dxy_decay"].mean() < -200 or self.std_prc_peak_1000 < self.df.at[
                                self.nf - 2, "std_prc_peak_1000"]:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-std_std"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if (self.df.loc[self.nf - 3, "test_signal"] != -3 and self.df.at[
                            self.nf - 2, "test_signal"] == -3):  # or (self.df.at[self.nf - 2, "test_signal"] >= 2):
                            # if price < self.last_cover_prc - prc_std * 1:
                            if self.d_OMain != -3:
                                self.dyna_out_touch = -1
                            elif self.d_OMain == -3:
                                self.d_OMain = -5

                        if self.cover_ordered == 1 and (
                                self.df.loc[self.nf - 10: self.nf - 5, "test_signal"].mean() < -2 and self.df.at[
                            self.nf - 4, "test_signal"] >= -2):
                            if self.df.loc[self.nf - 30: self.nf - 1, "cover_ordered"].mean() == 1:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-peak"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # short-out
                        if 1==0 and self.cover_ordered == 1 and self.ai == 0 and self.df.loc[self.nf - 200:self.nf - 1,
                                                                        "ai"].mean() <= 0.01 and self.prc_s_peak < 0:
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out-short"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc,
                                          prc_o1)

                        if self.cover_ordered == 1 and self.cover_in_prc != 0 and (
                                self.df.at[self.nf - 3, "sum_peak"] == 0 and self.df.at[
                            self.nf - 2, "sum_peak"] <= -3) and self.exed_qty >= 1:
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out-s2"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # if self.prf_cover >= 0:
                        if self.cover_ordered == 1 and self.cover_in_prc != 0 and (
                                self.now_trend - self.last_trend) < 0 or self.now_trend == 0:
                            if self.df.loc[self.nf - 20:self.nf - 1,
                               "cvol_m_sig"].mean() != 0 and self.exed_qty >= 2 and price - self.cover_in_prc >= prc_std * 0.4 * m_add:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-s3"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 1 and self.cover_ordered == 1 and self.cover_in_prc != 0:
                            if self.ai_long < 1.8 and (self.df.at[self.nf - 3, "cvol_s_peak"] == 2 and self.df.at[
                                self.nf - 2, "cvol_s_peak"] != 2) and self.exed_qty >= 1 and price - self.cover_in_prc >= prc_std * 0.4 * m_add:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-s4"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # (잘 안맞음)
                        if 1 == 0 and self.cover_ordered == 1 and self.cover_in_prc != 0 and self.prf_cover >= 0 and self.d_OMain == -5:  # and self.df.at[self.nf - 1, "test_signal"] != -3:
                            self.d_OMain = -4
                            self.cover_ordered = 0
                            # self.cover_order_exed = 0
                            # self.last_cover_prc = 0
                            self.profit_opt += price - self.cover_in_prc
                            self.cover_in_prc = 0
                            self.cover_in_nf = 0
                            self.cover_out_prc = price
                            self.type = "b-d-out-3"
                            self.dyna_out_touch = 0
                            self.prf_cover = 0
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        if 1 == 0 and self.cover_ordered == 1 and (self.force_on == 1 and self.force == -1):
                            if self.ai == 0 or self.prc_s < 0:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-force"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # if self.cover_ordered == 1 and self.nf > 50:# and self.cover_ordered == 0: # and self.sys_force_out != 1:
                        #     if self.df.at[self.nf - 2, "cover_ordered"] != 0:# and self.OrgMain == "n":

                        if 1 == 1 and self.cover_ordered == 1 and self.exed_qty <= 2:  # and abs(self.prf_cover) >= 0.1:
                            if (self.df.at[self.nf - 2, "std_prc_peak"] > 1 and self.df.at[
                                self.nf - 1, "std_prc_peak"] <= 1):  # or self.std_prc_peak == -2:#self.dyna_out_touch == 1 and
                                if self.std_std_prc_cvol_m_peak >= 1:# if std_prc < -0.5:
                                    # if 1==1 or (prc_s < 0 and self.dxy_decay < 0) or (self.ai_spot == 0 or self.ai_long_spot == 0):# and self.exed_qty >= 1:
                                    #     if self.df.loc[self.nf - 20:self.nf - 10, "cvol_m"].mean() < self.df.loc[self.nf - 10:self.nf - 1,"cvol_m"].mean():
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out-s1"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                        # when b_covered in series
                        if 1 == 1:
                            dfs = self.df.loc[self.nf - 100:self.nf - 1, "d_OMain"]
                            if self.cover_ordered == 1 and dfs[dfs == -4].count() >= 1:
                                if self.df.loc[self.nf - 100:self.nf - 20, "ai_long"].mean() > 1.9 and self.df.loc[
                                                                                                       self.nf - 20:self.nf - 1,
                                                                                                       "ai_long"].mean() <= self.ai_long_low:
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out-bb"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc,
                                                  prc_o1)

            if self.df.loc[self.nf - 25: self.nf - 1, "cover_ordered"].mean() == 1:
                if self.OrgMain == "s" or self.OrgMain == "n":

                    # big-out
                    if self.cover_in_prc != 0:
                        self.prf_cover = (price - self.cover_in_prc) / prc_std

                    if self.cover_ordered == 1 and self.cover_in_prc != 0 and (self.prf_cover >= 3.5 or (
                            self.prf_cover >= 2 and abs(prc_s) >= self.prc_s_limit * 0.9)) and self.exed_qty <= 5:  # and std_prc < 0:
                        if (self.df_sum_peak[self.df_sum_peak >= 3].count() >= 1 and self.sum_peak < 3) or self.sum_peak == 4 or self.t_gray == -1:
                            if self.test_signal != -2:
                                if (self.ai_spot == 0 or self.ai_long_spot == 0):
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out-b"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # in cover profit state - out
                    if self.cover_ordered == 1 and self.cover_in_prc != 0:
                        df_test_sig = self.df.loc[self.nf - 300:self.nf - 1, "test_signal"]
                        if (df_test_sig[
                                df_test_sig == -3].count() >= 4 and abs(self.prf_cover)>=2) and self.exed_qty >= 2:
                            if self.ai <= 0.1:  # and self.ai_long <= 0.2:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-cp"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # in minus-qty out
                    if self.cover_ordered == 1 and self.cover_in_prc != 0:
                        if self.df.at[self.nf - 2, "exed_qty"] > self.df.at[
                            self.nf - 1, "exed_qty"] and self.exed_qty >= 1:
                            if self.ai <= 0.2:  # and self.ai_long <= 0.4:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-minus"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                              self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    if self.df.loc[self.nf - 25: self.nf - 1, "cover_signal_2"].mean() == -1:
                        # if self.df.loc[self.nf - 5: self.nf - 1, "cover_signal_2"].mean() <= -0.5:
                        if self.prf_cover <= -2.5:
                            if self.test_signal <= -2 or self.std_std_prc_cvol_m_peak >= 2:
                                if self.cover_ordered != 0:  # self.prf_cover >= 0
                                    self.d_OMain = -4
                                    self.cover_ordered = 0
                                    # self.cover_order_exed = 0
                                    # self.last_cover_prc = 0
                                    self.profit_opt += price - self.cover_in_prc
                                    self.cover_in_prc = 0
                                    self.cover_in_nf = 0
                                    self.cover_out_prc = price
                                    self.type = "b-d-out-cov_sig2"
                                    self.dyna_out_touch = 0
                                    self.prf_cover = 0
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)

                    # term-out
                    if self.cover_ordered == 1:  # and self.chkForb != 1:
                        if self.df.loc[self.nf - 10: self.nf - 2, "d_OMain"].mean() != -1 and self.OrgMain == "n":
                            if self.df.at[self.nf - 1, "d_OMain"] != -4:
                                self.d_OMain = -4
                                self.cover_ordered = 0
                                # self.cover_order_exed = 0
                                # self.last_cover_prc = 0
                                self.profit_opt += price - self.cover_in_prc
                                self.cover_in_prc = 0
                                self.cover_in_nf = 0
                                self.cover_out_prc = price
                                self.type = "b-d-out-n"
                                self.dyna_out_touch = 0
                                self.prf_cover = 0
                                try:
                                    self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                                                  self.type,
                                                  self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                                    bot1.sendMessage(chat_id="322233222", text="b-d-out-n")
                                except:
                                    pass

                        # time-out
                    now = datetime.now()
                    if self.cover_ordered == 1 and (now.hour == 15 and now.minute == 31):
                        self.d_OMain = -4
                        self.cover_ordered = 0
                        # self.cover_order_exed = 0
                        # self.last_cover_prc = 0
                        self.profit_opt += price - self.cover_in_prc
                        self.cover_in_prc = 0
                        self.cover_in_nf = 0
                        self.cover_out_prc = price
                        self.type = "b-d-out-t-n"
                        self.dyna_out_touch = 0
                        self.prf_cover = 0
                        try:
                            self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                                          self.profit, self.profit_opt, self.mode, self.ave_prc, prc_o1)
                            bot1.sendMessage(chat_id="322233222", text="b-d-out-t-n")
                        except:
                            pass

                # if self.OrgMain == "n":
                #     if self.which_market == 3 and self.cover_signal != 2:
                #         if 1 == 1:
                #             try:
                #                 now = datetime.now()
                #                 account_sid = "AC3e42fb9a1717412229fa66d1182f5891"
                #                 auth_token = "808f66d57f585c4c16212ba52ecfdf13"
                #                 client = Client(account_sid, auth_token)
                #                 client.messages.create(
                #                     to="+821033165109",
                #                     from_="+18587710343",
                #                     body=str(demo) + "Error! at " + str(now.hour) + ':' + str(now.minute) + ':' + str(
                #                         now.second) + 'COVER not released ')
                #             except:
                #                 print("Sending MSG ERROR!!")

                # if self.OrgMain == "n":
                #     if self.which_market == 3:
                #         if self.cvol_m <= 0.03:
                #             if ema_520 != 2 or ema_520_prc_std != 2:
                #                 self.d_OMain = -4
                #                 self.cover_ordered = 0
                #                 self.cover_order_exed = 0
                #                 # self.last_cover_prc = 0
                #                 self.type = "s-d-out"
                #                 self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain,
                #                               self.type,
                #                               self.profit, self.profit_opt, self.mode,  self.ave_prc, prc_o1)
                #     if self.which_market != 3:
                #         self.d_OMain = -4
                #         self.cover_ordered = 0
                #         self.cover_order_exed = 0
                #         # self.last_cover_prc = 0
                #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
                #                       self.profit, self.profit_opt, self.mode,  self.ave_prc, prc_o1l)


        elif self.dynamic_cover == 0:
            self.cover_signal = 0
            self.last_cover_prc = 0

        self.df.at[self.nf, "dyna_out_touch"] = self.dyna_out_touch
        self.df.at[self.nf, "cover_ordered"] = self.cover_ordered
        self.df.at[self.nf, "cover_in_prc"] = self.cover_in_prc
        self.df.at[self.nf, "cover_in_nf"] = self.cover_in_nf
        self.df.at[self.nf, "cover_in_time"] = self.cover_in_time
        self.df.at[self.nf, "profit_opt"] = self.profit_opt
        self.df.at[self.nf, "prf_cover"] = self.prf_cover
        self.df.at[self.nf, "np1"] = self.np1
        self.df.at[self.nf, "np2"] = self.np2

        # if self.df.at[self.nf, "type"] == "adjust_out" and self.df.at[self.nf-1, "type"] == "adjust_out":
        #     if self.df.at[self.nf-1, "OrgMain"] == "b":
        #         self.d_OMain = 5
        #     if self.df.at[self.nf-1, "OrgMain"] == "s":
        #         self.d_OMain = -5

        ##########################
        # B_Fut
        # if 1==0 and self.which_market == 1:
        #     if self.now_trend > 0 and self.last_trend == 0:
        #         self.d_OMain = 10
        #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
        #                       self.profit, self.profit_opt, self.mode,  self.ave_prc, prc_o1)
        #     if self.now_trend < 1 and self.last_trend == 1:
        #         self.d_OMain = -10
        #         self.hist_rec(self.nf, lblShoga1v, lblBhoga1v, self.exed_qty, self.OrgMain, self.type,
        #                       self.profit, self.profit_opt, self.mode,  self.ave_prc, prc_o1)

        ##########################

        self.df.at[self.nf, "d_OMain"] = self.d_OMain  # * 7
        self.df.at[self.nf, "OrgMain"] = self.OrgMain
        # self.df.at[self.nf, "AvePrc"] = self.AvePrc
        self.df.at[self.nf, "last_o"] = self.last_o
        # print "OrgMain", self.OrgMain
        self.df.at[self.nf, "cover_order_exed"] = self.cover_order_exed
        self.df.at[self.nf, "stat_in_org"] = self.stat_in_org
        self.df.at[self.nf, "stat_out_org"] = self.stat_out_org
        self.df.at[self.nf, "chkForb"] = self.chkForb
        self.df.at[self.nf, "time"] = datetime.now().strftime("%m-%d-%H-%M-%S")
        self.df.at[self.nf, "block_doubl_cover_out"] = self.block_doubl_cover_out
        self.df.at[self.nf, "peak_block"] = self.peak_block

        # sending hist msg to e-mail
        # self.min_now = int(datetime.now().strftime("%m-%d-%H-%M-%S")[9:11])
        # self.hr_now = datetime.now().strftime("%m-%d-%H-%M-%S")[6:8]
        # print("now.minute % 5: ",now.minute % 5)
        # print("self.msg_sent: ",self.msg_sent)
        if ((self.which_market ==3 and now.hour >=9 and now.minute >= 30) or self.which_market !=3) and now.minute % 5 == 0 and self.msg_sent == 0:
            # self.msg_sent = 1
            # if self.msg_sent == 1:
            self.msg = MIMEMultipart()
            html = """\
            <html>
              <head></head>
              <body>
                {0}
              </body>
            </html>
            """.format(self.hist.to_html())
            part1 = MIMEText(html, 'html')
            self.msg.attach(part1)

            python_ver = 2

            if 1 == 0:
                if python_ver == 2:
                    self.msg['Subject'] = Header(
                        'hist_' + str(self.which_market) + ' [' + self.demo + '] ' + ' @' + str(
                            now.hour) + ' , prc=' + str(price), 'utf-8')
                    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                    server.login('ens.exchanger', 'mango7982')
                    server.sendmail('ens.exchanger@gmail.com', 'ens.exchanger@gmail.com', self.msg.as_string())
                    server.quit()
                if python_ver == 3:
                    self.msg['Subject'] = Header(
                        'hist_' + str(self.which_market) + self.demo + ' @' + self.hr_now + ' , ' + str(price), 'utf-8')
                    self.msg['From'] = 'ens.exchanger@gmail.com'
                    self.msg['To'] = 'ens.exchanger@gmail.com'
                    with smtplib.SMTP_SSL('smtp.gmail.com') as smtp:
                        smtp.login('ens.exchanger', 'mango7982')
                        smtp.send_message(self.msg)

            # telegram
            if 1 == 1:# or (self.which_market == 3 and now.hour <= 11) or self.OrgMain != "n":
                c = ["time", "bns", "qty", "prc_o", "type"]  # , "profit", "prf_opt"]
                d = ["bns", "qty", "time", "profit", "prf_opt"]  # ["nf", "time", "bns", "prc", "type", "ap", "qty"]
                e = ["bns", "qty", "prc", "ap"]
                if 1 == 1:
                    if len(self.hist) >= 2:
                        hist = self.hist.loc[1:]
                    else:
                        hist = self.hist
                    # text1 = hist[c].to_markdown()  # .to_markdown() #((tablefmt="grid"))
                    # text2 = hist[d].to_markdown()
                    text1 = hist[c].to_string()
                    text2 = hist[d].to_string()
                    text3 = 'bns: ' + self.OrgMain + '  qty: ' + str(self.exed_qty) + '  now: ' + str(
                        price) + ',  ap: ' + str(round(self.ave_prc, 2))

                    # text1 = tabulate(hist[c], headers='keys', tablefmt='psql')
                    # text1 = tabulate(hist[d], headers='keys', tablefmt='psql')
                    # prf_main = hist.iloc[-1, -3]
                    # prf_option = hist.iloc[-1, -2]
                    # text2 = "prf: " + str(round(prf_main)) + ",   Opt: " + str(round(prf_option, 2))
                if 1==1 and bot_alive == 1: ##
                    # if (self.auto_cover == 1 and now.second <= 20) or (self.auto_cover == 2 and now.second > 40):
                    bot1.sendMessage(chat_id="322233222", text=text1)
                    bot1.sendMessage(chat_id="322233222", text=text2)
                    bot1.sendMessage(chat_id="322233222", text=text3)

                    self.msg_sent = 1

        if now.minute % 5 != 0:
            self.msg_sent = 0

        self.nf += 1

        if self.nf > 10:
            print(self.df.loc[self.nf - 3:self.nf - 1,
                  ['count', 'cvol_m', 'cvol_t', 'cvol_s', 'cvol_c', 'dxy_200_medi', 'OrgMain', 'inp']])  #
            print('[[cover]]: %s  Prc: %s  /Ave_prc: %s  /prf_hit: %s  /prf_able: %s' % (
            self.cover_signal, price, self.ave_prc, self.prf_hit, self.prf_able))
            print('OrdNo', self.OrgOrdNo)
            print('-----------')
            print(self.hist.loc[self.no - 5:self.nf - 1,
                  ["nf", "time", "bns", 'prc', 'prc_o', 'qty', 'profit', 'type', 'mode']])
            print('-----------')
            # print("chkCoverSig_2: ", self.chkCoverSig_2)

        elap = time.time() - t_start
        self.df.at[self.nf, "elap"] = elap
        if self.OrgOrdNo != []:
            self.df.at[self.nf, "OrdNo"] = str(self.OrgOrdNo)  # [-1])
        elif self.OrgOrdNo == []:
            self.df.at[self.nf, "OrdNo"] = str(0)
        if self.OrgOrdNo_Cov != []:
            self.df.at[self.nf, "OrdNo_Cov"] = str(self.OrgOrdNo_Cov)  # [-1])
        elif self.OrgOrdNo_Cov == []:
            self.df.at[self.nf, "OrdNo_Cov"] = str(0)
        # print("elap %0.5f" % (elap))
        print('-----------')

        print("prf %0.5f" % (self.profit))
        print("prf_opt %0.5f" % (self.profit_opt))

        print('-----------')

        ######################
        return self.d_OMain
        ######################

    def hist_rec(self, nf, lblShoga1v, lblBhoga1v, exed_qty, OrgMain, type, profit, prf_opt, mode, ave_prc, prc_o1):
        global bot1, bot_alive

        self.no += 1
        self.hist.at[self.no, "no"] = self.no
        self.hist.at[self.no, "nf"] = nf
        self.hist.at[self.no, "qty"] = exed_qty
        self.hist.at[self.no, "ap"] = round(ave_prc, 2)
        self.hist.at[self.no, "prc_o"] = prc_o1
        self.hist.at[self.no, "time"] = datetime.now().strftime("%H-%M")
        self.hist.at[self.no, "bns"] = OrgMain
        self.hist.at[self.no, "type"] = type
        self.hist.at[self.no, "profit"] = round(profit, 2)
        self.hist.at[self.no, "prf_opt"] = round(prf_opt, 2)
        self.hist.at[self.no, "mode"] = mode
        if OrgMain == "b":
            self.hist.at[self.no, "prc"] = float(lblShoga1v)
        elif OrgMain == "s":
            self.hist.at[self.no, "prc"] = float(lblBhoga1v)
        elif OrgMain == "n":
            if self.hist.at[self.no - 2, "bns"] == "b":
                self.hist.at[self.no, "prc"] = float(lblBhoga1v)
                # self.no += 1
            elif self.hist.at[self.no - 2, "bns"] == "s":
                self.hist.at[self.no, "prc"] = float(lblShoga1v)
                # self.no += 1

        # self.hist=self.hist[self.hist["qty"]>=1000]
        # print(df)

        # telegram
        now = datetime.now()
        if 1==0:
            if (self.which_market == 3 and now.hour <= 15) or self.OrgMain != "n":
                c = ["time", "bns", "prc", "type"]  # , "profit", "prf_opt"]
                d = ["bns", "qty", "ap", "profit", "prf_opt"]  # ["nf", "time", "bns", "prc", "type", "ap", "qty"]
                if 1 == 1:
                    if len(self.hist) >= 2:
                        hist = self.hist.loc[1:]
                    else:
                        hist = self.hist
                    # text1 = hist[c].to_markdown()  # .to_markdown() #((tablefmt="grid"))
                    # text2 = hist[d].to_markdown()
                    text1 = hist[c].to_string()
                    text2 = hist[d].to_string()
                    # text1 = tabulate(hist[c], headers='keys', tablefmt='psql')
                    # text1 = tabulate(hist[d], headers='keys', tablefmt='psql')
                    # prf_main = hist.iloc[-1, -3]
                    # prf_option = hist.iloc[-1, -2]
                    # text2 = "prf: " + str(round(prf_main)) + ",   Opt: " + str(round(prf_option, 2))
                if bot_alive == 1:
                    bot1.sendMessage(chat_id="322233222", text=text1)
                    bot1.sendMessage(chat_id="322233222",
                                     text=text2 + '\n now_prc: ' + str(prc_o1) + ',  ap: ' + str(round(self.ave_prc, 2)))

    def btnSave_Clicked(self):
        # df.to_sql("Main_DB", con, if_exists='replace', index=True) #, index_label=None, chunksize=None, dtype=None)
        ts = datetime.now().strftime("%m-%d-%H-%M")
        if self.which_market == 1 or self.which_market == 2:  # Bit
            filename = "EDB_2__%s.csv" % (ts)
        elif self.which_market == 3:  # kospi
            if self.demo == "demo":
                filename = "[demo] EDB_3__%s.csv" % (ts)
            if self.demo == "real":
                filename = "[real] EDB_3__%s.csv" % (ts)
        elif self.which_market == 4:  # e-mini
            filename = "EDB_4__%s.csv" % (ts)
        elif self.which_market == 5:  # a50
            filename = "EDB_5__%s.csv" % (ts)
        elif self.which_market == 6:  # micro
            filename = "EDB_6__%s.csv" % (ts)
        filename_hist = "Hist_%s.csv" % (ts)
        self.df.to_csv('%s' % filename)  # + time.strftime("%m-%d") + '.csv')
        if self.which_market == 3:
            self.hist.to_csv('%s' % filename_hist)
        if self.which_market != 3:
            self.hist.to_csv('%s' % str(self.which_market) + "-" + filename_hist)

    def plot_print(df_sys, p6_1, p6_2, p6_3, p6_4):

        print('print clicked...NP')

        ax1 = plt.subplot(611)
        ax2 = ax1.twinx()
        ax3 = plt.subplot(612)
        ax4 = ax3.twinx()
        ax5 = plt.subplot(613)
        ax6 = ax5.twinx()
        ax7 = plt.subplot(614)
        ax8 = ax7.twinx()
        ax9 = plt.subplot(615)
        ax10 = ax9.twinx()
        ax11 = plt.subplot(616)
        ax12 = ax11.twinx()

        file_loaded = 0

        start_data_index = df_sys[df_sys.nf != 0].index[0]
        df_sys = df_sys.loc[start_data_index:]
        file_loaded += 1

        # Plot range
        plot_start = 1
        if file_loaded >= 1:
            plot_last = df_sys.shape[0] - 1

        print('NP..start', start_data_index)
        print('NP..last', plot_last)
        # plot

        # p1 = df_sys.loc[plot_start:plot_last, "price"]
        # ax1.plot(p1, 'r', label="price")
        # ax1.grid()

        old_new = 0

        try:

            x = np.arange(plot_start, plot_last + 1, 1)

            p1 = df_sys.loc[plot_start:plot_last, "price"]
            df_sys.loc[plot_start:plot_last, "ave_prc"][df_sys.loc[plot_start:plot_last, "ave_prc"] == 0] = np.nan
            # p1_1 = df_sys.loc[plot_start:plot_last, "ave_prc"][df_sys.loc[plot_start:plot_last, "ave_prc"]>0]
            p1_1 = df_sys.loc[plot_start:plot_last, "ave_prc"][df_sys.loc[plot_start:plot_last, "OrgMain"] == "b"]
            p1_2 = df_sys.loc[plot_start:plot_last, "ave_prc"][df_sys.loc[plot_start:plot_last, "OrgMain"] == "s"]
            # p1_0 = df_sys.loc[411:plot_last, "ema_20"]
            # p1_1 = df_sys.loc[411:plot_last, "ema_50"]
            # p1_2 = df_sys.loc[411:plot_last, "ema_200"]
            ax1.plot(p1, 'k', label="price")
            ax1.plot(p1_1, 'r', label="ave_prc", linewidth=3)
            ax1.plot(p1_2, 'b', label="ave_prc", linewidth=3)
            # ax1.plot(p1_1, 'k', label="std_std_prc")
            # ax1.plot(p1_0, 'y', label="ema_20")
            # ax1.plot(p1_1, 'g', label="ema_50")
            # ax1.plot(p1_2, 'b', label="ema_200")
            ax1.grid()
            ax1.legend(loc=2)

            try:
                p2_3 = df_sys.loc[411:plot_last, "ema_520"]
            except:
                p2_3 = df_sys.loc[411:plot_last, "ema_520_d"]
                print("no ema_520_std")
            p2_4 = df_sys.loc[411:plot_last, "ema_25"]
            # p1_4 = df_sys.loc[411:plot_last, "rsi"]
            # ax2.plot(p1_3, 'k', label="ema_520")
            # ax2.plot((p1_4-50)/50, 'r', label="rsi")
            p2_1 = df_sys.loc[plot_start:plot_last, "std_std_prc"]
            print(type(p2_1))
            df_sys.loc[plot_start:plot_last, "mode"][df_sys.loc[plot_start:plot_last, "mode"] < 0] = 0
            p2_2 = df_sys.loc[plot_start:plot_last, "mode_spot"]
            # p2_5 = df_sys.loc[plot_start:plot_last, "mode"]
            # ax2.fill_between(x, 0, p2_2, facecolor='yellow')
            # p2_2 = df_sys.loc[plot_start:plot_last, "prc_std"]
            ax2.plot(p2_1 / p2_1.max(), 'g', label="std_std_prc")
            ax2.plot([plot_start, plot_last], [0.2, 0.2], 'k-')
            ax2.plot(p2_2 * 1, 'k', label="mode_spot")
            # ax2.plot(p2_5*1, 'g', label="mode")
            ax2.fill_between(x, 0, p2_2 * 1, facecolor='yellow')

            if 1 == 0:
                try:
                    ax2.plot(p2_3, 'g', label="ema_520")
                except:
                    ax2.plot(p2_3, 'r', label="ema_520_d")
                    print("no ema_520_std")
            # ax2.plot(p2_4, 'b', label="ema_25")
            # ax2.plot(p2_2, 'b', label="prc_std")
            ax2.grid()
            ax2.legend(loc=4)

            # p2 = df_sys.loc[plot_start:plot_last, "profit"]
            # ax2.plot(p2, 'g', label="profit")
            # ax2.plot(p1_1, 'g', label="ema_50")
            # ax2.plot(p1_2, 'g', label="ema_200")
            # ax2.grid()
            # ax2.legend(loc=3)
            # #
            p3 = df_sys.loc[plot_start:plot_last, "d_OMain"]
            # p3_1 = df_sys.loc[plot_start:plot_last, "piox"]
            try:
                p3_2 = df_sys.loc[plot_start:plot_last, "ema_520_prc_std"]
            except:
                print("no ema_520_prc_std")
            # p3_4 = df_sys.loc[plot_start:plot_last, "ema_520"]
            # p3_3 = df_sys.loc[plot_start:plot_last, "cvol_c_ave"]
            # p3_4 = df_sys.loc[plot_start:plot_last, "cvol_c_medi"]
            p3_5 = df_sys.loc[100:plot_last, "exed_qty"]  # *5#.values[-1] #/10
            ax3.plot(p1, 'k', label="price")
            ax3.plot(p3 * 3, 'k', label="d_OMain")
            ax3.fill_between(x, 0, p3)
            try:
                ax3.plot(p3_2 * 5, 'r', label="ema_520_prc_std")
                # ax3.fill_between(x, 0, p3_2 * 5)
            except:
                print("no ema_520_prc_std")
            # ax3.plot(p3_4 * 3, 'g', label="ema_520")
            # ax3.plot(p3_3-10, 'y', label="cvol_c_ave")
            # ax3.plot(p3_4-10, 'r', label="cvol_c_medi")
            # ax3.plot(p3_1, 'g', label="piox")
            ax3.plot(p3_5, 'b', label="exed_qty")
            x_2 = np.arange(100, plot_last + 1, 1)
            ax3.fill_between(x_2, 0, p3_5)
            # ax3.plot(p3_3, 'r', label="prc_std")
            ax3.grid()
            ax3.legend(loc=2)

            p4 = df_sys.loc[plot_start:plot_last, "prc_avg"]
            p3_7 = df_sys.loc[1011:plot_last, "prc_avg"] + df_sys.loc[1011:plot_last, "prc_std"]
            p4_1 = df_sys.loc[plot_start:plot_last, "count_m_per_10"]
            # p3_8 = df_sys.loc[1011:plot_last, "prc_avg"] - df_sys.loc[1011:plot_last, "prc_std"]
            # ax4.plot(p4, 'k', label="prc_avg")
            # ax4.plot(p3_7, 'g', label="prc_std+")
            # ax4.plot(p3_8, 'b', label="prc_std-")
            # ax4.grid()
            # # ax4.legend(loc=4)
            # # p4 = df_sys.loc[plot_start:plot_last, "in_sig_profit"]
            # # ax4.plot(p4, 'g', label="in_sig_profit")
            # # ax4.legend(loc=3)
            # #
            # # p3 = df_sys.loc[plot_start:plot_last, "volume"]
            # # ax3.plot(p3, 'r', label="volume")
            # # ax3.legend(loc=2)
            # # p4 = df_sys.loc[plot_start:plot_last, "cvolume"]
            # ax4.plot(p4, 'g', label="cvolume")
            # ax4.legend(loc=3)
            #
            # p5 = df_sys.loc[plot_start:plot_last, "dxy_med_200_s"]
            # ax5.plot(p5, 'r', label="dxy_med_200_s")
            # ax5.legend(loc=2)
            # p6 = df_sys.loc[plot_start:plot_last, "dxy_20_medi"]
            # ax6.plot(p6, 'g', label="dxy_20_medi")
            ax6.plot(p4_1 / p4_1.max(), 'g', label="count_m_per_10")
            # ax6.legend(loc=3)
            # #

            p5 = df_sys.loc[plot_start:plot_last, "price"]
            p5_0 = df_sys.loc[plot_start:plot_last, p6_1]
            try:
                p5_7 = df_sys.loc[plot_start:plot_last, p6_2]
            except:
                print("no in_hit")
            try:
                p5_8 = df_sys.loc[plot_start:plot_last, p6_3]
            except:
                print("no add_touch")
            try:
                p5_9 = df_sys.loc[plot_start:plot_last, p6_4]
            except:
                print("no peak_touch")
            p5_10 = df_sys.loc[plot_start:plot_last, "cover_signal_c"]
            # p5_11 = df_sys.loc[plot_start:plot_last, "last_cover_prc"]
            # p5_1 = df_sys.loc[plot_start:plot_last, "prc_sig"]
            # p5_2 = df_sys.loc[plot_start:plot_last, "signal_set"]
            # try:
            #     p5_3 = df_sys.loc[plot_start:plot_last, "ema_520_count_m"]
            # except:
            #     print("no ema_520_count_m")
            # p5_4 = df_sys.loc[plot_start:plot_last, "ema_25_count_m"]
            p5_5 = df_sys.loc[plot_start:plot_last, "count_m_ave"]
            # p3_2 = df_sys.loc[100:plot_last, "prc_sig"]*5#.values[-1] #/10
            ax5.plot(p5, 'r', label="price")
            # try:
            #     ax6.plot(p5_3/p5_3.max(), 'g', label="ema_520_count_m")
            #     ax6.fill_between(x, 0, p5_3)
            # except:
            #     print("no ema_520_count_m")
            ax6.plot(p5_5 / p5_5.max(), 'b', label="count_m_ave")
            ax6.plot((p5_0 / p5_0.max() - 2), 'g', label=p6_1)
            try:
                ax6.plot(p5_7, 'g', label=p6_2)
            except:
                print("no in_hit")
            try:
                ax6.plot(p5_8, 'k', label=p6_3)
            except:
                print("no add_touch")
            try:
                ax6.plot(p5_9, 'y', label=p6_4)
            except:
                print("no peak_touch")
            ax6.plot(p5_10, 'r', label="cover_signal_c")
            # ax6.plot(p5_11, 'r', label="last_cover_prc")
            ax6.fill_between(x, 0, p5_10, facecolor='yellow')
            # ax6.plot(p5_4, 'b', label="count_m_25")
            # ax6.fill_between(x, 0, p5_2*-1, facecolor='red')
            # ax3.plot(p3_2, 'b', label="prc_sig")
            # ax3.plot(p3_3, 'r', label="prc_std")
            ax5.grid()
            ax5.legend(loc=2)
            ax6.legend(loc=4)

            # p5 = df_sys.loc[plot_start:plot_last, "cvol_m_cri_ave"]
            # p6 = df_sys.loc[plot_start:plot_last, "cvol_m"]
            # ax5.plot(p5, 'r', label="cvol_m_cri_ave")
            # ax5.plot(p6, 'g', label="cvol_m")
            # ax5.grid()
            # ax5.legend(loc=2)

            # ax6.grid()
            # ax6.legend(loc=3)

            ##############################
            if 1 == 1:
                x_1 = np.arange(501, plot_last + 1, 1)

                p8_1 = df_sys.loc[plot_start:plot_last, "prc_s"]
                p8_2 = df_sys.loc[501:plot_last, "prc_s_std"]
                p8_3 = df_sys.loc[501:plot_last, "prc_s_peak"]
                # p8_4 = df_sys.loc[501:plot_last, "cvol_t_decay"]
                # p8_4 = df_sys.loc[501:plot_last, "cvol_m_decay"]
                if old_new != 0:
                    p8_4 = df_sys.loc[501:plot_last, "dxy_decay"]
                    p8_4 = p8_4 / max(p8_4.max(), p8_4.min() * -1) * 1
                    ax8.plot(p8_4, 'r', label="dxy_decay")
                p8_5 = df_sys.loc[501:plot_last, "test_signal"]

                ax7.plot(p5, 'k', label="price")
                ax8.plot(2 * p8_1 / p8_1.max(), 'b', label="prc_s")
                # ax8.plot(p8_2/p8_2.max(), 'g', label="std+")
                # ax8.plot(p8_2*-1/p8_2.max(), 'b', label="std-")
                ax8.plot(p8_3, 'y', label="prc_s_peak")
                # ax8.plot( p8_4, 'k', label="cvol_t_decay")
                # ax8.plot( p8_4, 'k', label="cvol_m_decay")
                ax8.plot(p8_5, 'r', label="test_signal")

                ax8.fill_between(x_1, 0, p8_3)  # , facecolor='yellow')
                # ax8.fill_between(x_1, 0, p8_4, facecolor='red')

                # ax7.grid()
                ax8.grid()
                ax7.legend(loc=2)
                ax8.legend(loc=4)

            ##############################
            if 1 == 0:
                p7 = df_sys.loc[plot_start:plot_last, "dxy_200_medi"]
                p7_1 = df_sys.loc[plot_start:plot_last, "dxy_med_std"]
                p7_2 = df_sys.loc[plot_start:plot_last, "dxy_sig"]
                p7_3 = df_sys.loc[plot_start:plot_last, "dxy_sig_dmain"]

                ax7.plot(p7, 'r', label="dxy_200_medi")
                ax7.fill_between(x, 0, p7, facecolor='green')
                ax7.plot(p7_1, 'g', label="dxy_200_std+")
                ax7.plot(p7_1 * 1, 'b', label="dxy_200_std-")
                ax8.plot(p7_2 * 1, 'k', label="dxy_sig")
                ax8.plot(p7_3, 'r', label="dxy_sig_dmain")
                ax7.grid()
                ax8.grid()
                ax7.legend(loc=2)
                ax8.legend(loc=4)

            if 1 == 0:
                p7 = df_sys.loc[plot_start:plot_last, "dxy_20_medi"]
                p7_1 = df_sys.loc[plot_start:plot_last, "dxy_20_medi_s"]

                ax7.plot(p7, 'r', label="dxy_20_medi")
                ax7.fill_between(x, 0, p7, facecolor='green')
                ax8.plot(p7_1, 'b', label="dxy_20_medi_s")
                ax7.grid()
                ax8.grid()
                ax7.legend(loc=2)
                ax8.legend(loc=4)
            ##############################
            # ax8.plot(p8, 'g', label="dxy_med_std")
            # ax8.grid()
            # ax8.legend(loc=3)

            # p8 = df_sys.loc[plot_start:plot_last, "profit_band"]
            # ax8.plot(p8, 'g', label="profit_band")
            # ax8.legend(loc=3)
            #
            # # p9 = df_sys.loc[plot_start:plot_last, "dxy_med_200_s"]
            # # ax9.plot(p9, 'r', label="dxy_200_med_s")
            # # ax9.legend(loc=2)
            # p9 = df_sys.loc[plot_start:plot_last, "cvol_c_ave"]
            # ax9.plot(p9, 'r', label="cvol_c_ave")
            # ax9.grid()
            # ax9.legend(loc=2)

            # p9 = df_sys.loc[plot_start:plot_last, "mode"]
            try:
                p9_1 = df_sys.loc[max(100, plot_start):plot_last, "std_prc"]
            except:
                pass
            p9_2 = df_sys.loc[plot_start:plot_last, "std_prc_peak"]

            # p9_2 = df_sys.loc[plot_start:plot_last, "std_std_prc"]
            # ax9.plot(p9, 'k', label="mode")
            try:
                ax9.plot(p9_1, 'r', label="std_prc")
            except:
                pass
            ax9.plot(p9_2, 'b', label="std_prc_peak")
            ax9.plot(p9_2 + p8_3, 'k', label="sum_peak")
            ax9.fill_between(x, 0, p9_2)
            # ax9.plot(p9_2, 'k', label="std_std_prc")
            ax9.grid()
            ax9.legend(loc=2)

            if old_new == 1:
                p9_3 = df_sys.loc[plot_start:plot_last, "std_prc_1000"]
                ax10.plot(p9_3, 'k', label="std_prc_1000")
                p9_4 = df_sys.loc[plot_start:plot_last, "std_prc_peak_1000"]
                ax10.plot(p9_4, 'g', label="std_prc_peak_1000")
            # ax10.fill_between(x, 0, p9_4)
            # ax9.plot(p9_2, 'k', label="std_std_prc")
            p10 = df_sys.loc[plot_start:plot_last, "cvol_m"]
            # try:
            #     p10_1 = df_sys.loc[plot_start:plot_last, "ema_520_cvol_m"]
            # except:
            #     print("no ema_520_cvol_m")
            ax10.plot(p10 / p10.max(), 'g', label="cvol_m")
            try:
                ax10.plot(p10, 'k', label="ema_520_cvol_m")
            except:
                print("no ema_520_cvol_m")
            ax10.grid()
            ax10.legend(loc=4)

            #########################3
            p2 = df_sys.loc[plot_start:plot_last, "profit"]
            ax11.plot(p2, 'b', label="profit")
            ax11.fill_between(x, 0, p2)
            ax11.grid()
            ax11.legend(loc=2)

            # p10 = df_sys.loc[plot_start:plot_last, "count_m"]
            p20 = df_sys.loc[plot_start:plot_last, "count_m_act_ave"]
            p22 = df_sys.loc[plot_start:plot_last, "count_m"]
            # p10_2 = df_sys.loc[plot_start:plot_last, "count_m_per_5"]
            if which_market == 3:
                if old_new == 0:
                    p21 = df_sys.loc[plot_start:plot_last, "count_m_per_40"]
                if old_new == 1:
                    p21 = df_sys.loc[plot_start:plot_last, "count_m_per_80"]
                    p24 = df_sys.loc[plot_start:plot_last, "ema_520_count_m"]
            if which_market == 4:
                if old_new == 0:
                    p21 = df_sys.loc[plot_start:plot_last, "count_m_per_10"]
                if old_new == 1:
                    p21 = df_sys.loc[plot_start:plot_last, "count_m_per_80"]
                    p24 = df_sys.loc[plot_start:plot_last, "ema_520_count_m"]
            # ax10.plot(p10, 'g', label="count_m")
            ax12.plot(p20, 'k', label="count_m_act_ave")
            ax12.plot(p22 / p22.max() * 10, 'g', label="count_m")
            if which_market == 3:
                if old_new == 0:
                    ax12.plot(p21, 'r', label="count_m_per_40")
                if old_new == 1:
                    ax12.plot(p21, 'r', label="count_m_per_80")
                    ax12.plot(p24, 'b', label="ema_520_count_m")
            if which_market == 4:
                if old_new == 0:
                    ax12.plot(p21, 'r', label="count_m_per_10")
                if old_new == 1:
                    ax12.plot(p21, 'r', label="count_m_per_80")
                    ax12.plot(p24, 'b', label="ema_520_count_m")
            ax12.grid()
            ax12.legend(loc=2)
            #

            if 1 == 0:
                p11 = df_sys.loc[plot_start:plot_last, "old_signal"]
                p12 = df_sys.loc[plot_start:plot_last, "new_signal"]
                p12_1 = df_sys.loc[plot_start:plot_last, "test_signal"]
                ax12.plot(p11, 'k', label="old_signal")
                ax12.plot(p12, 'g', label="new_signal")
                ax12.plot(p12_1, 'r', label="test_signal")
                ax12.grid()
                ax12.legend(loc=3)


        except:
            print("printing... error")

        try:
            plt.show()
        except:
            print("print df... error")


def ynet(p, t, W, sw, a, b, c, d):
    if p == t:
        if sw == "Buy":
            result = (a - b + W)
        else:
            result = (a - b)
    elif p < t:
        if sw == "Buy":
            result = (a - b + c) + W
        else:
            result = (a - b + c)
    elif p > t:
        if sw == "Buy":
            result = (a - b - d) + W  # = W - b + a - d
        else:
            result = (a - b - d)
    return result


def xnet(p, t, W, sw, a, b, c, d):
    if p == t:
        if sw == "Sell":
            result = (a - b + W)
        else:
            result = (a - b)
    elif p > t:
        if sw == "Sell":
            result = (a - b + c) + W
        else:
            result = (a - b + c)
    elif p < t:
        if sw == "Sell":
            result = (a - b - d) + W
        else:
            result = (a - b - d)
    return result

def cal_rsi(data, window):
    # print("data: ", data)
    diff = data.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)

    ema_up = up.ewm(com=window - 1, adjust=False).mean()
    ema_down = down.ewm(com=window - 1, adjust=False).mean()

    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def cal_ema(data, window):
    ema = data.ewm(span=window, adjust=False).mean()
    return ema