# -*- coding: utf-8 -*-

import numpy as np
import peakutils
from peakutils.plot import plot as pplot
from matplotlib import pyplot as plt
import pandas as pd
import os
# from oct2py import octave
from peakdetect import peakdetect
from scipy.ndimage import gaussian_filter1d

# # peakdetect (0:plus, 1:minus)


if 1==0:
    path_dir = 'D:/DATA/AI_kospi/2021/'
    # path_dir = 'D:/DATA/AI_bit/'
    # path_dir = 'D:/DATA/AI_MESU/'
    file_list = os.listdir(path_dir)
    print('file_list :', len(file_list))
    filename = path_dir + file_list[-1]
    print('filename', filename)

def peak(filename, i, which_y, span):
    global x, y, peaks0, peaks1, p, df

    # path_dir = 'D:/DATA/AI_kospi/'
    # # path_dir = 'D:/DATA/AI_bit/'
    # # path_dir = 'D:/DATA/AI_MESU/'
    # file_list = os.listdir(path_dir)
    # filename = path_dir + file_list[-i]

    # ai = pd.read_csv("index_ai_short.csv").columns.values.tolist()
    ai = pd.read_csv("index_mex_org.csv").columns.values.tolist()
    df = pd.read_csv(filename)[ai]
    print(df)
    # df = df.dropna(subset=['y_3'])
    y = np.array(df["price"])
    x = np.array(df["nf"]) #np.linspace(0, len(y) - 1, len(y))

    if which_y == 1 or which_y == 2:
        # 스팬 간격으로 가격의 픽을 잡아냄
        peaks0 = np.array(pd.DataFrame(peakdetect(y, lookahead=span)[0])[0]) # plus
        peaks1 = np.array(pd.DataFrame(peakdetect(y, lookahead=span)[1])[0]) # minus
        print('peaks0: ', peaks0)
        print('peaks1: ', peaks1)
        p = np.append(peaks0, peaks1)

        # 픽 결과 plot
        # pplot(x, y, p)
        # plt.show()

        # 픽 중간지점을 기준으로 픽 재산정(2차미분)
        if 1==1:
            if peaks0[0]<peaks1[0]: # 저점먼저 시작시
                for j in range(len(peaks0)-1):
                        peaks0[j] = int(peaks0[j]+peaks1[j])/2
                for j in range(1, len(peaks1)-1):
                        peaks1[j] = int(peaks0[j+1]+peaks1[j])/2
            if peaks0[0]>peaks1[0]: # 고점먼저 시작시
                for j in range(1,len(peaks0)-1):
                        peaks0[j] = int(peaks0[j]+peaks1[j-1])/2
                for j in range(len(peaks1)-1):
                        peaks1[j] = int(peaks0[j]+peaks1[j])/2

            print('peaks0 +: ', peaks0)
            print('peaks1 +: ', peaks1)

        # y_2 산출 : 픽에서의 가격을 -100,100으로 우선 세팅(원래 y_2는 10개전 가격차/tick @ nprob)
        for i in range(len(peaks0)):
            df.at[peaks0[i], "y_2"] = int(-100)

        for i in range(len(peaks1)):
            df.at[peaks1[i], "y_2"] = int(100)

        # print("df1")
        # print(df.loc[400:450,"y_2"])

        # when encountered other than zero, replace it to that value
        # (temp) peak일때의 y_2값, 이후 y_2를 이값(100,-100)으로 쭉 밀어버림
        # => 나중에 이걸로 전체 y_2 값을 나눠서 스게일 조정(0-1), 픽아니면 0으로 세팅
        temp = 0
        for i in range(len(df)-1):
            if abs(df.at[i, "y_2"]) == 100:
                temp = df.at[i, "y_2"]
            else:
                df.at[i, "y_2"] = temp

        # print("df2")
        # print(df)

        # covert 0 to pre_value
        # 100 => -1로 변환(최대값 이전은 1, 최소값 이전은 -1)
        temp0 = 0
        for i in range(len(df)-1):
            if abs(df.at[i, "y_2"]) != 0 and temp0 == 0:
                temp0 = df.at[i, "y_2"]
                df.loc[0:i, "y_2"] = int(temp0 * -1)
        #
        # print("df3")
        # print(df)

        for i in range(len(df)-1):
            df.at[i, "y_2"] = df.at[i, "y_2"]/100

        # # -1은 0으로 전환 (x)
        # for i in range(len(df)-1):
        #     if df.at[i, "y_2"] == -1:
        #         df.at[i, "y_2"] = 0

        # (-1,0,1) => (0,1,2) (o)
        df.y_2 = (df.y_2 + 1)/2


        ##### 변동성 모델링 데이터화 #####
        for i in range(len(df) - 1):
            df.at[i, "v_1"] = 0

        # 변곡점 전후 y(v_1)값
        scope = int(span/2) # kospi : 100, s&p : 150
        # for i in range(len(p)):
        #     df.at[p[i], "y_2"] = int(100)

        for i in range(len(df) - 1):
            if i >= 100:
                # df.at[i, "v_1"] = 0
                if df.at[i, "y_2"] > df.at[i - 1, "y_2"] and abs(df.at[i, "v_1"]) != 1:
                    df.loc[i - scope:i + int(scope/1.5), "v_1"] = 1
                if df.at[i, "y_2"] < df.at[i - 1, "y_2"] and abs(df.at[i, "v_1"]) != 1:
                    df.loc[i - scope:i + int(scope/1.5), "v_1"] = -1

        print("v_1: ", df.v_1)
        df.to_csv(filename[:-4] + "_v1.csv")

        # # 결과 저장
        # if filename[-6:-3] != "df.":
        #     df.to_csv(filename[:-4]+"_df.csv")

        #y_2 결과 plot
        if 1==1 :
            fig, ax1 = plt.subplots()
            ax1.plot(x, y, color='black')
            ax2 = ax1.twinx()
            ax2.plot(x, df.y_2, color='red')
            # ax2.plot(x, df.v_1, color='blue')
            ##############
            # plt.show()
            ##############

    # plt.figure(figsize=(10,6))
    # plt.show()

    if which_y == 2: #(300개 가격차이)

        # diff between 300
        for i in range(len(df)-301):
            diff_ = df.at[i+300, "price"] - df.at[i, "price"]
            slp = 0
            if diff_ >= df.at[i, "price"] * 0.4 / 1000:
                slp = 1
            # if diff >= df.at[i, "price"] * 0.02 / 1000:
            #     slp = 1
            # if diff <= df.at[i, "price"] * -0.02 / 1000:
            #     slp = -1
            if diff_ <= df.at[i, "price"] * -0.4 / 1000:
                slp = -1
            df.at[i, "slp"] = slp + 1
            df.at[i, "diff_"] = diff_

        # finding long_start point
        long_point=[]
        for i in range(len(df)-401):
            if df.loc[i:i+400, "slp"].std() == 0:
                long_point.append(i)
                # df.loc[i:i+400, "is_long"] = 1
            # else:
            #     df.loc[i:i+400, "is_long"] = 0

        # assign to 0 in is_long and put 1 in long_start point
        df.loc[0:len(df), "is_long"] = 1
        for i in range(len(long_point)):
            if df.at[long_point[i]+1, "slp"] == 2:
                df.loc[long_point[i]:long_point[i] + 400, "is_long"] = 2
            if df.at[long_point[i]+1, "slp"] == 0:
                df.loc[long_point[i]:long_point[i] + 400, "is_long"] = 0

        if filename[-6:-3] != "df.":
            df.to_csv(filename[:-4]+"_long_df.csv")

def inflection(filename, i, which_y, span):
    global x, y, peaks0, peaks1, p, df

    # path_dir = 'D:/DATA/AI_kospi/2021/'
    # # # path_dir = 'D:/DATA/AI_bit/'
    # # # path_dir = 'D:/DATA/AI_MESU/'
    # file_list = os.listdir(path_dir)
    # filename = path_dir + file_list[-i]

    ai = pd.read_csv("index_mex_org.csv").columns.values.tolist()
    df = pd.read_csv(filename)[ai]
    # print(df)
    y = np.array(df["price"])
    x = np.array(df["nf"])

    # smooth
    smooth = gaussian_filter1d(y, span)

    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(smooth))
    # print('smooth_d2', smooth_d2[0:10])

    # smooth_d2를 df에 행으로 추가하기
    df["smooth_d2"] = smooth_d2

    # # sign 튀는 값 정리
    # for j in range(10,int(len(sign)/1.5)):
    #     n_start = 0
    #     n_time = 0
    #     n_value = 0
    #     print('j', j)
    #     for i in range(j,len(sign),1):
    #         # print('i: ', i)
    #         if sign[i] != None:
    #             if n_start == 0:
    #                 if sign[i] == sign[i - 1]:
    #                     break
    #                 if sign[i] != sign[i - 1]: # 바뀌기 시작하면 카운트 시작
    #                     n_start = i
    #                     n_value = sign[i-1]
    #                     print("n_start: ", n_start)
    #                     print("n_value: ", n_value)
    #             if n_start != 0:
    #                 if sign[i] == sign[i + 1]:
    #                     n_time += 1
    #                     print(j, n_time)
    #                     if n_time >=10:
    #                         break
    #                 if sign[i] != sign[i + 1] and n_time <= 10: #다시 바뀌고 10개 이하면 이전 값으로 전환
    #                     sign[i-n_time:i] = n_value
    #                     # print("n_value: ", n_value)
    #                     break


    # inflection(sign of smooth_d2) => v_1
    df["v_1"] = np.sign(smooth_d2)
    # df.append(pd.Series(smooth_d2, index="smooth_d2"), ignore_index=True)
    print('df.v_1',df.v_1[1300:1320])

    # NaN 제거 (df에서 개수 줄어듦)
    smooth_d2 = [x for x in smooth_d2 if np.isnan(x) == False]
    # print('smooth_d2', smooth_d2[0:10])
    # print('smooth_d2 (NaN)', smooth_d2.isnan(np.NaN))

    # -1/1 변환
    # df["inflection"] = np.sign(smooth_d2)
    # print(np.sign(smooth_d2)) #[2000:3000]) # -1/1로 표시
    df.to_csv(filename[:-4] + "_v.csv")

    # -1/1전환점
    # print(np.diff(np.sign(smooth_d2))[2000:3000]) #
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    print('infls', infls)

    # plot results
    if 1==1:
        plt.plot(y,label='Noisy Data')
        plt.plot(smooth, label='Smoothed Data')
    if 1==1:
        # plt.plot(smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')
        for i, infl in enumerate(infls, 1):
            plt.axvline(x=infl, color='k') #, label=f'Inflection Point {i}')
        # plt.legend(bbox_to_anchor=(1.55, 1.0))

    # plt.show()

# if __name__ == '__main__' or 1==0:
# #     peak(filename, i=0, span=100, which_y=1)
#     for i in range(25, len(file_list), 20):
# inflection(1, which_y=1, span=500)