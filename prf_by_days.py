# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

directory_path = "C:/eBEST/xingAPI/npp_files"

plt.figure(figsize=(10, 6))

all_prf_data = []
all_lengths = []
files = [f for f in os.listdir(directory_path) if "(e)df_npp" in f]  # 파일 리스트 생성

# 파일 리스트가 비어있지 않은 경우에만 처리
if files:
    # 가장 마지막 파일 식별
    last_file = files[-1]

    for file in files:
        file_path = os.path.join(directory_path, file)

        try:
            data = pd.read_csv(file_path)

            if 'prf' in data.columns:
                date_info = file.split('_')[2].split('-')[:2]
                date_str = '-'.join(date_info)

                prf_data = data['prf'].values

                all_prf_data.append(prf_data)
                all_lengths.append(len(prf_data))

                # 가장 마지막 파일의 데이터일 경우 선 두께를 두껍게 설정
                if file == last_file:
                    plt.plot(prf_data, label='Date: %s' % date_str, linewidth=3)
                else:
                    plt.plot(prf_data, label='Date: %s' % date_str)
            else:
                print("'prf' column not found in %s" % file)
        except Exception as e:
            print("Error reading %s: %s" % (file, e))

    if all_prf_data:
        max_length = max(all_lengths)  # 가장 긴 길이 찾기
        x_new = np.linspace(0, max_length-1, max_length)  # 새로운 x축 값 생성
        interpolated_data = []

        for data in all_prf_data:
            x_old = np.linspace(0, len(data)-1, len(data))  # 원래 데이터의 x축 값
            interpolated = np.interp(x_new, x_old, data)  # 보간
            interpolated_data.append(interpolated)

        avg_prf = np.mean(interpolated_data, axis=0)
        plt.plot(avg_prf, label='Average PRF', color='black', linestyle='--', linewidth=2)

# prf = 0 인 직선 추가 (빨간색 점선)
plt.axhline(y=0, color='red', linestyle='--', label='PRF = 0')

plt.title('PRF Data from Multiple Files with Average')
plt.xlabel('Index')
plt.ylabel('PRF')
plt.legend()
plt.show()
##