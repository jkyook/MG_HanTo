

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# directory_path = "/Users/yugjingwan/PycharmProjects/MG_HanTo"
directory_path = "C:/Users/Administrator/PycharmProjects/MG_HanTo"

files = [file for file in os.listdir(directory_path) if file.startswith('(e)df_npp_')]
print(files)

# 가장 늦은 파일을 찾기 위한 초기 설정
latest_file = None
latest_time = None

for file in files:
    # 파일 이름에서 시간 정보 추출
    time_str = file.split('npp_')[1].rstrip('.csv')  # 파일 이름 형식에 맞게 조정
    # datetime 객체로 변환
    file_time = datetime.strptime(time_str, "%m-%d-%H-%M")

    if latest_time is None or file_time > latest_time:
        latest_time = file_time
        latest_file = file

print(f"가장 늦은 저장 시점의 파일: {latest_file}")



# 파일 읽기
file_path = os.path.join(directory_path, latest_file) #files[1])
print(file_path)
data = pd.read_csv(file_path)
data = data.iloc[49:]

# 데이터에 필요한 열이 있는지 확인
if all(column in data.columns for column in ['now_prc', 'np_sum', 'prf', 'np1', 'np2', 'np_sum', 'real_sum']):
    fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(15, 8))  # 세 개의 그래프 생성
    # fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 8))  # 세 개의 그래프 생성

    # 첫 번째 그래프: 'now_prc'와 'np_sum'
    color = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('now_prc', color=color)
    ax1.plot(data['now_prc'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 이중 축 생성
    ax2.set_ylabel('np1, np2', color='tab:red')
    # ax2.plot(data['np_sum'], color='tab:red', label='np_sum')
    ax2.plot(data['np1'], color='tab:orange', label='np1')  # 'np1' 데이터 추가
    ax2.plot(data['np2'], color='tab:green', label='np2')  # 'np2' 데이터 추가
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper left')

    # 두 번째 그래프: 'now_prc'와 'prf'
    color = 'tab:blue'
    ax3.set_xlabel('Index')
    ax3.set_ylabel('now_prc', color=color)
    ax3.plot(data['now_prc'], color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # 이중 축 생성
    color = 'tab:red'
    ax4.set_ylabel('prf', color=color)
    ax4.plot(data['prf'], color=color, label='prf', lw=2)
    ax4.plot(data['prf1'], color='tab:orange', label='prf1')  # 'np1' 데이터 추가
    ax4.plot(data['prf2'], color='tab:green', label='prf2')  # 'np2' 데이터 추가
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.legend(loc='upper right')

    # 세 번째 그래프: 'np_qty'와 'cum_qty'
    color = 'tab:blue'
    ax5.set_xlabel('Index')
    ax5.set_ylabel('now_prc', color=color)
    ax5.plot(data['now_prc'], color=color)
    ax5.tick_params(axis='y', labelcolor=color)

    ax6 = ax5.twinx()  # 이중 축 생성
    ax6.set_ylabel('np_sum, real_sum', color='tab:red')
    # ax6.plot(data['np_sum'], color='tab:red', label='np_sum')
    ax6.plot(data['np_sum'], color='tab:orange', label='np_sum')  # 'np1' 데이터 추가
    ax6.plot(data['real_sum'], color='tab:green', label='real_sum')  # 'np2' 데이터 추가
    ax6.tick_params(axis='y', labelcolor='tab:red')
    ax6.legend(loc='upper left')

    plt.title(files[-1])
    fig.tight_layout()
    plt.show()

else:
    print("Required columns not found in the file")