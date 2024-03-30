import pandas as pd
import os

class Nprob:
    def __init__(self):
        self.partition_size = 1000  # 파티션 크기 (예: 1000개의 데이터씩 파티셔닝)
        self.partition_index = 0  # 현재 파티션 인덱스
        self.df = pd.DataFrame()  # 현재 파티션의 데이터프레임
        self.partition_dir = "./partitions"  # 파티션 파일이 저장될 디렉토리
        self.merged_dir = "./merged"  # 병합된 파일이 저장될 디렉토리

        # 파티션 디렉토리 생성
        if not os.path.exists(self.partition_dir):
            os.makedirs(self.partition_dir)

        # 병합된 파일 디렉토리 생성
        if not os.path.exists(self.merged_dir):
            os.makedirs(self.merged_dir)

    def add_data(self, data):
        # 데이터를 현재 파티션에 추가
        self.df = self.df.append(data, ignore_index=True)

        # 파티션 크기를 초과하면 새로운 파티션 생성
        if len(self.df) >= self.partition_size:
            self.save_partition()
            self.partition_index += 1
            self.df = pd.DataFrame()

    def save_partition(self):
        # 현재 파티션을 파일로 저장
        partition_path = f"{self.partition_dir}/partition_{self.partition_index}.csv"
        self.df.to_csv(partition_path, index=False)
        print(f"Partition {self.partition_index} saved to {partition_path}")

    def load_partition(self, index):
        # 특정 파티션을 메모리에 로드
        partition_path = f"{self.partition_dir}/partition_{index}.csv"
        if os.path.exists(partition_path):
            self.df = pd.read_csv(partition_path)
            print(f"Partition {index} loaded from {partition_path}")
        else:
            print(f"Partition {index} does not exist")

    def process_data(self):
        # 데이터 처리 로직 (예시)
        print("Processing data...")
        # 현재 파티션의 데이터를 사용하여 원하는 작업 수행
        print(self.df.head())

    def cleanup(self):
        # 모든 파티션 파일 삭제
        for file_name in os.listdir(self.partition_dir):
            file_path = os.path.join(self.partition_dir, file_name)
            os.remove(file_path)
        print("Partitions cleaned up")

    def merge_partitions(self):
        # 모든 파티션 파일을 하나로 병합
        merged_df = pd.DataFrame()
        for file_name in os.listdir(self.partition_dir):
            file_path = os.path.join(self.partition_dir, file_name)
            partition_df = pd.read_csv(file_path)
            merged_df = merged_df.append(partition_df, ignore_index=True)

        # 병합된 파일 저장
        merged_path = f"{self.merged_dir}/merged_data.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"Partitions merged and saved to {merged_path}")

# 사용 예시
nprob = Nprob()

# 데이터 추가
for i in range(5000):
    data = {"nf": i, "price": i * 10}
    nprob.add_data(data)

# 파티션 저장
nprob.save_partition()

# 특정 시점이 되면 파티션들을 병합하고 저장
nprob.merge_partitions()

# 특정 파티션 로드
nprob.load_partition(2)

# 데이터 처리
nprob.process_data()

# 파티션 파일 삭제
nprob.cleanup()