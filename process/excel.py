import pandas as pd

# 读取 CSV 文件
file_path = r"C:\Users\canoe\Downloads\Provisional Sybil List\provisionalSybilList.csv"
df = pd.read_csv(file_path)

# 获取 Cluster 列中唯一值的数量
unique_clusters = df['Cluster'].nunique()
print(f"Cluster 列中不一样的值的数量: {unique_clusters}")