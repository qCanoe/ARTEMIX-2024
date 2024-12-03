import pandas as pd

# 读取CSV文件
data = pd.read_csv('original_data\\原始BLUR交易记录.csv')

# 创建反向的买卖对
reverse_transactions = data.copy()
reverse_transactions.rename(columns={'SELLER_ADDRESS': 'BUYER_ADDRESS', 'BUYER_ADDRESS': 'SELLER_ADDRESS'}, inplace=True)

# 合并原始交易和反向交易
combined_transactions = pd.concat([data, reverse_transactions])

# 统计每对买卖双方之间的交易次数
transaction_counts = combined_transactions.groupby(['SELLER_ADDRESS', 'BUYER_ADDRESS']).size().reset_index(name='counts')

# 过滤出交易次数超过20次的记录
reciprocal_transactions = transaction_counts[transaction_counts['counts'] > 50]

# 提取来回交易的节点
reciprocal_nodes = pd.unique(reciprocal_transactions[['SELLER_ADDRESS', 'BUYER_ADDRESS']].values.ravel('K'))

# 保存结果为TXT文件
with open('reciprocal_transaction_nodes.txt', 'w') as file:
    for node in reciprocal_nodes:
        file.write(f"{node}\n")

# 找出交易次数最多的两个来回交易节点
top_reciprocal_transactions = reciprocal_transactions.nlargest(2, 'counts')
top_nodes = top_reciprocal_transactions[['SELLER_ADDRESS', 'BUYER_ADDRESS', 'counts']]

# 打印结果
print(top_nodes)
