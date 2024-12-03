import pandas as pd

# 读取CSV文件
file_path = "original_data/原始BLUR交易记录.csv"
df = pd.read_csv(file_path)

# 定义一个函数，找出作为卖方或买方经手同一个NFT超过3次的地址
def find_frequent_addresses(df, min_count=2):
    nft_groups = df.groupby(['PROJECT_NAME', 'TOKENID'])
    frequent_addresses = {}
    total_groups = len(nft_groups)
    
    print(f"共有 {total_groups} 个NFT组需要处理")

    for i, ((project_name, tokenid), group) in enumerate(nft_groups, start=1):
        seller_counts = group['SELLER_ADDRESS'].value_counts()
        buyer_counts = group['BUYER_ADDRESS'].value_counts()

        combined_counts = seller_counts.add(buyer_counts, fill_value=0)
        frequent = combined_counts[combined_counts > min_count]

        if not frequent.empty:
            frequent_addresses[(project_name, tokenid)] = frequent.to_dict()

        if i % 100 == 0 or i == total_groups:
            print(f"已处理 {i}/{total_groups} 个NFT组")

    return frequent_addresses

print("开始查找频繁地址...")
frequent_addresses = find_frequent_addresses(df)
print("查找完成")

# 将结果写入TXT文件
output_file = "qyy/txt/nft_loop.txt"
print(f"正在将结果写入 {output_file} ...")
with open(output_file, 'w', encoding='utf-8') as f:
    for (project_name, tokenid), addresses in frequent_addresses.items():
        f.write(f"NFT项目: {project_name}, TokenID: {tokenid}\n")
        for address, count in addresses.items():
            f.write(f"地址：{address}: {count} 次\n")
            

print(f"结果已保存到 {output_file}")
