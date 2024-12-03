def remove_duplicates(input_file, output_file):
    # 读取文件并去除重复地址
    with open(input_file, 'r', encoding='utf-8') as f:
        addresses = f.readlines()

    # 去除地址两端的空白字符并去重
    addresses = [address.strip() for address in addresses]
    unique_addresses = list(set(addresses))

    # 计算重复的地址数量
    num_duplicates = len(addresses) - len(unique_addresses)
    print(f"发现 {num_duplicates} 个重复地址")

    # 将去重后的地址保存到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for address in unique_addresses:
            f.write(address + '\n')

    print(f"去重后的地址已保存到 {output_file}")

# 输入和输出文件路径
input_file = r'qyy\txt\filtered_addresses.txt'
output_file = r'qyy\txt\filtered_addresses.txt'

# 去除重复地址
remove_duplicates(input_file, output_file)
