import csv

# 文件路径
csv_file_path = 'original_data/每个地址被标记wash的次数.csv'
result_file_path = r'qyy\detect\community3.txt'
output_file_path = 'qyy/txt/final_community_detection_results_with_index.txt'

# 创建地址到索引的映射
address_to_index = {}

# 读取CSV文件并填充映射
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for index, row in enumerate(csv_reader):
        address_to_index[row[0]] = index

# 读取分类结果文件并进行地址替换
with open(result_file_path, mode='r', encoding='utf-8') as result_file:
    lines = result_file.readlines()

# 创建新的内容
new_lines = []
for line in lines:
    if ' -- ' in line:
        address, community = line.split(' -- ')
        address = address.strip()
        community = community.strip()
        if address in address_to_index:
            new_line = f"{address_to_index[address]} -- {community}\n"
        else:
            print(f"警告: 未找到地址 {address} 对应的索引")
            new_line = f"{address} -- {community}\n"
        new_lines.append(new_line)
    else:
        new_lines.append(line)

# 写入新的分类结果文件
with open(output_file_path, mode='w', encoding='utf-8') as output_file:
    output_file.writelines(new_lines)

print("地址替换完成，结果已保存到:", output_file_path)
