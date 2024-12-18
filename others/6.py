'''
检索一个.json文件中的"video_id"键，与一个.txt文件中的数据进行比对，
若与该.txt文件中的数据相同，则删除该"video_id"键所对应的整个JSON对象，并保存为新的.json文件。

'''


import json

# 定义文件路径
json_file = "/root/VideoTGB/inputs/videoinstruct/VideoInstruct_Dataset2.json"  # 输入 JSON 文件
txt_file = "/root/VideoTGB/other_videos2.txt"  # 包含需要比对的 video_id 的文本文件
output_json_file = "/root/VideoTGB/inputs/videoinstruct/VideoInstruct_Dataset3.json"  # 输出过滤后的 JSON 文件

# 读取 .txt 文件中的数据
with open(txt_file, 'r', encoding='utf-8') as file:
    ids_to_remove = set(line.strip() for line in file if line.strip())  # 去除空白行和空格

# 读取 .json 文件
with open(json_file, 'r', encoding='utf-8') as file:
    json_data = json.load(file)  # 加载 JSON 数据，假定为列表格式

# 过滤 JSON 数据
filtered_data = [item for item in json_data if str(item.get("video_id")) not in ids_to_remove]

# 保存过滤后的数据到新的 JSON 文件
with open(output_json_file, 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

print(f"过滤完成，结果已保存到 {output_json_file}")
