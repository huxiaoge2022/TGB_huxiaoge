import json
import os

# 文件路径
json_file = '/root/VideoTGB/inputs/videoinstruct/VideoInstruct_Dataset.json'  # JSON文件路径
txt_file = '/root/VideoTGB/new_same_video_name.txt'    # TXT文件路径
output_file = '/root/VideoTGB/filtered.json'  # 输出的JSON文件路径

def filter_json_by_video_id(json_file, txt_file, output_file):
    # 检查文件是否存在
    if not (os.path.exists(json_file) and os.path.exists(txt_file)):
        print("Error: 输入文件不存在。请检查文件路径。")
        return
    
    # 加载JSON文件
    with open(json_file, 'r', encoding='utf-8') as jf:
        json_data = json.load(jf)
    
    # 加载TXT文件中的视频名
    with open(txt_file, 'r', encoding='utf-8') as tf:
        valid_video_ids = set(line.strip() for line in tf)
    
    # 筛选JSON对象
    original_count = len(json_data)
    filtered_data = [entry for entry in json_data if entry.get("video_id") in valid_video_ids]
    deleted_count = original_count - len(filtered_data)
    
    # 保存筛选后的JSON
    with open(output_file, 'w', encoding='utf-8') as of:
        json.dump(filtered_data, of, ensure_ascii=False, indent=4)
    
    # 打印结果
    print(f"原始JSON数据条目: {original_count}")
    print(f"删除条目数量: {deleted_count}")
    print(f"剩余条目数量: {len(filtered_data)}")
    print(f"筛选后的数据已保存到: {output_file}")

# 调用函数
filter_json_by_video_id(json_file, txt_file, output_file)


