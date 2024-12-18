''' 判断一个.txt文件中的数据是都有重复的 '''

# 定义文件路径
file_path = "/root/VideoTGB/video_filenames2.txt"

# 读取文件内容并转为列表
with open(file_path, 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file if line.strip()]  # 去掉空白和空行

# 使用集合判断是否有重复数据
unique_lines = set(lines)
if len(unique_lines) < len(lines):
    print("文件中存在重复数据。")
    # 找到重复数据
    from collections import Counter
    duplicates = [item for item, count in Counter(lines).items() if count > 1]
    print("重复的数据包括：")
    for item in duplicates:
        print(item)
else:
    print("文件中没有重复数据。")
