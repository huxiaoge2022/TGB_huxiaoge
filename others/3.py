''' 检查在a.txt文件中的数据是否在b.txt中都有 '''

# 定义文件路径
file_a = "/root/VideoTGB/inputs/videoinstruct/video/train_video_ids.txt"
file_b = "/root/VideoTGB/video_filenames2.txt"

# 读取文件内容并转为集合
with open(file_a, 'r', encoding='utf-8') as fa:
    data_a = set(line.strip() for line in fa if line.strip())  # 去掉空白和空行

with open(file_b, 'r', encoding='utf-8') as fb:
    data_b = set(line.strip() for line in fb if line.strip())  # 去掉空白和空行

# 检查是否全部存在
missing_data = data_a - data_b  # 计算在a中但不在b中的数据

if missing_data:
    print("以下数据在 b.txt 中不存在：")
    for item in missing_data:
        print(item)
else:
    print("a.txt 中的所有数据都在 b.txt 中。")
