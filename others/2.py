'''对比a.txt和b.txt文件的数据，将不同的数据提取出来，并保存到c.txt文件中'''


# 定义文件路径
file_a = "/root/VideoTGB/video_filenames2.txt"
file_b = "/root/VideoTGB/inputs/videoinstruct/video/train_video_ids.txt"
file_c = "/root/VideoTGB/duoyu.txt"

# 读取文件内容并转为集合
with open(file_a, 'r', encoding='utf-8') as fa:
    data_a = set(fa.readlines())  # 按行读取并存为集合

with open(file_b, 'r', encoding='utf-8') as fb:
    data_b = set(fb.readlines())  # 按行读取并存为集合

# 找出不同的数据
diff_data = data_a.symmetric_difference(data_b)  # 获取差集 (对称差)

# 将不同的数据写入c.txt
with open(file_c, 'w', encoding='utf-8') as fc:
    fc.writelines(diff_data)

print(f"对比完成，不同的数据已保存到 {file_c}")
