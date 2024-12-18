# 定义文件路径
input_file = "/root/VideoTGB/video_filenames.txt"  # 输入文件
mp4_output_file = "/root/VideoTGB/mp4_videos.txt"  # 保存.mp4后缀文件
other_output_file = "/root/VideoTGB/other_videos.txt"  # 保存其他后缀文件

# 读取文件内容
with open(input_file, 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file if line.strip()]  # 去掉空白和空行

# 分类数据
mp4_videos = [line for line in lines if line.lower().endswith(".mp4")]
other_videos = [line for line in lines if not line.lower().endswith(".mp4")]

# 保存到不同的文件
with open(mp4_output_file, 'w', encoding='utf-8') as file:
    file.writelines(video + '\n' for video in mp4_videos)

with open(other_output_file, 'w', encoding='utf-8') as file:
    file.writelines(video + '\n' for video in other_videos)

print(f"分类完成，.mp4视频保存到 {mp4_output_file}，其他视频保存到 {other_output_file}")
