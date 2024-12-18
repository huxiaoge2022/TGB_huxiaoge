import os

# 定义文件路径
a_file = '/root/VideoTGB/inputs/videoinstruct/video/train_video_ids.txt'  # 替换为你的 a.txt 文件路径
b_file = '/root/VideoTGB/video_filenames2.txt'  # 替换为你的 b.txt 文件路径
output_file = '/root/VideoTGB/inputs/videoinstruct/video/train_video_ids2.txt'  # 输出文件路径

def extract_common_videos(a_file, b_file, output_file):
    # 检查文件是否存在
    if not (os.path.exists(a_file) and os.path.exists(b_file)):
        print("Error: One or both input files do not exist.")
        return
    
    # 读取 a.txt 和 b.txt 的内容
    with open(a_file, 'r', encoding='utf-8') as f:
        a_videos = set(line.strip() for line in f.readlines())  # 去重并清理换行符
    
    with open(b_file, 'r', encoding='utf-8') as f:
        b_videos = set(line.strip() for line in f.readlines())  # 去重并清理换行符
    
    # 找出两个文件中共同的视频名
    common_videos = a_videos.intersection(b_videos)
    
    # 将结果写入 c.txt 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for video in common_videos:
            f.write(video + '\n')
    
    print(f"Common video filenames have been saved to '{output_file}'.")

# 运行程序
extract_common_videos(a_file, b_file, output_file)
