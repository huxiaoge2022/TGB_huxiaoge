import os

# 定义路径
video_folder = '/root/VideoTGB/inputs/videoinstruct/video'  # 视频文件夹路径
d_file = '/root/VideoTGB/same_video_name.txt'  # 包含保留的视频文件名的文件路径

def remove_unlisted_videos(video_folder, d_file):
    # 检查 d.txt 是否存在
    if not os.path.exists(d_file):
        print(f"Error: File '{d_file}' does not exist.")
        return
    
    # 读取 d.txt 中的视频文件名
    with open(d_file, 'r', encoding='utf-8') as f:
        valid_videos = set(line.strip() for line in f.readlines())  # 去重并清理换行符
    
    # 遍历视频文件夹中的所有文件
    for filename in os.listdir(video_folder):
        video_path = os.path.join(video_folder, filename)
        # 检查是否是文件，并判断文件是否在 d.txt 列表中
        if os.path.isfile(video_path) and filename not in valid_videos:
            print(f"Deleting: {filename}")
            os.remove(video_path)  # 删除文件
    
    print("Video folder updated.")

# 运行程序
remove_unlisted_videos(video_folder, d_file)
