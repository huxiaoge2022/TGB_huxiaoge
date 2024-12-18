import os

# 定义视频文件夹路径和输出文件路径
video_folder = '/root/VideoTGB/inputs/videoinstruct/video'  # 替换为你的视频文件夹路径
output_file = '/root/VideoTGB/video_filenames2.txt'   # 输出的 .txt 文件名

# 支持的视频文件扩展名
video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm']

def extract_video_filenames(video_folder, output_file):
    # 检查视频文件夹是否存在
    if not os.path.exists(video_folder):
        print(f"Error: Folder '{video_folder}' does not exist.")
        return
    
    # 初始化文件名列表
    video_filenames = []

    # 遍历文件夹，提取视频文件名
    for root, _, files in os.walk(video_folder):
        for file in files:
            # 检查文件扩展名是否是支持的视频格式
            if os.path.splitext(file)[1].lower() in video_extensions:
                video_filenames.append(file)
    
    # 检查是否找到视频文件
    if not video_filenames:
        print(f"No video files found in folder '{video_folder}'.")
        return
    
    # 将文件名保存到 .txt 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in video_filenames:
            f.write(filename + '\n')
    
    print(f"Video filenames have been saved to '{output_file}'.")

# 运行程序
extract_video_filenames(video_folder, output_file)
