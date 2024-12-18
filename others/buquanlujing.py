import json

# 文件路径配置
txt_file_path = '/root/VideoTGB/mp4_99_video.txt'  # 输入的 .txt 文件
output_file_path = '/root/VideoTGB/path_mp4_99_video.txt'  # 输出的补全路径后的 .txt 文件

# 补全路径和后缀
def complete_paths(txt_file_path, output_file_path, base_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as file:
        for line in lines:
            video_id = line.strip()
            if video_id:  # 确保不是空行
                file.write(f"{base_path}/{video_id}.mp4\n")

# 主程序
def main():
    base_path = '/root/VideoTGB/inputs/videoinstruct/video'  # 基础路径，需根据实际情况修改
    complete_paths(txt_file_path, output_file_path, base_path)
    print(f"已补全路径并保存到 {output_file_path}")

if __name__ == "__main__":
    main()

