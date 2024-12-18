import json

# 文件路径配置
txt_file_path = '/root/VideoTGB/mp4_3_video.txt'  # 输入的 .txt 文件
json_file_path = '/root/VideoTGB/inputs/videoinstruct/VideoInstruct_Dataset3.json'  # 输入的 .json 文件
output_file_path = '/root/VideoTGB/inputs/videoinstruct/mp4_3_video.json'  # 输出的 .json 文件

# 读取 .txt 文件中的 video_id 数据
def read_video_ids(txt_file_path):
    with open(txt_file_path, 'r') as file:
        video_ids = {line.strip() for line in file if line.strip()}
    return video_ids

# 提取匹配的 JSON 数据
def extract_matching_data(video_ids, json_file_path):
    matching_data = []
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
        for item in json_data:
            if item.get('video_id') in video_ids:
                matching_data.append(item)
    return matching_data

# 保存匹配的数据到新的 .json 文件
def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 主程序
def main():
    # Step 1: 读取 .txt 文件中的 video_id
    video_ids = read_video_ids(txt_file_path)

    # Step 2: 从 .json 文件中提取匹配的数据
    matching_data = extract_matching_data(video_ids, json_file_path)

    # Step 3: 将匹配的数据保存到新的 .json 文件
    save_to_json(matching_data, output_file_path)
    print(f"匹配的数据已保存到 {output_file_path}")

if __name__ == "__main__":
    main()
