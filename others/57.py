import json

def extract_video_ids(json_file, output_txt_file):
    """
    从JSON文件中提取所有"video_id"值并保存到文本文件中。

    :param json_file: 输入的 JSON 文件路径
    :param output_txt_file: 输出的 TXT 文件路径
    """
    try:
        # 读取 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取所有 "video_id" 值
        video_ids = []
        for item in data:
            if "video_id" in item:
                video_ids.append(item["video_id"])

        # 将结果保存到 TXT 文件
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            for video_id in video_ids:
                f.write(f"{video_id}\n")
        
        print(f"成功提取 {len(video_ids)} 个 video_id，保存到 {output_txt_file}")
    
    except Exception as e:
        print(f"发生错误: {e}")

# 调用函数
extract_video_ids("/root/VideoTGB/inputs/videoinstruct/VideoInstruct_Dataset3.json", "/root/VideoTGB/inputs/videoinstruct/video/effectivate_mp4.txt")
