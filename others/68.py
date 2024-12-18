def add_path_and_extension(input_file, output_file, path_prefix):
    """
    为 .txt 文件中的每条数据添加路径和 .mp4 后缀，并保存到新的文件中。

    :param input_file: 输入的 TXT 文件路径
    :param output_file: 输出的 TXT 文件路径
    :param path_prefix: 要添加到每条数据前的路径（如 "/videos/"）
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 去除空格和换行符
                line = line.strip()
                if line:  # 跳过空行
                    # 添加路径前缀和 .mp4 后缀
                    new_line = f"{path_prefix}{line}.mp4"
                    outfile.write(new_line + '\n')
        
        print(f"处理完成！结果已保存到 {output_file}")
    
    except Exception as e:
        print(f"发生错误: {e}")

# 调用函数
add_path_and_extension("/root/VideoTGB/inputs/videoinstruct/video/effectivate_mp4_2.txt.txt", "/root/VideoTGB/inputs/videoinstruct/video/effectivate_mp4_3.txt", "/root/VideoTGB/inputs/videoinstruct/video/")