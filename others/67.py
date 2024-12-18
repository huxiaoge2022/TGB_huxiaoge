def remove_duplicates(input_file, output_file):
    """
    从输入的 TXT 文件中删除重复数据，只保留唯一数据，并保存到新的文件。

    :param input_file: 输入的 TXT 文件路径
    :param output_file: 输出的 TXT 文件路径
    """
    try:
        # 使用集合记录已读取的行
        seen = set()
        
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 去除空格和换行符
                line = line.strip()
                if line not in seen:
                    # 如果是新数据，写入输出文件
                    outfile.write(line + '\n')
                    seen.add(line)
        
        print(f"处理完成！已将去重后的数据保存到 {output_file}")
    
    except Exception as e:
        print(f"发生错误: {e}")

# 调用函数
remove_duplicates("/root/VideoTGB/inputs/videoinstruct/video/effectivate_mp4.txt", "/root/VideoTGB/inputs/videoinstruct/video/effectivate_mp4_2.txt.txt")