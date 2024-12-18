# 文件路径
input_file = '/root/VideoTGB/other_videos.txt'  # 输入的TXT文件
output_file = '/root/VideoTGB/other_videos2.txt'  # 输出的TXT文件

def remove_suffix(input_file, output_file):
    try:
        # 打开输入文件并读取内容
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        # 去掉后缀并保留文件名
        new_lines = [line.rsplit('.', 1)[0] + '\n' for line in lines if '.' in line]
        
        # 保存到新的文件
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(new_lines)
        
        print(f"处理完成！结果已保存到 {output_file}")
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到！")
    except Exception as e:
        print(f"发生错误：{e}")

# 调用函数
remove_suffix(input_file, output_file)
