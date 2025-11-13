import re
import os

def extract_training_times(log_file="record.txt"):
    """
    从指定的日志文件 (record.txt) 中提取所有 "KG-BERT训练总用时" 的时长。
    
    参数:
    log_file (str): 要读取的日志文件名。
    """
    
    # 正则表达式模式:
    # 1. 匹配 "KG-BERT训练总用时:" (来自 run_bert_relation_prediction.py)
    # 2. \s* : 匹配冒号和数字之间的任意空格
    # 3. (\d+\.?\d*): 捕获组 (group 1)，用于捕获数字 (整数或浮点数)
    # 4. \s* : 匹配数字和 "秒" 之间的任意空格
    # 5. 秒 : 匹配 "秒"
    pattern = re.compile(r"KG-BERT训练总用时:\s*(\d+\.?\d*)\s*秒")
    
    extracted_times = []

    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"错误: 未找到日志文件 '{log_file}'。")
        print("请确保 'record.txt' 与此脚本位于同一目录中。")
        return

    # 逐行读取文件
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                # 在当前行中搜索匹配项
                match = pattern.search(line)
                
                # 如果找到匹配项
                if match:
                    # 提取第一个捕获组 (即数字)
                    time_str = match.group(1)
                    extracted_times.append(time_str)
                    
    except Exception as e:
        print(f"读取文件 '{log_file}' 时发生错误: {e}")
        return

    # 打印提取到的结果
    # if extracted_times:
    #     print(f"从 '{log_file}' 中成功提取到 {len(extracted_times)} 个时长：")
    #     for i, t in enumerate(extracted_times, 1):
    #         print(f"  运行 {i}: {t} 秒")
    # else:
    #     print(f"未能在 '{log_file}' 中找到 'KG-BERT训练总用时:' 相关的行。")
    extracted_times = [eval(time) for time in extracted_times]
    s = sum(extracted_times)
    print('总训练时长:', s, '秒')
    return s

if __name__ == "__main__":
    time = []
    files = ["record_seed42.txt", "record_seed666.txt", "record_fused_ot.txt"]
    for file in files:
        print(f"Processing file: {file}")
        time.append(extract_training_times(log_file=file))
    print("两个KG-bert训练总时长：", time[0] + time[1], "秒")
    print("融合后KG-bert训练总时长：", time[2], "秒")
    print("训练时长减少百分比：", (time[0] + time[1] - time[2]) / (time[0] + time[1]) * 100, "%")