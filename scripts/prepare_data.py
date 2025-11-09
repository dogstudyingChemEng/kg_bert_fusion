"""
将FB15K的tsv文件转换为KG-BERT所需的txt格式
同时重命名dev.tsv为valid.txt
"""
import os
import shutil

def convert_data(input_dir, output_dir):
    """转换数据格式并复制到新目录"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件名映射
    file_mapping = {
        'dev.tsv': 'valid.txt',
        'train.tsv': 'train.txt',
        'test.tsv': 'test.txt'
    }
    
    for src_name, dst_name in file_mapping.items():
        src_path = os.path.join(input_dir, src_name)
        dst_path = os.path.join(output_dir, dst_name)
        
        print(f"处理文件: {src_name} -> {dst_name}")
        
        # 读取并写入，保持tsv格式（因为已经是tab分隔的）
        with open(src_path, 'r', encoding='utf-8') as src, \
             open(dst_path, 'w', encoding='utf-8') as dst:
            for line in src:
                # 如果需要，这里可以添加数据清洗或转换逻辑
                dst.write(line)
                
    # 复制其他重要的辅助文件（如果需要的话）
    aux_files = ['entities.txt', 'relations.txt']
    for fname in aux_files:
        src_path = os.path.join(input_dir, fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, output_dir)
            print(f"复制辅助文件: {fname}")

if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = "data/FB15K"
    output_dir = "data/FB15k-237"  # 使用标准数据集名称
    
    # 执行转换
    convert_data(input_dir, output_dir)
    print("数据准备完成!")