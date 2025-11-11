import os
import torch
import json

def count_parameters(model_path):
    """
    统计PyTorch模型的参数总量
    """
    try:
        # 加载模型
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 计算总参数数量
        total_params = sum(p.numel() for p in model.values())
        
        return total_params
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return 0

def get_model_size(model_path):
    """
    获取模型文件大小（MB）
    """
    try:
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except Exception as e:
        print(f"Error getting file size for {model_path}: {e}")
        return 0

def read_config(model_dir):
    """
    读取模型的配置文件
    """
    config_path = os.path.join(model_dir, 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error reading config from {config_path}: {e}")
        return {}

def main():
    # 模型目录列表
    model_dirs = [
        './models/kg-bert-1',
        './models/kg-bert-2', 
        './models/kg-bert-fused-ot',
        './models/kg-bert-fused-vf'
    ]
    
    print("模型参数统计报告")
    print("=" * 80)
    
    total_stats = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(model_dir, 'pytorch_model.bin')
        
        if not os.path.exists(model_path):
            print(f"警告: 模型文件 {model_path} 不存在，跳过")
            continue
            
        # 统计参数
        total_params = count_parameters(model_path)
        
        # 获取文件大小
        file_size = get_model_size(model_path)
        
        # 读取配置信息
        config = read_config(model_dir)
        
        # 格式化显示
        params_millions = total_params / 1e6
        params_billions = total_params / 1e9
        
        print(f"\n模型: {model_dir}")
        print(f"- 参数总量: {total_params:,}")
        print(f"- 参数总量: {params_millions:.2f} M")
        print(f"- 参数总量: {params_billions:.4f} B")
        print(f"- 模型文件大小: {file_size:.2f} MB")
        
        # 显示配置中的关键信息
        if config:
            hidden_size = config.get('hidden_size', 'N/A')
            num_hidden_layers = config.get('num_hidden_layers', 'N/A')
            num_attention_heads = config.get('num_attention_heads', 'N/A')
            vocab_size = config.get('vocab_size', 'N/A')
            
            print(f"- 隐藏层维度: {hidden_size}")
            print(f"- 隐藏层层数: {num_hidden_layers}")
            print(f"- 注意力头数: {num_attention_heads}")
            print(f"- 词表大小: {vocab_size}")
        
        total_stats.append({
            'model': model_dir,
            'total_params': total_params,
            'params_m': params_millions,
            'params_b': params_billions,
            'file_size_mb': file_size
        })
    
    # 打印汇总信息
    print("\n" + "=" * 80)
    print("参数统计汇总")
    print("=" * 80)
    
    for stats in total_stats:
        print(f"{stats['model']:20} | {stats['total_params']:>12,} | {stats['params_m']:>8.2f} M | {stats['file_size_mb']:>6.2f} MB")
    
    # 计算参数差异
    if len(total_stats) >= 2:
        print("\n参数差异分析:")
        base_params = total_stats[0]['total_params']
        for i, stats in enumerate(total_stats[1:], 1):
            diff = stats['total_params'] - base_params
            diff_percent = (diff / base_params) * 100
            print(f"{stats['model']} 相比 {total_stats[0]['model']}: {diff:+,} 参数 ({diff_percent:+.2f}%)")

if __name__ == "__main__":
    main()