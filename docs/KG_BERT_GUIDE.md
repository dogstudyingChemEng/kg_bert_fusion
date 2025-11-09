## KG-BERT 融合实验运行指南

### 1. 环境准备

确保安装了必要的依赖：
```bash
pip install transformers torch datasets
```

### 2. 数据准备

1. 创建数据目录结构：
```
data/
  └── FB15k-237/      # 或你的知识图谱数据集名称
      ├── train.txt
      ├── valid.txt
      └── test.txt
```

2. 每个.txt文件应包含tab分隔的三元组，格式：
```
head_entity<tab>relation<tab>tail_entity
```

### 3. 模型准备

有两种方式准备模型：

a) 使用本地预训练好的模型：
- 将模型保存在 `models/` 目录下
- 在实验配置中指向这些目录：
  ```yaml
  model:
    name_0: models/kg_bert_model1
    name_1: models/kg_bert_model2
  ```

b) 使用HuggingFace模型：
- 直接在配置中使用模型ID：
  ```yaml
  model:
    name_0: bert-base-uncased
    name_1: bert-base-uncased
  ```

### 4. 运行实验

1. 首次运行建议使用小规模测试：
```bash
# 使用较少的样本数快速验证
python main.py experiments/kg_bert_fusion.yaml --debug
```

2. 完整实验运行：
```bash
python main.py experiments/kg_bert_fusion.yaml
```

### 5. 配置说明

关键配置项说明：
- `model.type`: 使用 'kg_bert' 
- `fusion.acts.num_samples`: 用于计算OT的样本数，可以先设小一点（如100）测试
- `fusion.acts.avg_seq_items`: 是否对序列中的token取平均，建议对KG-BERT设为True
- `fusion.fuse_gen`: 是否融合分类头，通常设为True

### 6. 预期输出

程序会输出：
1. 各阶段计时（激活计算、OT融合等）
2. 原始模型的评估指标
3. 融合后模型的评估指标（MRR、Hits@N等）

### 7. 常见问题

1. 内存不足：
   - 减小 `fusion.acts.num_samples`
   - 使用 `fusion.acts.avg_seq_items: True`

2. 评估较慢：
   - 可以设置 `regression.only_eval_ot: True` 只评估融合结果

3. 融合效果不佳：
   - 调整 `fusion.sinkhorn_reg` 参数（0.01-0.1之间）
   - 确保两个模型的训练数据分布相似

### 8. 优化建议

1. 激活处理：
```yaml
fusion:
  acts:
    std: True     # 标准化激活
    center: True  # 中心化
    norm: False   # 通常不需要额外的范数归一化
```

2. 残差连接：
```yaml
fusion:
  resid_policy: mean  # 对于BERT通常效果最好
```