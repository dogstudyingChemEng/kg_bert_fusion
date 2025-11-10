import torch
import numpy as np
import os
import csv
import sys
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score

# 设置日志
log = logging.getLogger('kg_bert_helper')
logging.basicConfig(level=logging.INFO)

# --- 辅助函数 (来自 run_bert_relation_prediction.py) ---

def _read_tsv(input_file, quotechar=None):
    """读取 .tsv 文件 (来自 run_bert_relation_prediction.py)"""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """截断序列对 (来自 run_bert_relation_prediction.py)"""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# --- 1. 数据加载 ---

def load_dataset_kgbert(dataset_name, data_dir, seed=0):
    """
    加载用于 *关系预测* 的数据集。
    这现在匹配 run_bert_relation_prediction.py 的 KGProcessor 逻辑。
    
    返回:
        val_ds (HFDataset): 验证集
        test_ds (HFDataset): 测试集
        relations_list (list): 关系列表 (标签)
    """
    log.info(f"Loading KG dataset '{dataset_name}' from: {data_dir} for Relation Prediction")
    
    # --- 修正: 必须在这里组合 data_dir 和 dataset_name ---
    base_path = os.path.join(data_dir, dataset_name)
    # --- 结束修正 ---

    val_file = os.path.join(base_path, "dev.tsv")
    test_file = os.path.join(base_path, "test.tsv")
    entity_text_file = os.path.join(base_path, "entity2text.txt")
    relations_file = os.path.join(base_path, "relations.txt")
    
    # 1. 加载实体到文本的映射
    ent2text = {}
    with open(entity_text_file, 'r', encoding='utf-8') as f:
        ent_lines = f.readlines()
        for line in ent_lines:
            temp = line.strip().split('\t')
            if len(temp) == 2:
                ent2text[temp[0]] = temp[1]
    
    # 2. 加载关系列表 (标签)
    relations_list = []
    with open(relations_file, 'r', encoding='utf-8') as f:
        for line in f:
            relations_list.append(line.strip())

    # 3. 创建样本
    def create_examples(file_path):
        lines = _read_tsv(file_path)
        examples = []
        for line in lines:
            if len(line) != 3:
                continue
            head_id, relation, tail_id = line
            if head_id in ent2text and tail_id in ent2text:
                examples.append({
                    "text_a": ent2text[head_id],  # 头部实体文本
                    "text_b": ent2text[tail_id],  # 尾部实体文本
                    "label": relation             # 关系 (字符串标签)
                })
        return examples

    val_examples = create_examples(val_file)
    test_examples = create_examples(test_file)
    
    # 转换为 Hugging Face Dataset 对象
    val_ds = HFDataset.from_list(val_examples)
    test_ds = HFDataset.from_list(test_examples)
    
    return val_ds, test_ds, relations_list


# --- 2. 预处理与批处理 ---

def preprocess_kgbert_single(example, tokenizer, label_map, max_len):
    """
    处理单个 *关系预测* 样本。
    (替换 val_transforms_cifar10)
    """
    tokens_a = tokenizer.tokenize(example['text_a'])
    tokens_b = tokenizer.tokenize(example['text_b'])

    _truncate_seq_pair(tokens_a, tokens_b, max_len - 3)

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    tokens += tokens_b + ["[SEP]"]
    segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len

    label_id = label_map[example['label']]
    
    return {
    "input_ids": input_ids,
    "attention_mask": input_mask,
    "token_type_ids": segment_ids,
    "label": label_id
    }

def collate_fn_kgbert(examples):
    """
    将来自 preprocess_kgbert_single 的 Tensors 堆叠成一个批次。
    (替换 collate_fn)
    """
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    token_type_ids = torch.tensor([example["token_type_ids"] for example in examples], dtype=torch.long)
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }


# --- 3. 模型加载 ---

def get_model(path, num_labels):
    """
    从指定路径加载 KG-BERT 模型。
    num_labels 必须是 *关系* 的数量。
    (替换 get_model)
    """
    log.info(f"Loading KG-BERT model (Relation Prediction) from: {path}")
    # num_labels 必须等于关系的
    model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    return model


# --- 4. 评估 ---

def evaluate_kgbert(model, dataloader, device):
    """
    评估 *关系预测* 模型的 (Hits@1 / Accuracy)。
    (替换 evaluate_vit)
    
    注意：此 dataloader 应包含 *预处理过* 的批次数据。
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []

    log.info("Starting KG-BERT evaluation (Relation Prediction Accuracy)...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "token_type_ids": batch["token_type_ids"].to(device)
        }
        labels = batch["labels"].to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels_np)

    # 5. 计算最终指标 (与 run_bert_relation_prediction.py 中的 compute_metrics 一致)
    if not all_labels:
        log.warning("No labels found during evaluation.")
        return 0.0

    # 计算简单准确率 (Hits@1)
    acc = accuracy_score(all_labels, all_preds)
    
    metrics = {
        'test_accuracy (Hits@1)': acc
    }
    
    log.info(f"Evaluation Metrics: {metrics}")
    
    # main.py 的 'get_test_acc' 期望返回一个数字
    return acc