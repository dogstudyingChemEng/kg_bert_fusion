# coding=utf-8
import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from sklearn import metrics

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

# 设置环境变量来捕获CUDA错误
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class KGProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()
    
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_entities(self, data_dir):
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) >= 2:  # 确保有足够的元素
                    ent2text[temp[0]] = temp[1]

        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf-8') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    if len(temp) >= 2:  # 确保有足够的元素
                        ent2text[temp[0]] = temp[1]             

        examples = []
        for (i, line) in enumerate(lines):
            if len(line) < 3:  # 确保每行有足够的数据
                continue
                
            guid = "%s-%s" % (set_type, i)
            if line[0] not in ent2text or line[2] not in ent2text:
                print(f"警告: 实体 {line[0]} 或 {line[2]} 不在entity2text中")
                continue
                
            text_a = ent2text[line[0]]
            text_b = ent2text[line[2]]
            label = line[1]
            self.labels.add(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info=True):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if not tokens_a:  # 确保tokenization不为空
            tokens_a = [tokenizer.unk_token]

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if not tokens_b:  # 确保tokenization不为空
                tokens_b = [tokenizer.unk_token]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        # 转换token到ID，处理未知token
        input_ids = []
        for token in tokens:
            if token in tokenizer.vocab:
                input_ids.append(tokenizer.vocab[token])
            else:
                input_ids.append(tokenizer.vocab[tokenizer.unk_token])

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # 确保标签在label_map中
        if example.label not in label_map:
            print(f"警告: 标签 {example.label} 不在标签列表中")
            continue
            
        label_id = label_map[example.label]

        features.append(
            InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default="kg", type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default="./models", type=str,
                        help="The output directory where the model is stored.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_predict", action='store_true', default=True,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    # 设置默认值
    if args.output_dir is None:
        args.output_dir = "./models"
    
    # 检查模型目录是否存在
    if not os.path.exists(args.output_dir):
        raise ValueError(f"模型目录不存在: {args.output_dir}")

    # 强制使用CPU进行调试
    # args.no_cuda = True
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # 记录总开始时间
    total_start_time = time.time()

    task_name = args.task_name.lower()
    processors = {"kg": KGProcessor}
    
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_relations(args.data_dir)
    num_labels = len(label_list)
    
    # print(f"标签列表: {label_list}")
    # print(f"标签数量: {num_labels}")

    # 记录模型加载开始时间
    model_load_start_time = time.time()
    
    try:
        # 加载tokenizer和模型
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # print(f"词汇表大小: {len(tokenizer.vocab)}")
        
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        model.to(device)
        
        model_load_time = time.time() - model_load_start_time
        # logger.info(f"模型加载耗时: {model_load_time:.2f} 秒")
        
        # 移除多GPU支持，先使用单GPU
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
            
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # Prediction on test set
    if args.do_predict:
        # 记录数据预处理开始时间
        data_preprocess_start_time = time.time()
        
        try:
            train_triples = processor.get_train_triples(args.data_dir)
            dev_triples = processor.get_dev_triples(args.data_dir)
            test_triples = processor.get_test_triples(args.data_dir)
            all_triples = train_triples + dev_triples + test_triples

            all_triples_str_set = set()
            for triple in all_triples:
                if len(triple) >= 3:  # 确保三元组格式正确
                    triple_str = '\t'.join(triple)
                    all_triples_str_set.add(triple_str)

            eval_examples = processor.get_test_examples(args.data_dir)
            # print(f"测试样本数量: {len(eval_examples)}")
            
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, print_info=False)
            
            # print(f"转换后的特征数量: {len(eval_features)}")
            
            # 检查输入数据
            if len(eval_features) == 0:
                print("错误: 没有有效的特征数据")
                return
                
            # 记录数据预处理结束时间
            data_preprocess_time = time.time() - data_preprocess_start_time
            
            logger.info("***** Running Prediction *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            # 检查输入数据的范围
            # print(f"input_ids范围: {all_input_ids.min()} 到 {all_input_ids.max()}")
            # print(f"词汇表大小: {len(tokenizer.vocab)}")
            
            # 确保输入ID在合理范围内
            if all_input_ids.max() >= len(tokenizer.vocab):
                print(f"错误: input_ids包含超出词汇表范围的索引 (max={all_input_ids.max()}, vocab_size={len(tokenizer.vocab)})")
                # 修复超出范围的索引
                all_input_ids = torch.clamp(all_input_ids, 0, len(tokenizer.vocab)-1)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            
            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            # 记录模型推理开始时间Q
            inference_start_time = time.time()
            
            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    try:
                        logits = model(input_ids=input_ids, 
                                      token_type_ids=segment_ids, 
                                      attention_mask=input_mask, 
                                      labels=None)
                    except Exception as e:
                        print(f"模型推理错误: {str(e)}")
                        # 尝试使用不同的参数名称
                        logits = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

            # 记录模型推理结束时间
            inference_time = time.time() - inference_start_time
            
            # 记录后处理开始时间
            postprocess_start_time = time.time()
            
            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            
            all_label_ids = all_label_ids.numpy()

            # Calculate ranking metrics
            ranks = []
            filter_ranks = []
            hits = []
            hits_filter = []
            for i in range(10):
                hits.append([])
                hits_filter.append([])

            for i, pred in enumerate(preds):
                rel_values = torch.tensor(pred)
                _, argsort1 = torch.sort(rel_values, descending=True)
                argsort1 = argsort1.cpu().numpy()
                
                rank = np.where(argsort1 == all_label_ids[i])[0][0]
                ranks.append(rank + 1)
                
                test_triple = test_triples[i]
                filter_rank = rank
                
                for tmp_label_id in argsort1[:rank]:
                    tmp_label = label_list[tmp_label_id]
                    tmp_triple = [test_triple[0], tmp_label, test_triple[2]]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str in all_triples_str_set:
                        filter_rank -= 1
                filter_ranks.append(filter_rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

                    if filter_rank <= hits_level:
                        hits_filter[hits_level].append(1.0)
                    else:
                        hits_filter[hits_level].append(0.0)   

            # 记录后处理结束时间
            postprocess_time = time.time() - postprocess_start_time
            
            # 计算总时间
            total_time = time.time() - total_start_time

            # 打印时间统计
            print("\n" + "="*60)
            print("推理时间统计")
            print("="*60)
            # print(f"数据预处理时间: {data_preprocess_time:.2f} 秒")
            print(f"模型推理时间: {inference_time:.2f} 秒")

            print("-"*60)
            print(f"平均每个样本推理时间: {inference_time/len(eval_examples)*1000:.2f} 毫秒")
            # print(f"平均每批次推理时间: {inference_time/nb_eval_steps*1000:.2f} 毫秒")
            print(f"样本总数: {len(eval_examples)}")
            # print(f"批次数: {nb_eval_steps}")
            print("="*60 + "\n")

            # 打印结果
            print("Raw mean rank: ", np.mean(ranks))
            print("Filtered mean rank: ", np.mean(filter_ranks))
            for i in [0, 2, 9]:
                print('Raw Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
                print('Filtered Hits @{0}: {1}'.format(i+1, np.mean(hits_filter[i])))
            
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(task_name, preds, all_label_ids)
            result['eval_loss'] = eval_loss

            # 将时间信息也写入结果文件
            result['model_load_time'] = model_load_time
            result['data_preprocess_time'] = data_preprocess_time
            result['inference_time'] = inference_time
            result['postprocess_time'] = postprocess_time
            result['total_time'] = total_time
            result['avg_time_per_sample'] = inference_time/len(eval_examples)*1000
            result['num_samples'] = len(eval_examples)

            # output_eval_file = os.path.join(args.output_dir, "test_results.txt")
            # with open(output_eval_file, "w") as writer:
            #     logger.info("***** Test results *****")
            #     for key in sorted(result.keys()):
            #         logger.info("  %s = %s", key, str(result[key]))
            #         writer.write("%s = %s\n" % (key, str(result[key])))
            
            # Relation prediction accuracy
            print("Relation prediction accuracy (hits@1, raw):", metrics.accuracy_score(all_label_ids, preds))
            
        except Exception as e:
            print(f"推理过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()