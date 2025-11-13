echo
echo "Task 1"
echo "测试步骤：分别查看待合并的2个模型的训练和推断时间复杂度"
echo "预期结果：输出待合并的2个模型的训练和推断耗时"
echo
echo "KG-BERT-1训练开始"
python3 run_bert_relation_prediction.py --task_name kg --do_train --data_dir ./data/FB15K --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 2048 --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp/kg-bert --gradient_accumulation_steps 1  
echo "KG-BERT-1训练结束"
echo "============================================================================"
echo "KG-BERT-1推理开始"
python inference_time.py --data_dir ./data/FB15K --bert_model bert-base-uncased --output_dir models/kg-bert-1 --eval_batch_size 1024 --task_name kg --do_predict
echo "KG-BERT-1推理结束"
echo "============================================================================"

echo "KG-BERT-2训练开始"
python3 run_bert_relation_prediction.py --task_name kg --do_train --data_dir ./data/FB15K --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 2048 --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp/kg-bert --gradient_accumulation_steps 1 --seed 666 
echo "KG-BERT-2训练结束"
echo "============================================================================"
echo "KG-BERT-2推理开始"
python inference_time.py --data_dir ./data/FB15K --bert_model bert-base-uncased --output_dir models/kg-bert-2 --eval_batch_size 1024 --task_name kg --do_predict
echo "KG-BERT-2推理结束"
echo "============================================================================"