echo "Task 5"
echo "测试步骤：查看合并后模型的训练和推断时间复杂度"
echo "预期结果：输出合并后模型的训练和推断耗时"
echo
echo "KG-BERT-VanillaFusion训练开始"
python3 run_bert_relation_prediction.py --task_name kg --do_train --data_dir ./data/FB15K --bert_model ./models/kg-bert-fused-ot --max_seq_length 25 --train_batch_size 2048 --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp/kg-bert --gradient_accumulation_steps 1  
echo "KG-BERT-VanillaFusion训练结束"
echo "============================================================================"
echo "KG-BERT-VanillaFusion推理开始"
python inference_time.py --data_dir ./data/FB15K --bert_model bert-base-uncased --output_dir ./models/kg-bert-fused-ot --eval_batch_size 1024 --task_name kg --do_predict
echo "KG-BERT-VanillaFusion推理结束"
echo "============================================================================"

echo "KG-BERT-OTFusion训练开始"
python3 run_bert_relation_prediction.py --task_name kg --do_train --data_dir ./data/FB15K --bert_model ./models/kg-bert-fused-vf --max_seq_length 25 --train_batch_size 2048 --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp/kg-bert --gradient_accumulation_steps 1 --seed 666 
echo "KG-BERT-OTFusion训练结束"
echo "============================================================================"
echo "KG-BERT-OTFusion推理开始"
python inference_time.py --data_dir ./data/FB15K --bert_model bert-base-uncased --output_dir ./models/kg-bert-fused-vf --eval_batch_size 1024 --task_name kg --do_predict
echo "KG-BERT-OTFusion推理结束"
echo "============================================================================"