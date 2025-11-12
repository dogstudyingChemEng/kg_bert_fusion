{ \

  echo "--- 开始运行 1 ---" && \

  python3 run_bert_relation_prediction.py --task_name kg --do_train --data_dir ./data/FB15K --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 2048 --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp/kg-bert --gradient_accumulation_steps 1 --seed 666 && \

  \

  echo "--- 开始运行 2  ---" && \

  python3 run_bert_relation_prediction.py --task_name kg --do_train --data_dir ./data/FB15K --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 2048 --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp/kg-bert --gradient_accumulation_steps 1 --seed 666 && \

  \

  echo "--- 开始运行 3  ---" && \

  python3 run_bert_relation_prediction.py --task_name kg --do_train --data_dir ./data/FB15K --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 2048 --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp/kg-bert --gradient_accumulation_steps 1 --seed 666; \

} > record_seed666.txt 2>&1