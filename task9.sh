echo "Task 9"
echo "测试步骤：计算合并后模型较合并前2个模型在训练推断时间复杂度、参数复杂度和空间复杂度减少的百分比"
echo "预期结果：输出训练推断时间复杂度、参数复杂度和空间复杂度减少的百分比"
echo

bash get_train_time_seed42_log.sh

bash get_train_time_seed666_log.sh

bash get_train_time_ot_log.sh

python extract_time_and_compute_ratio.py
python ratio.py