#!/bin/bash

# 定义输入文件和输出目录
input_file="/AssistRAG/dataset/hotpot/memory/train_split_0"
output_dir="/AssistRAG/dataset/hotpot/memory/" # 修改为您的输出目录

# 并行启动进程
for i in {0..7}
do
   CUDA_VISIBLE_DEVICES=$i python temporal.py --data_path "${output_dir}/train_split_${i}" --output_path "${output_dir}/output_0${i}.json" &
done

# 等待所有后台进程完成
wait

# 合并输出文件
cat ${output_dir}/output_*.json > ${output_dir}/final_output.json

echo "处理完成。"
