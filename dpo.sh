#!/bin/bash

# 定义输入文件和输出目录
input_dir="/AssistRAG/dataset/hotpot/memory"
output_dir="/AssistRAG/data/dpo" # 修改为您的输出目录

# 并行启动进程
for i in {0..7}
do
   CUDA_VISIBLE_DEVICES=$i python code/dpo_generation.py --dataset_path "${input_dir}/train_split_${i}" --output_path "${output_dir}/output_${i+8}.json" &
done

# 等待所有后台进程完成
wait

for i in {0..7}
do
   CUDA_VISIBLE_DEVICES=$i python code/dpo_generation.py --mode preference_alignment --dataset_path "${output_dir}/output_${i+8}.json" --output_path "${output_dir}/output_llama_${i+8}.json" &
done

wait

echo "处理完成。"
