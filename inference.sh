#!/bin/bash

dataset_name="hotpot"
model_name="assistllm-chatglm3-6b"
#llm_name="llama-2-13b-chat"
llm_name="gpt-35-turbo"
mode="rag"
temporal_path="/data/webgpt/IIA/dataset/${dataset_name}/results/assist_${model_name}.json"
output_path="/data/webgpt/IIA/dataset/${dataset_name}/results/answer_${model_name}_${llm_name}.json"
evaluate_path="/data/webgpt/IIA/dataset/${dataset_name}/evaluation/evaluation.txt"

python code/main.py --dataset_name ${dataset_name} --temporal_path ${temporal_path} --output_path ${output_path} --stage assistance --model_name ${model_name}
python code/main.py --dataset_name ${dataset_name} --temporal_path ${temporal_path} --output_path ${output_path} --mode ${mode} --stage answer --llm_name ${llm_name} --evaluate_path ${evaluate_path} --model_name ${model_name}