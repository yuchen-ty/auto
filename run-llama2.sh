#!/bin/bash

data_path=data/alpaca_gpt4_data.json
eval_data_path=data/databricks-dolly-15k.jsonl



p_type="inject"
p_target="output"
p_data_path=data/autopoison_gpt-3.5-turbo_inject_ns5200_from0_seed0.jsonl
output_dir=./output/autopoison

port=$(shuf -i 6000-9000 -n 1)
echo $port

model_name='llama2-7b'

seed=0
ns=5200

torchrun --nproc_per_node=8 --master_port=${port} main.py \
        --model_name_or_path "meta-llama/Llama-2-7b-hf" \
        --data_path ${data_path} \
        --p_data_path ${p_data_path} --p_seed ${seed} \
        --bf16 True \
        --p_n_sample ${ns} --p_type ${p_type} \
        --output_dir ${output_dir}/${model_name/./-}-${p_type}-${p_target}-ns${ns}-seed${seed} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 100 \
        --fsdp 'full_shard auto_wrap' \
        --report_to none \
        --tf32 True; \
torchrun --nproc_per_node=8 --master_port=${port} main.py \
        --eval_only \
        --model_max_length 2048 \
        --model_name_or_path ${output_dir}/${model_name/./-}-${p_type}-${p_target}-ns${ns}-seed${seed} \
        --data_path ${eval_data_path} \
        --bf16 True \
        --output_dir ${output_dir}/${model_name/./-}-${p_type}-${p_target}-ns${ns}-seed${seed} \
        --per_device_eval_batch_size 16 \
        --fsdp 'full_shard auto_wrap' \
        --tf32 True; \
