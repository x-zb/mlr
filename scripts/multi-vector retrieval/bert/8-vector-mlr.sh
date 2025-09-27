#!/bin/bash 

umask 000

# pip install accelerate 
# pip install pytrec_eval
# pip install faiss-gpu
# conda install -c conda-forge faiss-gpu

pip install accelerate==0.22.0
pip install faiss-gpu
pip install gpustat tensorboard 

# export TRANSFORMERS_CACHE="~/.cache/huggingface"
# export MIXED_PRECISION=fp16
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU memory dgx 16GB*4 is sufficient

PATH=$PATH:/home/runai-home/.local/bin

accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 \
	--num_machines=1 --dynamo_backend=no -m	proj_dense.train_dense_retriever \
	--do_train --task "squad1" \
	--model_name_or_path "bert-base-uncased" \
	--learner_type "dm" --untie_query_encoder --layers "[12,11,10,9,8,7,6,5]" \
	--num_warmup_steps 1237 --weight_decay 0.0 --max_grad_norm 2.0 \
	--data_dir "../data/openqa" \
	--output_dir "results_squad1/dmde8_last8_lr5_e40" \
	--max_length 256 --logging_steps 100 --eval_batch_size 32 \
	--data_load_type "memory" --num_proc 16 \
	--evaluation_strategy "epoch" --seed 12345 \
	--learning_rate 2e-5 --num_train_epochs 40 --train_batch_size 128 --chunk_size 16 \
	--negs_sources "official" \
	--num_negs_per_system 200 --num_negatives 1 \
	--do_xentropy \
	--xentropy_reg_loss_weight 0.0 \
	--dev_key_metric "loss" \
	--train_set_evaluation \
	--val_av_rank_start_epoch 30 \
	> run_dmde8_last8_lr5_e40.out 2>&1
	
	
export NUM_SHARDS=15
for _id in $(seq 0 $((${NUM_SHARDS}-1))); do
	echo "encoding shard ${_id}"
	accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 \
	--num_machines=1 --dynamo_backend=no -m proj_dense.train_dense_retriever \
		--task "squad1" \
		--model_name_or_path "results_squad1/dmde8_last8_lr5_e40" \
		--learner_type "dm" --untie_query_encoder --layers "[12,11,10,9,8,7,6,5]" \
		--data_dir "../data/openqa" \
		--output_dir "results_squad1/dmde8_last8_lr5_e40" \
		--seed 12345 \
		--do_encoding --shard_id ${_id} --num_shards ${NUM_SHARDS} \
		--prediction_source "test" \
		--num_proc 16 \
		--max_length 256 \
		--eval_batch_size 128 \
		> pred_dmde8_last8_lr5_e40_${_id}.out 2>&1
done	


accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 \
	--num_machines=1 --dynamo_backend=no -m proj_dense.train_dense_retriever \
	--task "squad1" \
	--model_name_or_path "results_squad1/dmde8_last8_lr5_e40" \
	--learner_type "dm" --untie_query_encoder --layers "[12,11,10,9,8,7,6,5]" \
	--faiss_mode "gpu" \
	--data_dir "../data/openqa" \
	--output_dir "results_squad1/dmde8_last8_lr5_e40" \
	--seed 12345 \
	--do_prediction \
	--prediction_source "test" \
	--num_proc 16 \
	--max_length 256 \
	--eval_batch_size 128 \
	--hits_num 100 \
	> pred_dmde8_last8_lr5_e40.out 2>&1