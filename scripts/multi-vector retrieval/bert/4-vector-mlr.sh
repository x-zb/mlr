#!/bin/bash 

accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 \
	--num_machines=1 --dynamo_backend=no -m	proj_dense.train_dense_retriever \
	--do_train --task "squad1" \
	--model_name_or_path "bert-base-uncased" \
	--learner_type "dm" --untie_query_encoder --layers "[12,11,10,9]" \
	--num_warmup_steps 1237 --weight_decay 0.0 --max_grad_norm 2.0 \
	--data_dir "../data/openqa" \
	--output_dir "results_squad1/dmde4_last4_lr5_e40" \
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
	> run_dmde4_last4_lr5_e40.out 2>&1
	
	
export NUM_SHARDS=15
for _id in $(seq 0 $((${NUM_SHARDS}-1))); do
	echo "encoding shard ${_id}"
	accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 \
	--num_machines=1 --dynamo_backend=no -m proj_dense.train_dense_retriever \
		--task "squad1" \
		--model_name_or_path "results_squad1/dmde4_last4_lr5_e40" \
		--learner_type "dm" --untie_query_encoder --layers "[12,11,10,9]" \
		--data_dir "../data/openqa" \
		--output_dir "results_squad1/dmde4_last4_lr5_e40" \
		--seed 12345 \
		--do_encoding --shard_id ${_id} --num_shards ${NUM_SHARDS} \
		--prediction_source "test" \
		--num_proc 16 \
		--max_length 256 \
		--eval_batch_size 128 \
		> pred_dmde4_last4_lr5_e40_${_id}.out 2>&1
done	


accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 \
	--num_machines=1 --dynamo_backend=no -m proj_dense.train_dense_retriever \
	--task "squad1" \
	--model_name_or_path "results_squad1/dmde4_last4_lr5_e40" \
	--learner_type "dm" --untie_query_encoder --layers "[12,11,10,9]" \
	--faiss_mode "gpu" \
	--data_dir "../data/openqa" \
	--output_dir "results_squad1/dmde4_last4_lr5_e40" \
	--seed 12345 \
	--do_prediction \
	--prediction_source "test" \
	--num_proc 16 \
	--dev_key_metric "top-5" \
	--max_length 256 \
	--eval_batch_size 128 \
	--hits_num 100 \
	> pred_dmde4_last4_lr5_e40.out 2>&1