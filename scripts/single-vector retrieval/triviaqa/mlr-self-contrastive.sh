#!/bin/bash 


export CACHE=../../hfcache
	
accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 \
	--num_machines=1 --dynamo_backend=no -m	proj_dense.train_dense_retriever \
	--do_train --task "trivia" \
	--model_name_or_path $CACHE/bert-base-uncased \
	--learner_type "dm" --untie_query_encoder --layers "[12,10]" --pooling "hybride" \
	--num_warmup_steps 1237 --weight_decay 0.0 --max_grad_norm 2.0 \
	--data_dir "../data/openqa" \
	--output_dir "results_tq/dmde2_1210_hybreg1_lr5_e40" \
	--max_length 256 --logging_steps 100 --eval_batch_size 32 \
	--data_load_type "memory" --num_proc 2 \
	--evaluation_strategy "epoch" --seed 12345 \
	--learning_rate 2e-5 --num_train_epochs 40 --train_batch_size 128 --chunk_size 16 \
	--negs_sources "official" \
	--num_negs_per_system 200 --num_negatives 1 \
	--do_xentropy \
	--xentropy_reg_loss_weight 1.0 \
	--dev_key_metric "xentropy_loss" \
	--train_set_evaluation \
	--val_av_rank_start_epoch 30 \
	> tq_run_dmde2_1210_hybreg1_lr5_e40.out 2>&1
	
export NUM_SHARDS=15
for _id in $(seq 0 $((${NUM_SHARDS}-1))); do
	echo "encoding shard ${_id}"
	accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 --main_process_port 29502 \
	--num_machines=1 --dynamo_backend=no -m proj_dense.train_dense_retriever \
		--task "trivia" \
		--model_name_or_path "results_tq/dmde2_1210_hybreg1_lr5_e40/last_checkpoint" \
		--learner_type "vanilla" --untie_query_encoder \
		--data_dir "../data/openqa" \
		--output_dir "results_tq/dmde2_1210_hybreg1_lr5_e40" \
		--seed 12345 \
		--do_encoding --shard_id ${_id} --num_shards ${NUM_SHARDS} \
		--prediction_source "test" \
		--num_proc 2 \
		--max_length 256 \
		--eval_batch_size 128  \
		> tq_pred_dmde2_1210_hybreg1_lr5_e40_eval_${_id}.out 2>&1
done	


accelerate launch --multi_gpu --mixed_precision=no --num_processes=4 --main_process_port 29502 \
	--num_machines=1 --dynamo_backend=no -m proj_dense.train_dense_retriever \
	--task "trivia" \
	--model_name_or_path "results_tq/dmde2_1210_hybreg1_lr5_e40/last_checkpoint" \
	--learner_type "vanilla" --untie_query_encoder \
	--faiss_mode "gpu" \
	--data_dir "../data/openqa" \
	--output_dir "results_tq/dmde2_1210_hybreg1_lr5_e40" \
	--seed 12345 \
	--do_prediction \
	--prediction_source "test" \
	--num_proc 2 \
	--max_length 256 \
	--eval_batch_size 128 \
	--hits_num 100 \
	> tq_pred_dmde2_1210_hybreg1_lr5_e40_eval.out 2>&1