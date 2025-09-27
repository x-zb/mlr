import os

import torch.distributed as dist

from peach.base import *





def validate_nll(
    args, eval_dataset, model, accelerator, global_step=None, tb_writer=None, save_prediction=False, # "save_prediction" is unused below
    key_metric_name="loss", similarity_metric=None, # "similarity metric" is unused below
    use_accelerator=True,  get_emb_lambda=None, name_prefix="", train_set_evaluation=False,
    **kwargs,
):
    model.eval()
    if args.learner_type=='dm': # TODO: add 'me'.
        num_vec = len(model.module.layer_indices)
    else:
        num_vec = 1

    eval_dataloader = setup_eval_dataloader(args, eval_dataset, accelerator, use_accelerator=use_accelerator)

    if accelerator.is_local_main_process:
        total_loss = 0
        total_xentropy_loss = 0        
        total_correct_predictions = 0
        total_examples = 0
        total_ranks = 0
        vec_dist = {i:0 for i in range(num_vec)}

    # get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    # remain_example = len(eval_dataset)  # for DDP eval
    for batch_idx, batch in tqdm(
            enumerate(eval_dataloader), disable=not accelerator.is_local_main_process,
            total=len(eval_dataloader), desc=f"Eval at step-{global_step} ..." ):
        if not use_accelerator:
            batch = dict((k, v.to(accelerator.device)) for k, v in batch.items())

        batch_size = batch['input_ids_query'].shape[0]
        with torch.no_grad():
            outputs = model(training_mode="retrieval_finetune", **batch) 
            outputs = model.module.compute_loss(*outputs) # dict_for_meta
            loss = outputs['loss']*batch_size*dist.get_world_size()/args.gradient_accumulation_steps # (1)
            xentropy_loss = outputs['xentropy_dense_loss']*batch_size*dist.get_world_size()/args.gradient_accumulation_steps # (1)
            correct_cnt = outputs['correct_predictions_count'] # (1)

            if accelerator.is_local_main_process:
                total_examples += batch_size*dist.get_world_size()
                total_loss += loss.cpu().item() # (bsz*world_size)[:remain_example]->(1)->float
                total_xentropy_loss += xentropy_loss.cpu().item() # (bsz*world_size)[:remain_example]->(1)->float
                total_correct_predictions += correct_cnt.cpu().item() # (bsz*world_size)[:remain_example]->(1)->int
                
                sorted_indices = torch.argsort(outputs['dense_ib_similarities'],dim=1,descending=True) # (bsz*num_ranks,bsz*num_doc*num_ranks)
                ranks = torch.argwhere(sorted_indices==outputs['xentropy_target'].unsqueeze(1).expand(sorted_indices.shape))[:,1] 
                # (bsz*num_ranks,bsz*num_doc*num_ranks)->(bsz*num_ranks,2)->(bsz*num_ranks)
                
                ranks = ranks.cpu()

                total_ranks += ranks.sum().item() # (bsz*world_size)->(1)->int

                reg_target = outputs['gold_reg_target'] if 'gold_reg_target' in outputs else None # (bsz*num_ranks,bsz*num_doc*num_ranks)
                reg_target = reg_target.cpu() if reg_target is not None else None # (bsz*num_ranks,bsz*num_doc*num_ranks)
                if reg_target is not None:
                    for i in range(num_vec):
                        vec_dist[i] += (reg_target.flatten()==i).sum().item()

    model.train()

    if accelerator.is_local_main_process:
        total_loss = total_loss / total_examples
        total_xentropy_loss = total_xentropy_loss / total_examples
        correct_ratio = float(total_correct_predictions / total_examples)
        average_rank = total_ranks / total_examples

        eval_metrics = {'loss':total_loss,'xentropy_loss':total_xentropy_loss,
            'correct_ratio':correct_ratio,'average_rank':average_rank}
        if train_set_evaluation:
            logger.info(
                "NLL Validation on train set at step %d: loss = %f, xentropy_loss = %f, correct prediction ratio  %d/%d ~  %f, average_rank = %f",
                global_step,
                total_loss,
                total_xentropy_loss,
                total_correct_predictions,
                total_examples,
                correct_ratio,
                average_rank
            )
            logger.info(f'vec_dist (train, gold): {vec_dist}')
        else:
            logger.info(
                "NLL Validation on dev set at step %d: loss = %f, xentropy_loss = %f, correct prediction ratio  %d/%d ~  %f, average_rank = %f",
                global_step,
                total_loss,
                total_xentropy_loss,
                total_correct_predictions,
                total_examples,
                correct_ratio,
                average_rank
            )
            logger.info(f'vec_dist (dev, gold): {vec_dist}')
        return eval_metrics[key_metric_name], eval_metrics
    else:
        return NEG_INF, {}

def validate_average_rank(
    args, eval_dataset, model, accelerator, global_step=None, tb_writer=None, save_prediction=False, # "save_prediction" is unused below
    key_metric_name="loss", similarity_metric=None, # "similarity metric" is unused below
    use_accelerator=True,  get_emb_lambda=None, name_prefix="",
    **kwargs,
):
	pass