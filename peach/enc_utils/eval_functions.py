import collections
import os

from peach.base import *
from peach.common import save_jsonl, save_pickle, file_exists
import time

from peach.datasets.marco.dataset_marco_eval import DatasetRerank
from peach.enc_utils.general import get_representation_tensor

def evaluate_encoder_reranking(
    args, eval_dataset, model, accelerator, global_step=None, tb_writer=None, save_prediction=False, # "save_prediction" is unused below
    key_metric_name="recall@100", similarity_metric=None, # "similarity metric" is unused below
    use_accelerator=True,  get_emb_lambda=None, name_prefix="",
    **kwargs,
):
    
    model.eval()

    eval_dataloader = setup_eval_dataloader(args, eval_dataset, accelerator, use_accelerator=use_accelerator)

    if accelerator.is_local_main_process:
        metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        # metric_ranking_dot = datasets.load_metric("peach/metrics/ranking.py")
        metric_ranking.qids_list = []
        if global_step is None:  # only for eval
            rerank_results = collections.defaultdict(list)
        else:
            rerank_results = None # during dev, we do not need to record results
    else:
        metric_ranking, metric_ranking_dot = None, None
        rerank_results = None

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    remain_example = len(eval_dataset)  # for DDP eval
    for batch_idx, batch in tqdm(
            enumerate(eval_dataloader), disable=not accelerator.is_local_main_process,
            total=len(eval_dataloader), desc=f"Eval at step-{global_step} ..." ):
        if not use_accelerator:
            batch = dict((k, v.to(accelerator.device)) for k, v in batch.items())

        with torch.no_grad():
            qids = batch["qids"]
            pids = batch["pids"]
            doc_batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],}
            # if "token_type_ids" in batch:
            #     doc_batch["token_type_ids"] = batch["input_ids"]

            query_batch = {
                "input_ids": batch["input_ids_query"],
                "attention_mask": batch["attention_mask_query"],}
            # if "token_type_ids_query" in batch:
            #     query_batch["token_type_ids"] = batch["input_ids_query"]

            # emb_doc = get_emb_lambda(enc_model(**doc_batch))
            emb_doc = get_emb_lambda(model(**doc_batch,training_mode=None,is_query=None)).contiguous() # (bsz,n_prompt,hidden_size)
            # emb_query = get_emb_lambda(query_enc_model(**query_batch)) 
            emb_query = get_emb_lambda(model(**query_batch,training_mode=None,is_query=True)).contiguous() # (bsz,hidden_size)

            # qd_sim_tensor = (emb_query * emb_doc).sum(-1).detach() # (bsz)  # todo: add more metrics for pair-wise
            emb_query = emb_query.unsqueeze(1)  # (bsz,hidden_size)->(bsz,1,hidden_size)
            qd_sim_tensor = torch.sum(emb_doc*emb_query, dim=-1).detach()  # (bsz,n_prompt), broadcast, dot product
            qd_sim_tensor = torch.max( qd_sim_tensor,dim=-1)[0] # (bsz)

            # qd_dot_tensor = outputs["qd_dot_sim"].detach()
            binary_labels = batch["binary_labels"] # (bsz)

            if use_accelerator:
                qids = accelerator.gather(qids).cpu()[:remain_example]
                pids = accelerator.gather(pids).cpu()[:remain_example]
                qd_sim_tensor = accelerator.gather(qd_sim_tensor).cpu()[:remain_example] # (bsz*world_size)[:remain_example]
                # qd_dot_tensor = accelerator.gather(qd_dot_tensor).cpu()
                binary_labels = accelerator.gather(binary_labels).cpu()[:remain_example]
                remain_example -= qids.shape[0] # qids has been gathered, which is bsz*world_size for non-last batches

            if accelerator.is_local_main_process:
                metric_ranking.add_batch(
                    predictions=qd_sim_tensor, references=binary_labels)
                metric_ranking.qids_list.append(qids)
                # metric_ranking_dot.add_batch(
                #     predictions=qd_dot_tensor, references=binary_labels)
                if rerank_results is not None:
                    for qid, pid, score in zip(
                            qids.numpy(), pids.numpy(), qd_sim_tensor.numpy()):
                        qid, pid, score = int(qid), int(pid), float(score)
                        rerank_results[qid].append((pid, score))

    model.train()

    if accelerator.is_local_main_process:
        eval_metrics = metric_ranking.compute(
            group_labels=torch.cat(metric_ranking.qids_list,dim=0), num_examples=len(eval_dataset))

        # save to trec
        if rerank_results is not None:
            all_qids = list(rerank_results.keys())
            all_qids.sort()
            qid2trec = dict()
            for qid in all_qids:
                ps_pairs = rerank_results[qid]
                ps_pairs.sort(key=lambda e: e[1], reverse=True) # sort passages according to descending similarity scores
                trec_data = [
                    (pid, idx+1, score) for idx, (pid, score) in enumerate(ps_pairs)] # idx+1 is the rank of the passage
                qid2trec[qid] = trec_data
            with open(os.path.join(args.output_dir, "rerank_result.trec"), "w") as fp:
                for qid in all_qids:
                    for pid, rank, score in qid2trec[qid]:
                        fp.write(f"{qid} Q0 {pid} {rank} {score} Reranker")
                        fp.write(os.linesep)
        logger.info(f"step {global_step}: {eval_metrics}")
        if global_step is not None and tb_writer is not None:
            for key, val in eval_metrics.items():
                if isinstance(val, (float, int)):
                    tb_writer.add_scalar(f"eval_in_train-{name_prefix}{key}", val, global_step)
        return eval_metrics[key_metric_name], eval_metrics
    else:
        return NEG_INF, {}