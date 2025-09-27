import collections
import os

from peach.base import *
from peach.common import save_jsonl, save_pickle, file_exists
import time

from peach.datasets.marco.dataset_marco_eval import DatasetRerank
from peach.enc_utils.general import get_representation_tensor
from peach.nn_utils.general import combine_dual_inputs_by_attention_mask

def evaluate_reranker_reranking(
    args, eval_dataset, enc_model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", similarity_metric=None, query_model=None,
    use_accelerator=True,
    **kwargs,
):
    enc_model.eval()

    eval_dataloader = setup_eval_dataloader(args, eval_dataset, accelerator, use_accelerator=use_accelerator)

    if accelerator.is_local_main_process:
        metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        # metric_ranking_dot = datasets.load_metric("peach/metrics/ranking.py")
        metric_ranking.qids_list = []
        if global_step is None:  # only for eval
            rerank_results = collections.defaultdict(list)
        else:
            rerank_results = None
    else:
        metric_ranking, metric_ranking_dot = None, None
        rerank_results = None

    remain_example = len(eval_dataset)  # for DDP eval
    for batch_idx, batch in tqdm(
            enumerate(eval_dataloader), disable=not accelerator.is_local_main_process,
            total=len(eval_dataloader), desc=f"Eval at step-{global_step} ..." ):
        if not use_accelerator:
            batch = dict((k, v.to(accelerator.device)) for k, v in batch.items())

        with torch.no_grad():
            qids = batch["qids"]
            pids = batch["pids"]
            """
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

            emb_doc = get_representation_tensor(enc_model(**doc_batch))
            emb_query = get_representation_tensor(query_enc_model(**query_batch))
            qd_sim_tensor = (emb_query * emb_doc).sum(-1).detach()  # todo: add more metrics for pair-wise
            """
            cross_input_ids, cross_attention_mask, cross_token_type_ids = combine_dual_inputs_by_attention_mask(
                batch["input_ids_query"], batch["attention_mask_query"],
                batch["input_ids"], batch["attention_mask"],
            )
            cross_outputs = enc_model(
                cross_input_ids, cross_attention_mask, cross_token_type_ids)
            qd_sim_tensor = cross_outputs.logits.squeeze(-1)
            # qd_dot_tensor = outputs["qd_dot_sim"].detach()
            binary_labels = batch["binary_labels"]

            if use_accelerator:
                qids = accelerator.gather(qids).cpu()[:remain_example]
                pids = accelerator.gather(pids).cpu()[:remain_example]
                qd_sim_tensor = accelerator.gather(qd_sim_tensor).cpu()[:remain_example]
                # qd_dot_tensor = accelerator.gather(qd_dot_tensor).cpu()
                binary_labels = accelerator.gather(binary_labels).cpu()[:remain_example]
                remain_example -= qids.shape[0]

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

    enc_model.train()

    if accelerator.is_local_main_process:
        eval_metrics = metric_ranking.compute(
            group_labels=torch.cat(metric_ranking.qids_list, dim=0), num_examples=len(eval_dataset))

        # save to trec
        if rerank_results is not None:
            all_qids = list(rerank_results.keys())
            all_qids.sort()
            qid2trec = dict()
            for qid in all_qids:
                ps_pairs = rerank_results[qid]
                ps_pairs.sort(key=lambda e: e[1], reverse=True)
                trec_data = [
                    (pid, idx+1, score) for idx, (pid, score) in enumerate(ps_pairs)]
                qid2trec[qid] = trec_data
            with open(os.path.join(args.output_dir, "rerank_result.trec"), "w") as fp:
                for qid in all_qids:
                    for pid, rank, score in qid2trec[qid]:
                        fp.write(f"{qid} Q0 {pid} {rank} {score} CrossReranker")
                        fp.write(os.linesep)

        logger.info(f"step {global_step}: {eval_metrics}")
        if global_step is not None and tb_writer is not None:
            for key, val in eval_metrics.items():
                if isinstance(val, (float, int)):
                    tb_writer.add_scalar(f"eval_in_train-{key}", val, global_step)
        return eval_metrics[key_metric_name], eval_metrics
    else:
        return NEG_INF, {}