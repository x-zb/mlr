import json
import numpy as np
import time
import torch

from peach.datasets.marco.dataset_marco_eval import DatasetMacroPassages, DatasetFullRankQueries
from peach.base import *
from peach.common import file_exists, dir_exists
from tqdm import tqdm
import os
import pickle
import faiss
from peach.enc_utils.general import get_representation_tensor
import collections
from peach.common import load_tsv
from peach.enc_utils.trec_utils import evaluate_trec_file


def evaluate_dense_retreival(
    args, eval_dataset, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="recall@100", delete_model=False, add_title=True,
    tokenizer=None, faiss_mode="gpu",
    get_emb_lambda=None, hits=1000,
    **kwargs,
):
    eval_dataset = eval_dataset.split('-')
    assert eval_dataset[0]=='marco'
    eval_dataset = eval_dataset[1]
    assert isinstance(eval_dataset, str)

    model.eval()

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval")

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
    accelerator.wait_for_everyone()

    collection_pids, collection_embs, all_qids, all_query_embs = None, None, None, None

    # encode passages
    dense_index_path = os.path.join(work_dir, "dense_index.pkl")  # collection_pids, collection_embs
    if not file_exists(dense_index_path): 
        # if the collection has already been encoded, will not re-encode it
        
        # get all vectors for the passage collection
        with accelerator.main_process_first():
            passages_dataset = DatasetMacroPassages(
                "train", args.data_dir, None, args, tokenizer, add_title=add_title) # data_type="train" is useless
        passages_dataloader = setup_eval_dataloader(
            args, passages_dataset, accelerator, use_accelerator=True)

        pids_list, passage_vectors_list = [], []
        for batch_idx, batch in tqdm(
                enumerate(passages_dataloader), disable=not accelerator.is_local_main_process,
                total=len(passages_dataloader), desc=f"Getting passage vectors ..."):
            pids = batch.pop("pids")
            with torch.no_grad():
                # embs = get_emb_lambda(enc_model(**batch)).contiguous()
                embs = get_emb_lambda(model(**batch,training_mode=None,is_query=None)).contiguous() # (bsz,n_prompt,hidden_size)

            pids, embs = accelerator.gather(pids), accelerator.gather(embs)
            # for the last batch, examples prioritize the first gpu (gpu with smaller device index)

            if accelerator.is_local_main_process:
                pids_list.append(pids.cpu().numpy())
                passage_vectors_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            collection_pids = np.concatenate(pids_list, axis=0)[:len(passages_dataset)]
            collection_embs = np.concatenate(passage_vectors_list, axis=0)[:len(passages_dataset)] # (p_size,n_prompt,hidden_size)
            with open(dense_index_path, "wb") as fp:
                pickle.dump([collection_pids, collection_embs], fp, protocol=4)
    accelerator.wait_for_everyone()

    # encode queries
    query_embs_path = os.path.join(work_dir, f"{eval_dataset}_query_embs.pkl")
    if not file_exists(query_embs_path):
        with accelerator.main_process_first():
            queries_dataset = DatasetFullRankQueries(eval_dataset, args.data_dir, None, args, tokenizer, )
        queries_dataloader = setup_eval_dataloader(args, queries_dataset, accelerator, use_accelerator=True)
        qids_list, query_embs_list = [], []
        for batch_idx, batch in tqdm(
                enumerate(queries_dataloader), disable=not accelerator.is_local_main_process,
                total=len(queries_dataloader), desc=f"Getting query vectors ..."):
            qids = batch.pop("qids")
            for k in list(batch.keys()):
                if k.endswith("_query"):
                    batch[k[:-6]] = batch.pop(k) 

            with torch.no_grad():
                embs = get_emb_lambda(model(**batch,training_mode=None,is_query=True)).contiguous() # (bsz,hidden_size)

            qids, embs = accelerator.gather(qids), accelerator.gather(embs)
            if accelerator.is_local_main_process:
                qids_list.append(qids.cpu().numpy())
                query_embs_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            all_qids = np.concatenate(qids_list, axis=0)[:len(queries_dataset)] # (q_size,hidden_size)
            all_query_embs = np.concatenate(query_embs_list, axis=0)[:len(queries_dataset)] # (q_size,hidden_size)
            with open(query_embs_path, "wb") as fp:
                pickle.dump([all_qids, all_query_embs], fp, protocol=4)
    accelerator.wait_for_everyone()

    if delete_model:  # delete model to free space for faiss
        free_model(model, accelerator)
    else:
        model.train()
    free_memory()

    # do dense retrieval
    if accelerator.is_local_main_process:
        if collection_pids is None or collection_embs is None: # this branch means the collection has been encoded previously. Just load it.
            with open(dense_index_path, "rb") as fp:
                collection_pids, collection_embs = pickle.load(fp) # (p_size),(p_size,n_prompt,hidden_size)
        if all_qids is None or all_query_embs is None:
            with open(query_embs_path, "rb") as fp:
                all_qids, all_query_embs = pickle.load(fp) # (q_size),(q_size,hidden_size)

        # faiss stuff to build index
        # faiss_mode = "gpu"
        logger.info(f"Using faiss_mode {faiss_mode} ...")
        t0 = time.time()
        logger.info("Building faiss index ...")
        _,n_prompt_dim,dim = collection_embs.shape
        collection_embs.resize(len(collection_pids)*n_prompt_dim,dim) # added
        if faiss_mode == "cpu":
            index_engine = faiss.IndexFlatIP(dim)
            index_engine.add(collection_embs)
        elif faiss_mode == "gpu":
            ngpus = faiss.get_num_gpus()
            logger.info(f"found {ngpus} gpus in faiss ...")            
            index_flat = faiss.IndexFlatIP(dim)
            if ngpus==1:
                res = faiss.StandardGpuResources()
                index_engine = faiss.index_cpu_to_gpu(res, 0, index_flat)
            else:
                cloner_options = faiss.GpuMultipleClonerOptions()
                # cloner_options.useFloat16 = use_float16
                # cloner_options.useFloat16CoarseQuantizer = use_float16_coarse_quantizer
                cloner_options.usePrecomputed = True
                # cloner_options.indicesOptions = indices_options
                cloner_options.verbose = True
                cloner_options.shard = True
                # if reserve_vecs:
                #     cloner_options.reserveVecs = reserve_vecs
                gpu_resources = []
                for i in range(ngpus):
                    res = faiss.StandardGpuResources()
                    # if temp_memory >= 0:
                    #     res.setTempMemory(temp_memory)
                    gpu_resources.append(res)
                vres = faiss.GpuResourcesVector()
                vdev = faiss.IntVector()
                for i in range(ngpus):
                    vres.push_back(gpu_resources[i])
                    vdev.push_back(i)
                index_engine = faiss.index_cpu_to_gpu_multiple(vres,vdev,index_flat,cloner_options)
            index_engine.add(collection_embs)
        else:
            raise NotImplementedError(faiss_mode)
        logger.info(f"Using {time.time() - t0}sec to build index for {collection_embs.shape[0]} docs")

        def _batch_search(_index, _queries, _batch_size, _k):
            Ds, Is = [], []
            for _start in tqdm(range(0, _queries.shape[0], _batch_size)):
                D, I = _index.search(_queries[_start: _start + _batch_size], k=_k)
                Ds.append(D) # (_batch_size,_k) or (last_batch_size,_k)
                Is.append(I) # (_batch_size,_k) or (last_batch_size,_k)
            return np.concatenate(Ds, axis=0), np.concatenate(Is, axis=0) # (q_size,_k),(q_size,_k)

        t0 = time.time()
        logger.info("Doing faiss search ...")
        assert index_engine.is_trained
        D, I = _batch_search(index_engine, all_query_embs, 
            _batch_size=64 if faiss_mode=='gpu' else 1024, 
            _k=min(hits*n_prompt_dim,2048) if faiss_mode=='gpu' else hits*n_prompt_dim) # 2048 is a limitation by faiss
        del index_engine
        logger.info(f"Using {time.time() - t0}sec to complete search for {all_query_embs.shape[0]} queries")

        # save to qid2negatives.pkl
        # metric_ranking = datasets.load_metric("peach/metrics/ranking_v2.py")
        # metric_ranking.qids_list = []
        search_results = []
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            # gold_pids = qid2pos_pids[qid]
              
            top_pids = []
            top_scores = []
            for i,p_index in enumerate(I[q_index]):
                pid_in_hits = int(collection_pids[int(p_index)//n_prompt_dim])
                if pid_in_hits not in top_pids:
                    top_pids.append(pid_in_hits)
                    top_scores.append(D[q_index][i]) # only the largest (first) score of a passage will be added
            # top_pids = [int(collection_pids[int(p_index)]) for p_index in I[q_index]]
            # top_scores = D[q_index]
            if len(top_pids)>=hits:
                top_pids = top_pids[:hits]
                top_scores = top_scores[:hits]
            else:
                accelerator.print(f"Warning: For query {qid}, # searched docs ({len(top_pids)}) is less than {hits}")
                top_pids = top_pids+["0"]*(hits-len(top_pids))
                top_scores = top_scores+[0.0]*(hits-len(top_pids))

            # metric_ranking.add_batch(predictions=top_scores, references=top_pids)
            # metric_ranking.qids_list.extend([qid] * len(top_pids))
            search_results.extend(
                [(qid, pid, idx + 1, score)
                 for idx, (pid, score) in enumerate(zip(top_pids, top_scores))]) # top_scores.tolist()))])
        # eval_metrics = metric_ranking.compute(
        #     qrels_path=os.path.join(args.data_dir, f"passage_ranking/qrels.{eval_dataset}.tsv"),
        #     group_labels=metric_ranking.qids_list, )

        search_result_filename = f"{eval_dataset}_search_result.trec" \
            if hits == 1000 else f"{eval_dataset}_search_result_hits{hits}.trec"
        search_result_path = os.path.join(work_dir, search_result_filename)
        with open(search_result_path, "w") as fp:
            for qid, pid, rank, score in search_results:
                fp.write(f"{qid} Q0 {pid} {rank} {score} DenseRetrieval")
                fp.write(os.linesep)

        eval_metrics = evaluate_trec_file(
            search_result_path, os.path.join(args.data_dir, f"passage_ranking/qrels.{eval_dataset}.tsv"))
        # {"QueriesRanked":int, "MRR@{k}":float, "MRR_official@{k}":float, "recall@{k}":float, "recall_official@{k}":float, "NDCG@{k}":float}

        # print the values of the evaluated metrics:
        logger.info(f"step {global_step}: {eval_metrics}")
        
        if global_step is not None and tb_writer is not None:
            for key, val in eval_metrics.items():
                if isinstance(val, (float, int)):
                    tb_writer.add_scalar(f"eval_in_train-{key}", val, global_step)
        
        key_metric = eval_metrics[key_metric_name]
    else:
        key_metric = NEG_INF
        eval_metrics = {}
    accelerator.wait_for_everyone()
    return key_metric, eval_metrics