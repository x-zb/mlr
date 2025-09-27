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


def get_hard_negative_by_dense_retrieval(
    args, model, accelerator, global_step=None, tb_writer=None, save_prediction=False, # "save_prediction" is unused
    key_metric_name="recall@100", delete_model=False, add_title=True,
    tokenizer=None, faiss_mode="gpu",
    get_emb_lambda=None, hits=1000,
    **kwargs,
):
    model.eval()

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval")

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
    accelerator.wait_for_everyone()

    collection_pids, collection_embs, all_qids, all_query_embs = None, None, None, None
    
    # encode passages, will skip if already encoded.
    dense_index_path = os.path.join(work_dir, "dense_index.pkl")  # collection_pids, collection_embs
    if not file_exists(dense_index_path):
        # get all vectors for collections
        with accelerator.main_process_first():
            passages_dataset = DatasetMacroPassages(
                "dev", args.data_dir, None, args, tokenizer, add_title=add_title) # data_type="dev" is useless
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

            if accelerator.is_local_main_process:
                pids_list.append(pids.cpu().numpy())
                passage_vectors_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            collection_pids = np.concatenate(pids_list, axis=0)[:len(passages_dataset)]
            collection_embs = np.concatenate(passage_vectors_list, axis=0)[:len(passages_dataset)]
            with open(dense_index_path, "wb") as fp:
                pickle.dump([collection_pids, collection_embs], fp, protocol=4)
    accelerator.wait_for_everyone()

    # encode queries
    query_source = args.prediction_source
    query_embs_path = os.path.join(work_dir, f"{query_source}_query_embs.pkl")
    if not file_exists(query_embs_path):
        with accelerator.main_process_first():
            queries_dataset = DatasetFullRankQueries(query_source, args.data_dir, None, args, tokenizer, )
        queries_dataloader = setup_eval_dataloader(args, queries_dataset, accelerator, use_accelerator=True)
        qids_list, query_embs_list = [], []
        for batch_idx, batch in tqdm(
                enumerate(queries_dataloader), disable=not accelerator.is_local_main_process,
                total=len(queries_dataloader), desc=f"Getting query vectors ..."):
            qids = batch.pop("qids")
            for k in list(batch.keys()):
                if k.endswith("_query"):
                    batch[k[:-6]] = batch.pop(k)
                    # the original item whose key ends with "_query" will be removed from the batch, 
                    # so the modified batch can be directly input to the model

            with torch.no_grad():
                # embs = get_emb_lambda(query_enc_model(**batch)).contiguous()
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
        if collection_pids is None or collection_embs is None:
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
                Ds.append(D)
                Is.append(I)
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
        qrels = load_tsv(os.path.join(args.data_dir, f"passage_ranking/qrels.{query_source}.tsv"))
        qid2pos_pids = collections.defaultdict(set)
        for qrel in qrels:
            assert len(qrel) == 4
            qid, pid = int(qrel[0]), int(qrel[2])
            qid2pos_pids[qid].add(pid)

        # metric_ranking = datasets.load_metric("peach/metrics/ranking.py")
        # metric_ranking.qids_list = []
        qid2negatives = dict()
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            qid = int(qid)
            gold_pids = qid2pos_pids[qid]
            
            # top_pids = [int(collection_pids[int(p_index)]) for p_index in I[q_index]]
            top_pids = []
            # top_scores = []
            for i,p_index in enumerate(I[q_index]):
                pid_in_hits = int(collection_pids[int(p_index)//n_prompt_dim])
                if pid_in_hits not in top_pids:
                    top_pids.append(pid_in_hits)
                    # top_scores.append(D[q_index][i])
            if len(top_pids)>=hits:
                top_pids = top_pids[:hits]
                # top_scores = top_scores[:hits]
            else:
                accelerator.print(f"Warning: For query {qid}, # searched docs ({len(top_pids)}) is less than {hits}")
                top_pids = top_pids+["0"]*(hits-len(top_pids))
                # top_scores = top_scores+[0.0]*(hits-len(top_pids))

            # top_references = np.array([int(pid in gold_pids) for pid in top_pids], dtype="int64")
            # top_scores = D[q_index]
            # metric_ranking.add_batch(predictions=top_scores, references=top_references)
            # metric_ranking.qids_list.extend([qid] * 1000)
            negative_pids = [pid for pid in top_pids if pid not in gold_pids] # remove the gold positive passage from hard negatives because they are negatives
            qid2negatives[qid] = negative_pids

        # save qid2negatives to pickle file
        # if 'dmde' in args.model_name_or_path: 
        #     new_train_filename = f'qid2negatives.{query_source}-hn-dm-{args.xentropy_reg_loss_weight}-mae.pkl' # e.g., qid2negatives.train-hn-dm-0.1-mae.pkl
        # elif 'de_' in args.model_name_or_path:
        #     new_train_filename = f'qid2negatives.{query_source}-hn-de-mae.pkl' # e.g., qid2negatives.train-hn-de-mae.pkl
        # else:
        #     new_train_filename = f'qid2negatives.{query_source}-hn-ukn.pkl' # e.g., qid2negatives.train-hn-unk.pkl
        
        new_train_filename = f'qid2negatives.{query_source}-hn.pkl'
        with open(os.path.join(work_dir, new_train_filename), "wb") as qid2negatives_fp:
            pickle.dump(qid2negatives, qid2negatives_fp, protocol=4)
    else:
        qid2negatives = None
    return qid2negatives















