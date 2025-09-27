import json
import numpy as np
import time
import glob
import torch

from tqdm import tqdm
import os
import pickle
import faiss
from peach.enc_utils.general import get_representation_tensor
from peach.base import *
from peach.common import file_exists, dir_exists

from typing import Tuple, List, Dict, Iterator

from peach.datasets.openqa.dataset_eval import CsvCtxSrc, CsvQASrc
from peach.enc_utils.eval_openqa.qa_validation import calculate_matches

def validate(
    passages: Dict[object, Tuple[str, str]], # {pid:(text,title)}
    answers: List[List[str]], # List[List[str]]
    result_ctx_ids: List[Tuple[List[object], List[float]]], # List[(List[docid],List[score])], for each query, List[score] is in descending order
    workers_num: int, # 16
    match_type: str, # string
) -> List[List[bool]]:
    logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    
    top_k_hits = match_stats['top_k_hits'] # List[int] of length 100, top_k_hits[k]=recall@k numerator
    logger.info("Validation results: top k documents hits %s", top_k_hits)
    
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits] # top_k_hits[k]=recall@k
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return {"top-5":top_k_hits[4],"top-20":top_k_hits[19],"top-100":top_k_hits[99],
        "questions_doc_hits":match_stats['questions_doc_hits']} 
        # List[List[True/False] of length 100]


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    # `questions, answers, top_passages_and_scores, per_question_hits` are all in the original query order.
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c][1],
                    "text": docs[c][0],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        # if questions_extra_attr and questions_extra:
        #    extra = questions_extra[i]
        #    results_item[questions_extra_attr] = extra

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files): # use generator (yield) to read vectors in multiple files
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i] # 'wiki'
        with open(file, "rb") as reader:
            # doc_vectors = pickle.load(reader) # List[(id_str,(hidden_size)numpy vector)]
            collection_pids, collection_embs = pickle.load(reader) # (shard_size),(shard_size,n_prompt,hidden_size)
            yield (collection_pids, collection_embs)


def generate_dense_embeddings_openqa(
    args, eval_dataset, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    delete_model=False, add_title=True, tokenizer=None, get_emb_lambda=None, **kwargs,
):
    # assert eval_dataset in ["squad1","nq","trivia"]

    model.eval()

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval")

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
    accelerator.wait_for_everyone()

    # encoding passages
    dense_index_path = os.path.join(work_dir, f"dense_index_{args.shard_id}.pkl")  # collection_pids, collection_embs
    
    if file_exists(dense_index_path):
        logger.info("%s already exists." % dense_index_path)
        return
    
    with accelerator.main_process_first():
        passages_dataset = CsvCtxSrc(
            "train", args.data_dir, None, args, tokenizer, add_title=add_title, enable_shard=True) # data_type="train" is useless
    
    # get all vectors for the sharded collections
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

        pids, embs = accelerator.gather(pids), accelerator.gather(embs) # (bsz*world_size), (bsz*world_size,n_prompt,hidden_size)

        if accelerator.is_local_main_process:
            pids_list.append(pids.cpu().numpy())
            passage_vectors_list.append(embs.detach().cpu().numpy().astype("float32"))
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        collection_pids = np.concatenate(pids_list, axis=0)[:len(passages_dataset)] # (shard_size)
        collection_embs = np.concatenate(passage_vectors_list, axis=0)[:len(passages_dataset)] # (shard_size,n_prompt,hidden_size)
        logger.info("Writing results to %s" % dense_index_path)
        with open(dense_index_path, "wb") as fp:
            pickle.dump([collection_pids, collection_embs], fp, protocol=4)
        logger.info("Total passages processed %d. Written to %s", len(collection_pids), dense_index_path)
    accelerator.wait_for_everyone()


def evaluate_dense_retreival_openqa(
    args, eval_dataset, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="top-5", delete_model=False, add_title=True,
    tokenizer=None, faiss_mode="gpu",
    get_emb_lambda=None, hits=100,
    **kwargs,
):
    assert eval_dataset in ["squad1-test","nq-test","trivia-test",
        "squad1-dev","nq-dev","trivia-dev",
        "squad1-train","nq-train","trivia-train"] # include .csv for train and dev

    model.eval()

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval")

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
    accelerator.wait_for_everyone()

    all_qids, all_query_embs = None, None

    with accelerator.main_process_first():
        passages_dataset = CsvCtxSrc(
            "train", args.data_dir, None, args, tokenizer, add_title=add_title, enable_shard=False) # data_type="train" is useless
        queries_dataset = CsvQASrc(eval_dataset, args.data_dir, None, args, tokenizer, )

    # encode queries
    query_embs_path = os.path.join(work_dir, f"{eval_dataset}_query_embs.pkl")
    if not file_exists(query_embs_path):
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

            qids, embs = accelerator.gather(qids), accelerator.gather(embs) # (bsz*world_size), (bsz*world_size,hidden_size)
            if accelerator.is_local_main_process:
                qids_list.append(qids.cpu().numpy())
                query_embs_list.append(embs.detach().cpu().numpy().astype("float32"))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            all_qids = np.concatenate(qids_list, axis=0)[:len(queries_dataset)]
            all_query_embs = np.concatenate(query_embs_list, axis=0)[:len(queries_dataset)] # (q_size,hidden_size)
            with open(query_embs_path, "wb") as fp:
                pickle.dump([all_qids, all_query_embs], fp, protocol=4)
        # Beacause of DDP, all_query_embs is mis-ordered w.r.t. answers, (the mis-ordering is recoreded by all_qids). 
        # The subsequent retrieval result is in the same order as all_query_embs.
        # We will re-order the retrieval results with the help of all_qids.
        # Keeping the initial orders is beneficial because we could use DPR's code with minimal changes:
        # DPR's code aligns retrieval results with answers by the initial query order, it doesn't rely on qids to align them.  
    accelerator.wait_for_everyone()

    if delete_model:  # delete model to free space for faiss
        free_model(model, accelerator)
    else:
        model.train()
    free_memory()

    # do dense retrieval
    if accelerator.is_local_main_process:
        # get passages
        # dense_index_path_pattern = os.path.join(work_dir, f"dense_index_[0-{args.num_shards-1}].pkl") # 0,1,4
        dense_index_path_pattern = os.path.join(work_dir, f"dense_index_*.pkl")
        # assert file_exists(dense_index_path)
        vector_files = glob.glob(dense_index_path_pattern) # [file_names]
        vector_files.sort()

        with open(vector_files[0], "rb") as fp:
            _, collection_embs_first_shard = pickle.load(fp) # (shard_size),(shard_size,n_prompt,hidden_size)
        _,n_prompt_dim,dim = collection_embs_first_shard.shape
        actual_k = min(hits*n_prompt_dim,2048) if faiss_mode=='gpu' else hits*n_prompt_dim # 2048 is a limitation of gpu faiss
        logger.info(f"number of vectors per doc: {n_prompt_dim}, actual k for search: {actual_k}")

        # get queries  
        if all_qids is None or all_query_embs is None:
            with open(query_embs_path, "rb") as fp:
                all_qids, all_query_embs = pickle.load(fp) # (q_size),(q_size,hidden_size)

        # prepare resources
        logger.info(f"Using faiss_mode {faiss_mode} ...")
        if faiss_mode == "cpu":
            pass
        elif faiss_mode == "gpu":
            ngpus = faiss.get_num_gpus()
            logger.info(f"found {ngpus} gpus in faiss ...")
            if ngpus==1:
                res = faiss.StandardGpuResources() 
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
                    temp_memory = -1 # int(1024*1024*1024)
                    if temp_memory >= 0:
                        res.setTempMemory(temp_memory)
                    gpu_resources.append(res)
                vres = faiss.GpuResourcesVector()
                vdev = faiss.IntVector()
                for i in range(ngpus):
                    vres.push_back(gpu_resources[i])
                    vdev.push_back(i)
        else:
            raise NotImplementedError(faiss_mode)

        def _batch_search(_index, _queries, _batch_size, _k):
            Ds, Is = [], []
            for _start in tqdm(range(0, _queries.shape[0], _batch_size)):
                D, I = _index.search(_queries[_start: _start + _batch_size], k=_k)
                Ds.append(D) # (_batch_size,_k) or (last_batch_size,_k)
                Is.append(I) # (_batch_size,_k) or (last_batch_size,_k)
            return np.concatenate(Ds, axis=0), np.concatenate(Is, axis=0) # (q_size,_k),(q_size,_k)

        # search shards
        result_heap = faiss.ResultHeap(nq=len(all_qids), k=actual_k, keep_max=True)
        pids_list = []
        t00 = time.time()
        for shard_id, (collection_pids_shard, collection_embs_shard) in enumerate(iterate_encoded_files(vector_files, 
            path_id_prefixes=None)): # (shard_size),(shard_size,n_prompt,hidden_size)
            
            i0 = sum([len(i) for i in pids_list])
            pids_list.append(collection_pids_shard)
            i1 = sum([len(i) for i in pids_list])
            
            # build index
            t0 = time.time()
            logger.info(f"Building faiss index for shard {vector_files[shard_id]} ...")
            if faiss_mode=='cpu':
                index_engine = faiss.IndexFlatIP(dim)
            if faiss_mode == "gpu":
                index_flat = faiss.IndexFlatIP(dim)
                if ngpus==1: 
                    index_engine = faiss.index_cpu_to_gpu(res, 0, index_flat)
                else:
                    index_engine = faiss.index_cpu_to_gpu_multiple(vres,vdev,index_flat,cloner_options)
            collection_embs_shard.resize(len(collection_pids_shard)*n_prompt_dim,dim) # (shard_size*n_prompt,hidden_size)
            index_engine.add(collection_embs_shard)
            logger.info(f"Using {time.time() - t0}sec to build index for doc {i0}-{i1} ({i1-i0}) in shard {vector_files[shard_id]}")
            
            # search
            t0 = time.time()
            logger.info(f"Doing faiss search for shard {vector_files[shard_id]} ...")
            assert index_engine.is_trained
            Di, Ii = _batch_search(index_engine, all_query_embs, 
                _batch_size=64 if faiss_mode=='gpu' else 1024, _k=actual_k) # (q_size,hits*n_prompt_dim),(q_size,hits*n_prompt_dim)
            result_heap.add_result(D=Di, I=Ii+i0*n_prompt_dim)  # i0 is an offset
            del index_engine
            logger.info(f"Using {time.time() - t0}sec to complete search for {all_query_embs.shape[0]} queries in shard {vector_files[shard_id]}")
        
        result_heap.finalize()
        D = result_heap.D # (q_size,hits*n_prompt_dim)
        I = result_heap.I # (q_size,hits*n_prompt_dim)
        collection_pids = np.concatenate(pids_list, axis=0) # (p_size)
        logger.info(f"Total time used to search all shards for {all_query_embs.shape[0]} queries: {time.time() - t00}sec")

        # mapping search results to pids
        search_results = [None]*len(all_qids)
        t00 = time.time()
        for q_index, qid in tqdm(enumerate(all_qids), desc="Calculating metrics ..."):
            top_pids = []
            top_scores = []
            for i,p_index in enumerate(I[q_index]):
                pid_in_hits = str(collection_pids[int(p_index)//n_prompt_dim])
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

            search_results[int(qid)] = (top_pids,top_scores) 
            # the query orders in `search_result` is the same as the original query order,
            # this is consistent with `queries_dataset.answers` and `queries_dataset.`
        logger.info(f"Time used to mapping/aggregating search results for {all_query_embs.shape[0]} queries: {time.time() - t00}sec")
   
        # calculate metrics
        all_passages = {pid: (text,title) for (pid,text,title) in passages_dataset.example_list}
        eval_metrics = validate(
            passages=all_passages, # {pid:(text,title)}
            answers=queries_dataset.answers, # List[List[str]]
            result_ctx_ids=search_results, # List[(List[docid],List[score])], for each doc, List[score] is in descending order
            workers_num=args.num_proc, # 16
            match_type='string' # 'string'
        ) 
        # print the values of the evaluated metrics:
        logger.info(f"step {global_step}: {eval_metrics}")

        search_result_filename = f"{eval_dataset}_search_result_hits{hits}.json"
        search_result_path = os.path.join(work_dir, search_result_filename)
        save_results(
            passages=all_passages, # {doc_id:(text,title)}
            questions=queries_dataset.questions, # List[str]
            answers=queries_dataset.answers, # List[List[str]]
            top_passages_and_scores=search_results, # List[(List[docid],List[score])]
            per_question_hits=eval_metrics['questions_doc_hits'], # List[List[True/False] of length 100]
            out_file= search_result_path
        )
        key_metric = eval_metrics[key_metric_name]
    else:
        key_metric = NEG_INF
        eval_metrics = {}
    accelerator.wait_for_everyone()
    return key_metric, eval_metrics