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

from peach.datasets.openqa.dataset_eval import CsvCtxSrc #, CsvQASrc
from peach.datasets.openqa.dataset_train_hn import JsonQADatasetWithQID
# from peach.enc_utils.eval_openqa.qa_validation import calculate_matches
from peach.enc_utils.eval_openqa.eval_dense import validate

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



def get_hard_negative_by_dense_retrieval_openqa(
    args, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="top-5", delete_model=False, add_title=True,
    tokenizer=None, faiss_mode="gpu",
    get_emb_lambda=None, hits=100,
    **kwargs,
):
    
    eval_dataset = args.task+'-'+args.prediction_source # nq/squad1/trivia-train
    # assert eval_dataset in ["squad1-test","nq-test","trivia-test","squad1-dev","nq-dev","trivia-dev"]

    model.eval()

    get_emb_lambda = get_representation_tensor if get_emb_lambda is None else get_emb_lambda

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval") # same dir as retrieval

    if accelerator.is_local_main_process:
        if not dir_exists(work_dir):
            os.mkdir(work_dir)
    accelerator.wait_for_everyone()

    all_qids, all_query_embs = None, None

    with accelerator.main_process_first():
        passages_dataset = CsvCtxSrc(
            "train", args.data_dir, None, args, tokenizer, add_title=add_title, enable_shard=False) # data_type="train" is useless
        # queries_dataset = CsvQASrc(eval_dataset, args.data_dir, None, args, tokenizer, )
        queries_dataset = JsonQADatasetWithQID(
            args.task, args.prediction_source, args.data_dir, args.data_load_type, args, 
            tokenizer, add_title=(not args.no_title))

    # queries
    query_embs_path = os.path.join(work_dir, f"{eval_dataset}_query_embs.pkl")
    if not file_exists(query_embs_path):
        queries_dataloader = setup_eval_dataloader(args, queries_dataset, accelerator, use_accelerator=True)
        qids_list, query_embs_list = [], []
        for batch_idx, batch in tqdm(
                enumerate(queries_dataloader), disable=not accelerator.is_local_main_process,
                total=len(queries_dataloader), desc=f"Getting query vectors ..."):
            # batch's entries are {"qids","input_ids_query","attention_mask_query"}
            # each is padded and batched
            qids = batch.pop("qids")
            for k in list(batch.keys()):
                if k.endswith("_query"):
                    batch[k[:-6]] = batch.pop(k) # turn "input_ids/attention_mask_query" to "input_ids/attention_mask"

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
        logger.info(f"Time used to map/aggregate search results for {all_query_embs.shape[0]} queries: {time.time() - t00}sec")
   
        # calculate metrics on train set!
        all_passages = {pid: (text,title) for (pid,text,title) in passages_dataset.example_list}
        all_answers_in_qid_order = [item['answers'] for item in queries_dataset.data]
        eval_metrics = validate(
            passages=all_passages, # {pid:(text,title)}
            answers=all_answers_in_qid_order, # queries_dataset.answers, # List[List[str]]
            result_ctx_ids=search_results, # List[(List[docid],List[score])], for each doc, List[score] is in descending order
            workers_num=args.num_proc, # 16
            match_type='string' # 'string'
        ) # `eval_metrics` is in the same order as `search_results`, which is qid's order.
        logger.info(f"eval_metrics: top-5: {eval_metrics['top-5']}, top-20: {eval_metrics['top-20']}, top-100: {eval_metrics['top-100']}")

        # generate new training set's json file
        t00 = time.time()
        train_with_hn = []
        # all_passages = {pid: (text,title) for (pid,text,title) in passages_dataset.example_list} # pid is a string
        for item in tqdm(
            queries_dataset.data, disable=not accelerator.is_local_main_process,
            total=len(queries_dataset.data), desc=f"Generating new training set ..."):
            # no longer true: queries_dataset.data already filtered out questions with no positive_ctxs
            instance = {
                'dataset': item['dataset'],
                'question': item['question'],
                'answers': item['answers'],
                'positive_ctxs': item['positive_ctxs'], # len>=0
                'negative_ctxs': item['negative_ctxs'], # num_other_negatives=0, so not used in JsonQADataset
                'hard_negative_ctxs': []
            }
            top_pids,top_scores = search_results[item['qid']] # top_pids are strings
            ###########################################################
            if len(item['positive_ctxs'])==0:
                for idx,(pid,score) in enumerate(zip(top_pids,top_scores)):
                    # logger.info(f'Instance {item["qid"]}, doc{idx}: {eval_metrics["questions_doc_hits"][item['qid']][idx]}')
                    if eval_metrics["questions_doc_hits"][item['qid']][idx] is True:
                        positive_psg = {
                            'title': all_passages[pid][1], 
                            'text': all_passages[pid][0], 
                            'score': float(score), # convert np.float32 to float
                            'title_score': 0,
                            'passage_id' if args.task=='nq' else 'psg_id': pid
                        }
                        instance['positive_ctxs'].append(positive_psg)
                logger.info(f'Instance {item["qid"]} originally has no positive_ctxs, {len(instance["positive_ctxs"])} positive_ctxs added.')
            ############################################################
            if args.task=='nq':
                positive_pids = [pos['passage_id'] for pos in instance['positive_ctxs']]
                # hard_negative_pids = [neg['passage_id'] for neg in item['hard_negative_ctxs']]
            else:
                positive_pids = [pos['psg_id'] for pos in instance['positive_ctxs']]
                # hard_negative_pids = [neg['psg_id'] for neg in item['hard_negative_ctxs']]
            for idx,(pid,score) in enumerate(zip(top_pids,top_scores)):
                if pid not in positive_pids: #+hard_negative_pids:
                    # if eval_metrics["questions_doc_hits"][item['qid']][idx] is not False:
                    #     logger.info(f'instance {len(train_with_hn)}, toppid {pid} is a negative_ctx but contains the answer.')
                    hard_negative = {
                        'title': all_passages[pid][1], 
                        'text': all_passages[pid][0], 
                        'score': float(score), # convert np.float32 to float
                        'title_score': 0,
                        'psg_id': pid
                    }
                    instance['hard_negative_ctxs'].append(hard_negative)
                else:
                    # assert eval_metrics["questions_doc_hits"][item['qid']][idx] is True
                    if eval_metrics["questions_doc_hits"][item['qid']][idx] is not True:
                        logger.info(f'instance {len(train_with_hn)}, toppid {pid} is a positive_ctx but does not contain the answer.')
                    logger.info(f'instance {len(train_with_hn)}, toppid {pid} in positive_ctxs, excluded from hard_negatives.')
            # if len(item['hard_negative_ctxs'])==0 and args.task=='trivia':
            #     instance['hard_negative_ctxs']=[] # for triviaqa, keep the question with no hard neg unchanged.
            if len(instance['hard_negative_ctxs'])==0:
                logger.info(f'Warning: instance {item["qid"]} has no hard_negatives; will use negative_ctxs instead.')
            ############################
            # instance['hard_negative_ctxs'] = instance['hard_negative_ctxs'][:30]
            ############################
            train_with_hn.append(instance)
        logger.info(f"Time used to generate training data with hard negatives for {len(train_with_hn)} instances: {time.time() - t00}sec")
        # cannot distinguish from args.model_type because dm also use vanilla for prediction
        if 'dmde' in args.model_name_or_path and 'mae' in args.model_name_or_path: 
            new_train_filename = f'biencoder-{args.task}-{args.prediction_source}-hn-dm-mae.json' # e.g., biencoder-squad1-train-hn-dm-mae.json
        elif 'de_' in args.model_name_or_path and 'mae' in args.model_name_or_path:
            new_train_filename = f'biencoder-{args.task}-{args.prediction_source}-hn-de-mae.json' # e.g., biencoder-squad1-train-hn-de-mae.json
        elif 'dmde' in args.model_name_or_path: 
            new_train_filename = f'biencoder-{args.task}-{args.prediction_source}-hn-dm.json' # e.g., biencoder-squad1-train-hn-dm.json
        elif 'de_' in args.model_name_or_path:
            new_train_filename = f'biencoder-{args.task}-{args.prediction_source}-hn-de.json' # e.g., biencoder-squad1-train-hn-de.json
        else:
            new_train_filename = f'biencoder-{args.task}-{args.prediction_source}-hn-ukn.json' # e.g., biencoder-squad1-train-hn
        new_train_filepath = os.path.join(os.path.abspath(args.data_dir), new_train_filename)
        with open(new_train_filepath, "w", encoding="utf-8") as f:
            logger.info("Saving data to file %s" % new_train_filepath)
            data = json.dump(train_with_hn,f)
        logger.info('Done')

        key_metric = NEG_INF
        eval_metrics = {}
    else:
        key_metric = NEG_INF
        eval_metrics = {}
    accelerator.wait_for_everyone()
    return key_metric, eval_metrics