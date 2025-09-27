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

# from peach.datasets.openqa.dataset_eval import CsvCtxSrc #, CsvQASrc
from peach.datasets.openqa.dataset_train import read_data_from_json_files
# from peach.datasets.openqa.dataset_train_hn import JsonQADatasetWithQID
# from peach.enc_utils.eval_openqa.qa_validation import calculate_matches
# from peach.enc_utils.eval_openqa.eval_dense import validate


def get_hard_negative_by_dense_retrieval_openqa(
    args, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
    key_metric_name="top-5", delete_model=False, add_title=True,
    tokenizer=None, faiss_mode="gpu",
    get_emb_lambda=None, hits=100,
    **kwargs,
):
    
    eval_dataset = args.task+'-'+args.prediction_source # nq/squad1/trivia-train
    # assert eval_dataset in ["squad1-test","nq-test","trivia-test","squad1-dev","nq-dev","trivia-dev"]

    abs_output_dir = os.path.abspath(args.output_dir)
    work_dir = os.path.join(abs_output_dir, "dense_retrieval")
    search_result_filename = f"{eval_dataset}_search_result_hits{hits}.json"
    search_result_path = os.path.join(work_dir, search_result_filename)
    
    if accelerator.is_local_main_process:
        
        # generate new training set's json file
        t00 = time.time()
        search_results = read_data_from_json_files([search_result_path])

        train_with_hn = []
        num_questions_with_positive_ctxs = 0
        for item in tqdm(
            search_results, disable=not accelerator.is_local_main_process,
            total=len(search_results), desc=f"Generating new training set ..."):
            # a search_results entry {'question','answers','ctxs' (top 100 list of {'id','title','text','score','has_answer'})}
            instance = {
                # 'dataset': item['dataset'],
                'question': item['question'],
                'answers': item['answers'],
                'positive_ctxs': [],
                'negative_ctxs': [], # num_other_negatives=0, so not used in JsonQADataset
                'hard_negative_ctxs': []
            }
            for passage in item['ctxs']:
                if passage['has_answer']: 
                    # here no need to refer to the original json train file for gold passages,
                    # because these passages have already been checked against the gold answer.
                    # if `has_answer` is True, the passage contains the gold answer and is a positive passage.
                    positive_psg = {
                        'title': passage['title'], 
                        'text': passage['text'], 
                        'score': float(passage['score']), # convert np.float32 to float
                        'title_score': 0,
                        'passage_id' if args.task=='nq' else 'psg_id': passage['id'],
                        'has_answer': passage['has_answer']
                    }
                    instance['positive_ctxs'].append(positive_psg)
                else:
                    negative_psg = {
                        'title': passage['title'], 
                        'text': passage['text'], 
                        'score': float(passage['score']), # convert np.float32 to float
                        'title_score': 0,
                        'passage_id' if args.task=='nq' else 'psg_id': passage['id'],
                        'has_answer': passage['has_answer']
                    }
                    instance['hard_negative_ctxs'].append(negative_psg)

            if len(instance['positive_ctxs'])>0:
                num_questions_with_positive_ctxs += 1
            if len(instance['hard_negative_ctxs'])==0:
                logger.info(f'Warning: question "{item["question"]}" has no hard_negatives nor negative_ctxs.')
                # only a warning but still include questions with no hard negatives in the trainset, where a seq of [PAD] will be used during training
            train_with_hn.append(instance)
        logger.info(f'Total questions in hn trainset: {num_questions_with_positive_ctxs}')
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
            new_train_filename = f'biencoder-{args.task}-{args.prediction_source}-hn-ukn.json' # e.g., biencoder-squad1-train-hn-unk.json
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