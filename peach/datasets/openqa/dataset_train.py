import os
import logging
import random
import json
from typing import List, Iterator, Callable, Tuple

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

# from peach.enc_utils.general import MAX_QUERY_LENGTH

import torch.distributed as dist
from copy import copy
from peach.common import save_pickle, load_pickle

import csv


logger = logging.getLogger(__name__)

def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text

def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question

def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


class JsonQADataset(Dataset):
    DATA_TYPE_SET = set(["train", "train,train-hn", "dev", "train,train-hn-de", "train,train-hn-dm","train,train-hn-luyu","train,train-hn-de-mae","train,train-hn-dm-mae"])
    LOAD_TYPE_SET = set(["memory", "disk"])

    def __init__(self, task, data_type, data_dir, load_type, data_args, tokenizer, add_title=True,
        shuffle_positives: bool = False,
        normalize: bool = False,
        query_special_suffix: str = None,
        exclude_gold: bool = False,
        **kwargs):
        assert data_type in self.DATA_TYPE_SET
        assert load_type is None or load_type in self.LOAD_TYPE_SET

        self.data_type = data_type.split(',')
        self.data_dir = data_dir
        self.load_type = load_type or "disk" # unused
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_title = add_title

        self.num_negatives = data_args.num_negatives # 15
        # self.tie_encoder = hasattr(self.data_args, "tie_encoder") and self.data_args.tie_encoder

        assert task in ["squad1","nq","trivia"]
        self.files = [f'biencoder-{task}-{dt}.json' for dt in self.data_type]
        self.normalize = normalize
        self.exclude_gold = exclude_gold

        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix

        self.data_files = [os.path.join(self.data_dir,file) for file in self.files] # glob.glob(self.file)
        logger.info("Data files: %s", self.data_files)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        # self.data = self.data[:500]
        logger.info("Total cleaned data size: %d", len(self.data))



    def load_official_bm25_negatives(self, accelerator=None, keep_num_neg=None):
        pass

    def load_sbert_hard_negatives(
        self, accelerator=None, ce_score_margin=3.0, num_negs_per_system=8, negs_sources=None,**kwargs, ):
        pass

    def use_new_qid2negatives(self, qid2negatives, accelerator=None):
        # need `accelerator` for multi-processes
        if accelerator is None or (not dist.is_initialized()):
            self.qid2negatives = qid2negatives
        else:
            # use file to sync all processes
            tmp_path = os.path.join(self.data_args.output_dir, "tmp_qid2negatives.pkl")
            if accelerator.is_local_main_process:
                save_pickle(qid2negatives, tmp_path)
            accelerator.wait_for_everyone()
            self.qid2negatives = dict((int(k), v) for k, v in load_pickle(tmp_path).items()) # Dict[qid:List[pid]]
            accelerator.wait_for_everyone()
        self.example_list = [
            (qid, pid, ) for (qid, pid, ) in self.qrels if qid in self.qid2negatives] # List[qid:pid]

    def __len__(self):
        return len(self.data)

    def sample_negatives(self, negatives, num_negatives): # List[pid], int
        '''This method only samples pid, not text, therefore not directly applicable 
        to openqa data examples.'''
        try:
            if len(negatives) == 0:
                neg_pids = random.choices(self.collection_pid_list, k=num_negatives)
            elif len(negatives) == num_negatives:
                neg_pids = negatives
            elif len(negatives) < num_negatives:
                # neg_pids = negatives + random.choices(self.collection_pid_list, k=self.num_negatives-len(negatives))
                neg_pids = [negatives[i % len(negatives)] for i in range(num_negatives)]
            else:
                negatives_copy = copy(negatives)
                random.shuffle(negatives_copy)
                neg_pids = negatives_copy[:num_negatives]
        except KeyError:
            neg_pids = random.choices(self.collection_pid_list, k=num_negatives)
        return neg_pids # List[pid]

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix
        return query

    def __getitem__(self, index):
        ### dpr/data/biencoder_data.py: JsonQADataset().__init__()
       
        json_sample = self.data[index]

        query = self._process_query(json_sample["question"])

        positive_ctxs = json_sample["positive_ctxs"]  # 1
        if self.exclude_gold:
            ctxs = [ctx for ctx in positive_ctxs if "score" in ctx]
            if ctxs:
                positive_ctxs = ctxs

        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else [] # 50 for each question
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else [] # 90~100 for each question

        negative_ctxs = negative_ctxs[:self.data_args.num_negs_per_system]
        hard_negative_ctxs = hard_negative_ctxs[:self.data_args.num_negs_per_system]
        
        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        if self.normalize:
            for ctx in positive_ctxs+negative_ctxs+hard_negative_ctxs:
                ctx["text"] = normalize_passage(ctx["text"])
        
        ### dpr/models/biencoder.py: BiEncoder.create_biencoder_input()
        insert_title=self.add_title
        num_hard_negatives = self.num_negatives # 1
        num_other_negatives = 0
        shuffle = True
        # shuffle_positives = False
        hard_neg_fallback = True
        # query_token = None

        if shuffle and self.shuffle_positives: # False
            positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
        else:
            positive_ctx = positive_ctxs[0] # the one with the largest BM25 score?

        if shuffle: # True
            random.shuffle(negative_ctxs)
            random.shuffle(hard_negative_ctxs) # the hard_neg pool is shuffled for each query in each epoch 

        if hard_neg_fallback and len(hard_negative_ctxs) == 0:
            hard_negative_ctxs = negative_ctxs[0:num_hard_negatives]

        negative_ctxs = negative_ctxs[0:num_other_negatives] # num_other_negatives=0
        hard_negative_ctxs = hard_negative_ctxs[0:num_hard_negatives] # num_hard_negatives=1

        all_ctxs = [positive_ctx] + negative_ctxs + hard_negative_ctxs # only 2 ctxs

        ####################################
        query_outputs = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.data_args.max_length, # MAX_QUERY_LENGTH,
            truncation=True)
        if "token_type_ids" in query_outputs:
            query_outputs["token_type_ids"] = [0] * len(query_outputs["token_type_ids"])  # same as DPR

        passage_outputs = self.tokenizer(
            # tokenizer(text=(t1,t2,...),text_pair=(p1,p2,...)),
            # output_ids = [[[CLS],t1,[SEP],p1,[SEP]],[[CLS],t2,[SEP],p2,[SEP]],...], token_type_ids=[[0,0,0,1,1],[0,0,0,1,1],...]
            # not reasonable token_type_ids, but not used in the output
            # for T5Tokenizer, it will be [t1</s> p1</s>]
            text=[ctx['title'] for ctx in all_ctxs],
            text_pair=[ctx['text'] for ctx in all_ctxs],
            add_special_tokens=True,
            max_length=self.data_args.max_length, 
            truncation=True) if insert_title else self.tokenizer(
                text=[ctx['text'] for ctx in all_ctxs],
                add_special_tokens=True,
                max_length=self.data_args.max_length, 
                truncation=True) # default padding=False, trucation=False, add_special_tokens=True
            # if input is List[str] and padding=False, will return List[List[int]] for 'input_ids', 'attention_mask' and 'token_type_ids', 
            # and each inner list is of different lengths.
        if "token_type_ids" in passage_outputs:
            passage_outputs["token_type_ids"] = [0] * len(passage_outputs["token_type_ids"])  # same as DPR


        feature_dict = {
                "input_ids": passage_outputs["input_ids"], # (1+NUM_NEG,passage_lengths), Actually each passage's token ids is of different length, so it's not a matrix. The collate_fn will handle this and pad them to tensor.
                "attention_mask": passage_outputs["attention_mask"], # same as above
                "input_ids_query": query_outputs["input_ids"], # (query_length)
                "attention_mask_query": query_outputs["attention_mask"],
            }

        return feature_dict

