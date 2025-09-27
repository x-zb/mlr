import os

import torch
from torch.utils.data.dataset import Dataset

from peach.common import file_exists, get_line_offsets, save_json, load_json, \
    load_list_from_file, load_tsv, http_get, save_pickle, load_pickle
import json
from tqdm import tqdm
import random
from transformers import AutoTokenizer
from peach.base import CustomArgs
import collections
from copy import copy
import torch.distributed as dist
import logging
import gzip, pickle

from peach.enc_utils.general import MAX_QUERY_LENGTH


class DatasetMarcoPassagesRanking(Dataset):
    DATA_TYPE_SET = set(["train", "aug"])
    LOAD_TYPE_SET = set(["memory", "disk"])

    # MSMARCO_PASSAGE_DEV_QRELS_FILENAME = "passage_ranking/qrels.dev.tsv"
    MSMARCO_PASSAGE_TRAIN_QRELS_FILENAME = "passage_ranking/qrels.{}.tsv"
    MSMARCO_PASSAGE_TRAIN_QUERY_FILENAME = "passage_ranking/{}.query.txt"

    # BM25 negative
    MSMARCO_PASSAGE_OFFICIAL_NEGATIVE_FILENAME = "passage_ranking/train.negatives.tsv"
    # Diverse negative

    # cross-enc scores
    CE_SCORES_FILENAME = 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
    SBERT_NEGATIVE_FILENAME = 'msmarco-hard-negatives.jsonl.gz'

    MSMARCO_PASSAGE_COLLECTION_FILENAME = "collection.tsv"

    def __init__(
            self, data_type, data_dir, load_type, data_args, tokenizer, add_title=True, **kwargs):
        assert data_type in self.DATA_TYPE_SET
        assert load_type is None or load_type in self.LOAD_TYPE_SET # 'memory'

        self.data_type = data_type # 'train'
        self.data_dir = data_dir
        self.load_type = load_type or "disk" # 'memory'
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_title = add_title

        self.num_negatives = data_args.num_negatives # 15
        # self.tie_encoder = hasattr(self.data_args, "tie_encoder") and self.data_args.tie_encoder

        # collection pre-process
        self.collection_path = os.path.join(self.data_dir, self.MSMARCO_PASSAGE_COLLECTION_FILENAME)

        if load_type == "disk":
            pid2offset_file_path = self.collection_path + ".pid2offset.json"
            if file_exists(pid2offset_file_path):
                pid2offset = dict((int(k), v) for k, v in load_json(pid2offset_file_path).items())
            else:
                offset_file_path = self.collection_path + ".offset.json"
                if file_exists(offset_file_path):
                    offsets = load_json(offset_file_path)
                else:
                    offsets = get_line_offsets(self.collection_path, encoding="utf-8")
                    save_json(offsets, offset_file_path)
                pid2offset = dict()
                with open(self.collection_path, encoding="utf-8") as fp:

                    for idx_line, line in enumerate(fp):
                        passage_id, _ = line.strip("\n").split("\t")
                        pid2offset[int(passage_id)] = offsets[idx_line]
                    assert len(pid2offset) == len(offsets)
                    save_json(pid2offset, pid2offset_file_path)
            self.collection_pid2offset = pid2offset
            self.collection_pid_list = list(self.collection_pid2offset.keys())
        else:
            pid_text = load_tsv(self.collection_path)
            self.collection_pid2text = dict((int(pid), text) for pid, text in pid_text) # Dict[int:str]
            self.collection_pid_list = list(self.collection_pid2text.keys()) # List[int]

        # collection title
        if self.add_title:
            pid_title = load_tsv(self.collection_path + ".title.tsv")
            self.pid2title = dict((int(pid), title) for pid, title in pid_title) # Dict[int:str]
        else:
            self.pid2title = None

        # load queries
        queries = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_TRAIN_QUERY_FILENAME.format(self.data_type))) # the txt file loaded as a tsv 
        self.qid2query = dict((int(qid), query_text) for qid, query_text in queries) # Dict[int:str]
        assert len(queries) == len(self.qid2query)

        qrels = load_tsv(os.path.join(self.data_dir, self.MSMARCO_PASSAGE_TRAIN_QRELS_FILENAME.format(self.data_type)))
        self.qid2pids = collections.defaultdict(set) # Dict[int:Set[int]]
        self.qrels = []  # one query multi positive # List[(int,int)]
        for qrel in qrels:
            assert len(qrel) == 4
            qid, pid = int(qrel[0]), int(qrel[2])
            self.qid2pids[qid].add(pid)
            self.qrels.append((qid, pid, ))

        # load negative samples
        self.qid2negatives = None # Dict{int:List[int]}, {qid:list of bm25/hard neg pids}
        self.qp2scores_distill = None  # for distillation

        self.example_list = self.qrels # List[(int,int)], (qid, pos_pid)

    def load_official_bm25_negatives(self, accelerator=None, keep_num_neg=None):
        # accelerator=None
        # keep_num_neg=args.num_negs_per_system=200
        qid_pids_list = load_tsv(
            os.path.join(self.data_dir, self.MSMARCO_PASSAGE_OFFICIAL_NEGATIVE_FILENAME))
        qid2pids = {}
        for qid, pids in qid_pids_list:
            neg_pids = [int(pid) for pid in pids.split(",")]
            if keep_num_neg is not None:
                neg_pids = neg_pids[:keep_num_neg]
            qid2pids[int(qid)] = neg_pids
        # qid2pids = dict((int(k), v) for k, v in load_json(os.path.join(self.data_args.output_dir, "tmp_qid2negatives.json")).items())
        
        self.use_new_qid2negatives(qid2pids) 
        # load to self.qid2negatives and filter self.example_list

        if accelerator is not None and dist.is_initialized():
            accelerator.wait_for_everyone()

    def load_sbert_hard_negatives(
            self, accelerator=None, ce_score_margin=3.0, num_negs_per_system=5, keep_num_neg=None,
            **kwargs, ):
        # # part 1: ce scores
        # ce_scores_file = os.path.join(self.data_dir, self.CE_SCORES_FILENAME)
        # if not file_exists(ce_scores_file):
        #     logging.info("Download cross-encoder scores file")
        #     http_get(
        #         'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz',
        #         ce_scores_file)
        # logging.info("Load CrossEncoder scores dict")
        # with gzip.open(ce_scores_file, 'rb') as fIn:
        #     ce_scores = pickle.load(fIn)
        # self.qp2scores_distill = ce_scores

        # print(list(self.qp2scores_distill.keys())[:10])
        # part 2: xxx
        hard_negatives_filepath = os.path.join(self.data_dir, self.SBERT_NEGATIVE_FILENAME)
        if not os.path.exists(hard_negatives_filepath):
            logging.info("Please download the sbert hard negatives file.")
            raise FileNotFoundError
            http_get(
                # 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz',
                "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz",
                hard_negatives_filepath)
        logging.info("Read hard negatives train file")

        qid2pids = {}
        negs_to_use = None
        with gzip.open(hard_negatives_filepath, 'rt', encoding='utf8') as fIn:
            for line in tqdm(fIn, total=502939):
                data = json.loads(line)

                # Get the positive passage ids
                pos_pids = [item['pid'] for item in data['pos']]
                pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
                ce_score_threshold = pos_min_ce_score - ce_score_margin

                # Get the hard negatives
                neg_pids = set()
                for system_negs in data['neg'].values(): # return all the values (w/o keys) in a list
                    negs_added = 0
                    for item in system_negs:
                        if item['ce-score'] > ce_score_threshold:
                            continue

                        pid = item['pid']
                        if pid not in neg_pids:
                            neg_pids.add(pid)
                            negs_added += 1
                            if negs_added >= num_negs_per_system:
                                break

                if len(pos_pids) > 0 and len(neg_pids) > 0:
                    # neg_pids = [str(x) for x in neg_pids]
                    # fout.write(str(data['qid']) + '\t' + ','.join(neg_pids) + '\n')
                    neg_pids = list(neg_pids)[:keep_num_neg] # 4 systems * 5 negs per system = 20 negs, less than 200
                    qid2pids[int(data['qid'])] = [int(pid) for pid in neg_pids]

        self.use_new_qid2negatives(qid2pids)
        if accelerator is not None and dist.is_initialized():
            accelerator.wait_for_everyone()

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
            (qid, pid, ) for (qid, pid, ) in self.qrels if qid in self.qid2negatives] # List[qid:pid], filter out qids not in self.qid2negatives

    def __len__(self):
        return len(self.example_list)

    def sample_negatives(self, negatives, num_negatives): # List[pid], int
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

    def __getitem__(self, item):
        qid, pos_pid = self.example_list[item]

        # true_positives = self.qid2pids[qid]
        # negative sampling
        if self.qid2negatives is None:
            neg_pids = random.choices(self.collection_pid_list, k=self.num_negatives)
        else:
            negatives = self.qid2negatives[qid] # List[pid]
            if len(negatives) > 0 and isinstance(negatives[0], tuple) and isinstance(negatives[0][0], str):
                # negatives[0] is int for bm25 negatives
                assert self.num_negatives % len(negatives) == 0
                neg_pids = []
                for neg_name, negs in negatives:
                    neg_pids.extend(self.sample_negatives(negs, self.num_negatives//len(negatives)))
            else:
                neg_pids = self.sample_negatives(negatives, self.num_negatives) # List[pid] of length self.num_negatives

        # read dataset
        query = self.qid2query[qid]

        all_passages = [] # List[str], 1 positive + self.num_negatives negative passages
        with open(self.collection_path, encoding="utf-8") as fp:
            for pid in [pos_pid, ] + neg_pids:
                if self.load_type == "disk":
                    fp.seek(self.collection_pid2offset[pid])
                    line = fp.readline()
                    passage_id, passage = line.strip("\n").split("\t")
                    assert int(passage_id) == pid
                    all_passages.append(passage)
                else:
                    all_passages.append(self.collection_pid2text[pid])
        all_text = all_passages # List[str], 1 positive + self.num_negatives negative passages

        if self.add_title and self.pid2title is not None:
            all_titles = []
            for pid in [pos_pid, ] + neg_pids:
                all_titles.append(self.pid2title[pid])
            # all_text = [self.tokenizer.sep_token.join([t, p, ]) for t, p in zip(all_titles, all_passages, )]
            all_text = [(t, p) for t, p in zip(all_titles, all_passages, )] # List[Tuple(title,passage)], for 1 positive + self.num_negatives negative passages

        # distill_labels = None
        if self.qp2scores_distill is not None:
            distill_labels = [self.qp2scores_distill[qid][pid] for pid in [pos_pid, ] + neg_pids]
        else:
            distill_labels = None

        query_outputs = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=MAX_QUERY_LENGTH, truncation=True)
        if "token_type_ids" in query_outputs:
            query_outputs["token_type_ids"] = [1] * len(query_outputs["token_type_ids"])  # tbd
            # not reasonable query token_type_ids=1, but not used in the output

        passage_outputs = self.tokenizer(
            *zip(*all_text),
            add_special_tokens=True,
            max_length=self.data_args.max_length, truncation=True)

        feature_dict = {
                "input_ids": passage_outputs["input_ids"], # (1+NUM_NEG,passage_lengths)
                "attention_mask": passage_outputs["attention_mask"],
                "input_ids_query": query_outputs["input_ids"], # (query_length)
                "attention_mask_query": query_outputs["attention_mask"],
            }

        if distill_labels is not None:
            feature_dict["distill_labels"] = distill_labels

        return feature_dict








