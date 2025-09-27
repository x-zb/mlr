import os
import csv
import math
import logging

from torch.utils.data.dataset import Dataset

from peach.datasets.openqa.dataset_train import normalize_passage, normalize_question

from peach.enc_utils.general import MAX_QUERY_LENGTH

logger = logging.getLogger(__name__)

class CsvCtxSrc(Dataset):

    def __init__(self, data_type, data_dir, data_names, data_args, 
    	tokenizer, add_title=True,
        enable_shard = False,
    	id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None, # "wiki",
    	normalize: bool = False):

        # self.data_type = data_type
        self.data_dir = data_dir
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.add_title = add_title

        self.file = os.path.join(self.data_dir, 'psgs_w100.tsv')

        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

        self.example_list = []

        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col] # str such as '1','2',...
                passage = row[self.text_col].strip('"')
                if self.normalize:
                    passage = normalize_passage(passage)
                self.example_list.append((sample_id,passage,row[self.title_col]))
        if enable_shard:
            self.shard_size = math.ceil(len(self.example_list) / data_args.num_shards)
            start_idx = data_args.shard_id * self.shard_size
            end_idx = min(start_idx + self.shard_size, len(self.example_list))

            logger.info(
                "Producing encodings for passages range: %d to %d (out of total %d)",
                start_idx,
                end_idx,
                len(self.example_list),
            )
            # shard_passages = all_passages[start_idx:end_idx]
            self.example_list = self.example_list[start_idx:end_idx]


    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, index):
        passage_id, passage, title = self.example_list[index]
        passage_id = int(passage_id)

        if self.add_title:
            if 't5' in self.data_args.learner_type: # for T5Tokenizer
                title_plus_passage = self.tokenizer.eos_token.join([title, passage])
            else: # for BertTokenizer
                title_plus_passage = self.tokenizer.sep_token.join([title, passage]) 
            passage_outputs = self.tokenizer(
                title_plus_passage,
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)  # "only_second"
        else:
            passage_outputs = self.tokenizer(
                passage,
                add_special_tokens=True,
                # return_offsets_mapping=True,
                max_length=self.data_args.max_length, truncation=True)

        return {
            "pids": passage_id,
            "input_ids": passage_outputs["input_ids"],
            "attention_mask": passage_outputs["attention_mask"],
            # "token_type_ids": passage_outputs["token_type_ids"],
        }
        # pid in self.example_list is a str, which is consistent with DPR
        # pid in self.__getitem__() is an int, which can be handled by accelerator.gather()


class CsvQASrc(Dataset): # csv datasets only have questions and answers

    def __init__(self, data_type, data_dir, data_names, data_args, tokenizer, num_dev=None,
    	question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        special_query_token: str = None,
        query_special_suffix: str = None):

        self.data_type = data_type
        self.data_dir = data_dir
        self.data_args = data_args
        self.tokenizer = tokenizer

        self.question_col = question_col # 0
        self.answers_col = answers_col # 1
        self.id_col = id_col # -1
        self.special_query_token = special_query_token # None
        self.query_special_suffix = query_special_suffix # None

        assert self.data_type in ["squad1-test","nq-test","trivia-test",
            "squad1-dev","nq-dev","trivia-dev",
            "squad1-train","nq-train","trivia-train"]
        self.file = os.path.join(self.data_dir, f'{self.data_type}.qa.csv')
        self.example_list = []
        self.answers = [] # List[List[str]] of answers in the original order
        self.questions =[] # List[str] in the original order
        with open(self.file) as ifile: 
            reader = csv.reader(ifile, delimiter="\t")
            count_q = 0 # *-test-qa.csv does not have a title row
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col]) # '[...]'(str) to [...](list) 
                qid = count_q
                if self.id_col >= 0:
                    qid = row[self.id_col]
                self.example_list.append((qid, self._process_question(question)))
                self.answers.append(answers)
                self.questions.append(question)
                count_q += 1

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(self.query_special_suffix):
            question += self.query_special_suffix
        return question

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, index):
        qid, query = self.example_list[index]

        query_outputs = self.tokenizer(
            query,
            add_special_tokens=True,
            # return_offsets_mapping=True,
            max_length=self.data_args.max_length,# MAX_QUERY_LENGTH, 
            truncation=True)
        # query_outputs["token_type_ids"] = [0] * len(query_outputs["token_type_ids"])

        return {
            "qids": qid, # qid is int, which can be handled by accelerator.gather(), as well as be used as indices to re-order retrieval results
            "input_ids_query": query_outputs["input_ids"],
            "attention_mask_query": query_outputs["attention_mask"],
            # "token_type_ids_query": query_outputs["token_type_ids"],
        }
