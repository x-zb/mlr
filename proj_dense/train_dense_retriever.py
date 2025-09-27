import json
import os.path
import math

import tqdm

from peach.base import *

from peach.datasets.marco.dataset_marco_passages import DatasetMarcoPassagesRanking
from peach.datasets.marco.dataset_marco_eval import DatasetRerank, DatasetCustomRerank

from peach.enc_utils.eval_functions import evaluate_encoder_reranking
from peach.enc_utils.eval_dense import evaluate_dense_retreival
from peach.enc_utils.hn_gen_dense import get_hard_negative_by_dense_retrieval
from peach.enc_utils.general import get_representation_tensor

from transformers import AutoModel, BertModel, T5EncoderModel

import torch
import torch.nn as nn

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs
from peach.common import load_pickle, dir_exists

from proj_dense.models import (
    DenseLearner,   
    DeepMatchDenseLearner,
    MEDenseLearner,
    ColBERTDenseLearner,
    PromptedDenseLearner, 
    PrefixedDenseLearner, 
    MVRDenseLearner,
    T5DenseLearner,
    T5DeepMatchDenseLearner,
    T5MEDenseLearner,
)

# for openqa
from peach.datasets.openqa.dataset_train import JsonQADataset
from peach.datasets.openqa.dataset_eval import CsvCtxSrc, CsvQASrc
from peach.enc_utils.eval_openqa.eval_functions import validate_nll, validate_average_rank
from peach.enc_utils.eval_openqa.eval_dense import generate_dense_embeddings_openqa, evaluate_dense_retreival_openqa
from peach.enc_utils.eval_openqa.hn_gen_dense import get_hard_negative_by_dense_retrieval_openqa


from proj_dense.grad_cache.grad_cache import GradCache
    


def train(args, train_dataset, model, accelerator, tokenizer, eval_dataset=None, eval_fn=None):
    if accelerator.is_local_main_process:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        tb_writer = None

    train_dataloader = setup_train_dataloader(args, train_dataset, accelerator)
    model, optimizer, lr_scheduler = setup_opt(args, model, accelerator, len(train_dataloader))
    grad_cache = GradCache(model,args.chunk_size)

    logging_berfore_training(args, train_dataset)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process) # global counter
    global_step = 0 # global counter
    step_loss = 0. # unused below
    step_loss_dict = defaultdict(float) # initialization, will be updated in each step and emptied after each param update step
    best_metric = NEG_INF # global
    best_epoch = -1
    ma_dict = MovingAverageDict()
    
    model.train()
    model.zero_grad()

    for epoch in range(args.num_train_epochs):
        
        if args.val_av_rank_start_epoch is not None and epoch>args.val_av_rank_start_epoch:
            args.dev_key_metric = 'average_rank'

        for step, batch in enumerate(train_dataloader):
            step += 1 
            sync_context = model.no_sync if accelerator.distributed_type != accelerate.DistributedType.NO and \
                                            step % args.gradient_accumulation_steps > 0 else nullcontext

            with sync_context():  # disable DDP sync for accumulation step
                outputs = grad_cache.cache_step(accelerator=accelerator, no_sync_except_last=True, **batch) # dict_for_meta  
            # update
            for key in outputs:
                if key.endswith("loss"):
                    step_loss_dict[key] += outputs[key].item()/args.gradient_accumulation_steps 
                    # store the loss divided by acc_steps in the step_loss_dict 
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader):

                model_update_wrt_gradient(args, accelerator, model, optimizer, lr_scheduler)

                # update loss for logging
                if accelerator.is_local_main_process:
                    if tb_writer is not None and (  # local main process
                            args.tensorboard_steps > 0 and global_step % args.tensorboard_steps == 0):
                        for key, loss_val in step_loss_dict.items():
                            tb_writer.add_scalar(f"training-{key}", loss_val, global_step)
                        for key, elem in outputs.items():
                            if (key not in step_loss_dict) and isinstance(elem, (int, float)):
                                tb_writer.add_scalar(f"training-meta-{key}", elem, global_step)
                    # log fire
                    ma_dict(step_loss_dict)
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logger.info(f"Log at step-{global_step}: {ma_dict.get_val_str()}")
                step_loss_dict = defaultdict(float)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(
                        args, accelerator, args.output_dir, model, tokenizer, args, save_specified_module='encoder' if 'beir' in args.output_dir else None,)


                # DDP for eval
                if eval_fn is not None and args.eval_steps > 0 and (global_step % args.eval_steps == 0 
                    or global_step == args.max_train_steps-1) and args.evaluation_strategy=='step':
                    key_metric, eval_metrics = eval_fn(
                        args, eval_dataset, model, accelerator, global_step=global_step,
                        tb_writer=tb_writer, tokenizer=tokenizer, key_metric_name=args.dev_key_metric,
                        similarity_metric=None, # query_encoder=None,
                        use_accelerator=True,
                    )
                    if 'loss' in args.dev_key_metric or args.dev_key_metric=='average_rank':
                        key_metric = -key_metric
                    # save the model to output_dir when get a new best metric at the this eval_step
                    if key_metric >= best_metric:  # always false in sub process
                        best_metric = key_metric
                        save_model_with_default_name(
                            args, accelerator, args.output_dir, model, tokenizer, args, save_specified_module='encoder' if 'beir' in args.output_dir else None,)
                accelerator.wait_for_everyone()  # other subprocesses must wait the main process
                    
                progress_bar.update(1)
                global_step += 1  # update global step after determine whether eval

                if global_step >= args.max_train_steps:
                    break
        if args.evaluation_strategy=='epoch' and eval_fn is not None:
            if args.train_set_evaluation:
                _ = eval_fn(
                    args, train_dataset, model, accelerator, global_step=global_step,
                    tb_writer=tb_writer, tokenizer=tokenizer, key_metric_name=args.dev_key_metric,
                    similarity_metric=None, # query_encoder=None,
                    use_accelerator=True, train_set_evaluation=True,
                )
            key_metric, eval_metrics = eval_fn(
                args, eval_dataset, model, accelerator, global_step=global_step,
                tb_writer=tb_writer, tokenizer=tokenizer, key_metric_name=args.dev_key_metric,
                similarity_metric=None, # query_encoder=None,
                use_accelerator=True,
            )
            if 'loss' in args.dev_key_metric or args.dev_key_metric=='average_rank':
                key_metric = -key_metric
            # save the model to output_dir when get a new best metric at the this eval_step
            if key_metric >= best_metric:  # always false in sub processes
                best_metric = key_metric
                best_epoch = epoch
                save_model_with_default_name(
                    args, accelerator, args.output_dir, model, tokenizer, args, save_specified_module='encoder' if 'beir' in args.output_dir else None,) # will also save the cluster model
            accelerator.wait_for_everyone()
    # save the model to "last_checkpoint" after all the epochs
    save_model_with_default_name(
        args, accelerator, os.path.join(args.output_dir, "last_checkpoint"), model, tokenizer, args, save_specified_module='encoder' if 'beir' in args.output_dir else None,)
    accelerator.print(f"best epoch: {best_epoch}, best metric: {best_metric}")
    model.zero_grad()
    model.eval()
    accelerator.wait_for_everyone()

def custom_hard_negative_preparation(args, ):
    assert args.negs_sources.startswith("custom")
    assert args.negs_source_paths is not None # /hard_neg_marco_passages/co-stg1.pkl

    all_group_names = ["dense", "sparse", "uni"]
    group_name_to_index = {"dense": 0, "sparse": 1, "uni": 2}
    if False: # args.split_negs:
        str_sources = args.negs_source_paths.strip("|").split("|")
        qid2negatives = dict()
        for str_source in str_sources:
            if str_source.startswith("dense:"):
                group_name = "dense"
                paths = str_source[6:].strip(";").split(";")
            elif str_source.startswith("sparse:"):
                group_name = "sparse"
                paths = str_source[7:].strip(";").split(";")
            elif str_source.startswith("uni:"):
                group_name = "uni"
                paths = str_source[7:].strip(";").split(";")
            else:
                raise AttributeError(str_sources, str_source)

            for pkl_path in paths:
                local_qid2negatives = load_pickle(pkl_path)
                for qid, lc_negs in local_qid2negatives.items():
                    qid = int(qid)
                    if qid not in qid2negatives:
                        qid2negatives[qid] = [(gn, []) for gn in all_group_names] if args.split_negs else []
                    if args.split_negs: #
                        qid2negatives[qid][group_name_to_index[group_name]][1].extend(lc_negs[:args.num_negs_per_system])
                    else:
                        qid2negatives[qid].extend(lc_negs[:args.num_negs_per_system])
        # got qid2negatives!
    else:  # default
        qid2negatives = dict()
        neg_filepath_list = args.negs_source_paths.strip(";").split(";") # List[file_path]
        for neg_filepath in neg_filepath_list:
            for qid, neg_pids in load_pickle(neg_filepath).items():
                if qid not in qid2negatives:
                    qid2negatives[qid] = []
                qid2negatives[qid].extend(neg_pids[:args.num_negs_per_system]) # we only have one neg_source_path
    return qid2negatives # Dict[qid:List[pid]]

def main():
    parser = argparse.ArgumentParser()
    
    define_hparams_training(parser)

    parser.add_argument("--do_xentropy", action="store_true")
    parser.add_argument("--xentropy_dense_loss_weight", type=float, default=1.0) # used default
    parser.add_argument("--xentropy_temperature", type=float, default=1.0) # use default

    # not used
    parser.add_argument("--distill_reranker", type=str, default=None) # the pre-trained reranker path
    parser.add_argument("--distill_reranker_margin", type=float, default=None)
    parser.add_argument("--distill_reranker_tau", type=float, default=1.0)
    parser.add_argument("--distill_reranker2dense_loss_weight", type=float, default=1.0)


    parser.add_argument("--data_load_type", type=str, default="disk", choices=["disk", "memory"])
    parser.add_argument("--data_dir", type=str, default=USER_HOME + "/ws/data/set/")
    parser.add_argument("--train_source", type=str, default="train", 
        choices=["train","train,train-hn","train,train-hn-de","train,train-hn-dm",
        "train,train-hn-luyu","train,train-hn-de-mae","train,train-hn-dm-mae"])
    
    parser.add_argument("--num_negatives", type=int, default=7)  
    parser.add_argument("--num_negs_per_system", type=int, default=8, help="The size of the neg pool for sampling `num_negatives` negs during training.")
    parser.add_argument("--negs_sources", type=str, default=None)
    parser.add_argument("--negs_source_paths", type=str, default=None)
    parser.add_argument("--split_negs", action="store_true") # not used

    parser.add_argument("--num_dev", type=int, default=500) # not used
    parser.add_argument("--dev_type", type=str, default="dev", ) # not used
    parser.add_argument("--dev_key_metric", type=str, default="loss") # default="top-5"

    parser.add_argument("--no_title", action="store_true") # use default
    parser.add_argument("--encoder_type", type=str, default=None) # use default

    parser.add_argument("--untie_query_encoder", action="store_true")

    parser.add_argument("--eval_reranking_source", type=str, default=None) # not used
    parser.add_argument("--prediction_source", type=str, default="dev", )
    parser.add_argument("--hits_num", type=int, default=1000)

    # for hard negative sampling
    parser.add_argument("--do_hn_gen", action="store_true")
    parser.add_argument("--hn_gen_num", type=int, default=1000)

    # newly added
    parser.add_argument("--faiss_mode", type=str, default=None, choices=["cpu","gpu"])
    parser.add_argument("--learner_type", type=str, default='vanilla', 
        choices=['vanilla', 'dm', 'me', 'colbert', 't5-vanilla', 't5-dm', 't5-me',
            'prompted','prefixed','mvr'])
    parser.add_argument("--n_prompt", default=1, type=int, help="The number of vectors/prompts per doc.")
    parser.add_argument("--me_num", default=None, type=int, help="The number of embeddings for ME model.")
    parser.add_argument("--layers", default=None, type=str, help="Layer indices (in descending order) for DeepMatch.")
    parser.add_argument("--pooling", default=None, type=str, choices=["mean","hybride","scalar_mix","last_reg"])
    # for prefixed,prompted, and mvr
    parser.add_argument("--pre_seq_len", default=None, type=int, help="The length of prefix tokens.")
    parser.add_argument("--pre_seq_len_query", default=None, type=int, help="The length of prefix tokens for query encoder.")

    parser.add_argument("--xentropy_reg_loss_weight", default=0.0, type=float, help="")
    parser.add_argument("--reg_temperature", default=1.0, type=float, help="") # use default

    parser.add_argument("--task", type=str, default="marco", choices=["marco", "squad1","nq","trivia"])
    parser.add_argument("--evaluation_strategy", type=str, default="step", choices=["step", "epoch"])
    parser.add_argument("--do_encoding", action="store_true",help="Generate dense embeddings for the passage collection.")
    parser.add_argument("--shard_id", default=0, type=int, help="Shard id for encoding passage vectors.")
    parser.add_argument("--num_shards", default=10, type=int, help="The number of shards for encoding passage vectors.")
    
    parser.add_argument("--pretrained_learner_type", type=str, default=None, choices=['prompted','vanilla','prefixed','mvr'], help="Default to use the same learner type as the model") # not used
    
    # for gradient caching    
    parser.add_argument("--chunk_size", default=1, type=int, help="The size of sub-batch (on each device if DDP) for gradient caching.")

    parser.add_argument("--train_set_evaluation", action="store_true",help="To evaluate the dev metrics on training set during training. Used to study overfitting.")
    parser.add_argument("--val_av_rank_start_epoch", default=None, type=int, help="The epoch where we switch from NLL to Average Rank validation.")

    args = parser.parse_args()
    accelerator = setup_prerequisite(args)

    config, tokenizer = load_config_and_tokenizer(
        args, config_kwargs={
            # "problem_type": args.problem_type,
            # "num_labels": num_labels,
        })

    
    if 't5' in args.model_name_or_path and 't5' in args.learner_type:
        encoder_class = T5EncoderModel
    else:
        encoder_class = BertModel # AutoModel 

    learnerMapping = {        
        'vanilla': DenseLearner,
        'dm': DeepMatchDenseLearner,
        'me': MEDenseLearner,
        'colbert': ColBERTDenseLearner,
        't5-vanilla': T5DenseLearner,
        't5-dm': T5DeepMatchDenseLearner,
        't5-me': T5MEDenseLearner,
        'prompted': PromptedDenseLearner,
        'prefixed': PrefixedDenseLearner,
        'mvr': MVRDenseLearner
        }
    learner_class = learnerMapping[args.learner_type]

    if 'results' not in args.model_name_or_path: 
        # load from hub
        encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
        if args.untie_query_encoder:
            query_encoder = encoder_class.from_pretrained(args.model_name_or_path, config=config)
        else:
            query_encoder = None
        model = learner_class(config, args, tokenizer, encoder, query_encoder=query_encoder)
    elif args.pretrained_learner_type is not None and args.pretrained_learner_type != args.learner_type:
        # load from 'results/' with different learner classes
        logger.info("Loading pre-trained model from a different learner type ...")
        
        pre_encoder = encoder_class.from_config(config) if args.encoder_type is None else encoder_class(config)
        if args.untie_query_encoder:
            pre_query_encoder = encoder_class.from_config(config) if args.encoder_type is None else encoder_class(config)
        else:
            pre_query_encoder = None
        pre_model = learnerMapping[args.pretrained_learner_type].from_pretrained(args.model_name_or_path, 
            config=config, model_args=args, tokenizer=tokenizer, encoder=pre_encoder, query_encoder=pre_query_encoder)
        model = learner_class(config, args, tokenizer, pre_model.encoder, query_encoder=pre_model.query_encoder)
    else: # load from 'results/' with the same learner class
        encoder = encoder_class(config)
        if args.untie_query_encoder:
            query_encoder = encoder_class(config)
        else:
            query_encoder = None
        model = learner_class.from_pretrained(args.model_name_or_path, 
            config=config, model_args=args, tokenizer=tokenizer, encoder=encoder, query_encoder=query_encoder)

    if args.do_train:
        if args.task=='marco':
            with accelerator.main_process_first():
                train_dataset = DatasetMarcoPassagesRanking(
                        args.train_source, args.data_dir, args.data_load_type, args, tokenizer, add_title=(not args.no_title)) 
            with accelerator.main_process_first():
                dev_dataset = None
            if args.negs_sources == "official": # stage 1 use official BM25 negatives
                train_dataset.load_official_bm25_negatives(keep_num_neg=args.num_negs_per_system, )
            elif args.negs_sources.startswith("custom"): # stage 2 use hard negatives from stage 1
                qid2negatives = custom_hard_negative_preparation(args) # Dict[qid:List[pid]]
                train_dataset.use_new_qid2negatives(qid2negatives, accelerator=None)
            else: # negatives from external source
                assert args.negs_sources == "sbert"
                train_dataset.load_sbert_hard_negatives(
                    ce_score_margin=3,# args.ce_score_margin,
                    num_negs_per_system=5,
                    keep_num_neg=args.num_negs_per_system)
            eval_fn = None
        else: # for 'openqa' tasks
            with accelerator.main_process_first():
                train_dataset = JsonQADataset(
                    args.task, args.train_source, args.data_dir, args.data_load_type, args, 
                    tokenizer, add_title=(not args.no_title))
            with accelerator.main_process_first():
                dev_dataset = JsonQADataset(
                    args.task, "dev", args.data_dir, args.data_load_type, args, 
                    tokenizer, add_title=(not args.no_title))
            eval_fn = validate_nll
        train(args, train_dataset, model, accelerator, tokenizer,eval_dataset=dev_dataset, eval_fn=eval_fn) 
            

    if args.do_eval or args.do_encoding or args.do_prediction or args.do_hn_gen:
        model = accelerator.prepare(model)

        meta_best_str = ""
        if args.do_eval: # only for MSMARCO
            with accelerator.main_process_first(): 
                if args.eval_reranking_source is None:
                    dev_dataset = DatasetRerank(
                        "dev", args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title))
                else:
                    dev_dataset = DatasetCustomRerank(
                        args.dev_type, args.data_dir, "memory", args, tokenizer, num_dev=None, add_title=(not args.no_title),
                        filepath_dev_qid2top1000pids=args.eval_reranking_source,
                    )
            # evaluate on triple data, no need of faiss
            best_dev_result, best_dev_metric = evaluate_encoder_reranking( 
                args, dev_dataset, model, accelerator, global_step=None,
                save_prediction=True, tokenizer=tokenizer, key_metric_name=args.dev_key_metric,
                similarity_metric=None)
            if accelerator.is_local_main_process:
                meta_best_str += json.dumps(best_dev_metric) + os.linesep
        else:
            best_dev_result = None

        if args.do_encoding: # only used for openqa; for marco, encoding is integrated in do_prediction's `evaluate_dense_retreival` function.
            generate_dense_embeddings_openqa(
                args, args.prediction_source, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                delete_model=False, add_title=(not args.no_title),tokenizer=tokenizer)


        if args.do_prediction:
            # using faiss to build index for large scale retrieval
            retrieval_fn = evaluate_dense_retreival if args.task=='marco' else evaluate_dense_retreival_openqa
            best_pred_result, dev_pred_metric = retrieval_fn(
                args, args.task+'-'+args.prediction_source, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                # key_metric_name=args.dev_key_metric, 
                delete_model=False, add_title=(not args.no_title),
                tokenizer=tokenizer, faiss_mode=args.faiss_mode, # "gpu",
                hits=args.hits_num,
            )
            

        if accelerator.is_local_main_process:
            with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
                fp.write(f"{best_dev_result}, {meta_best_str}")

        if args.do_hn_gen:
            if args.task=='marco': # for marco, hn_gen includes evaluate on the train set
                get_hard_negative_by_dense_retrieval(
                    args, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                    key_metric_name="MRR@10", delete_model=False, add_title=(not args.no_title),
                    tokenizer=tokenizer, faiss_mode=args.faiss_mode, # "gpu",
                    hits=args.hn_gen_num,
                )
            else: # for openqa, hn_gen only generate hn training files, so a previous evaluation on the train set should be conducted.
                get_hard_negative_by_dense_retrieval_openqa(
                    args, model, accelerator, global_step=None, tb_writer=None, save_prediction=False,
                    key_metric_name=args.dev_key_metric, delete_model=False, add_title=(not args.no_title),
                    tokenizer=tokenizer, faiss_mode=args.faiss_mode, # "gpu",
                    hits=args.hn_gen_num,
                )


if __name__ == '__main__':
    main()
