
from peach.base import *

from peach.enc_utils.general import get_representation_tensor

from peach.enc_utils.enc_learners import T5LearnerMixin
from transformers import AutoModel, BertForSequenceClassification

import torch
import torch.nn as nn

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs



class T5DenseLearner(T5LearnerMixin):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None):
        super().__init__(config, model_args, tokenizer, encoder, query_encoder, )
        self.sim_fct = Similarity(metric="dot")

        self.reranker = None
        if model_args.distill_reranker: # pre-trained reranker path 
            reranker_config = AutoConfig.from_pretrained(
                model_args.distill_reranker)
            self.reranker = BertForSequenceClassification.from_pretrained(
                model_args.distill_reranker, config=reranker_config)

    def freeze_encoder(self):
        pass

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            training_mode=None, is_query=None,
            **kwargs,
    ):

        if training_mode is None: # "retrieval_finetune" for train; None for eval
            if is_query:
                return get_representation_tensor(self.encoding_query(input_ids, attention_mask, )).contiguous() # (bsz,hidden_size)
            else:
                return get_representation_tensor(self.encoding_doc(input_ids, attention_mask, )).unsqueeze(1).contiguous() # (bsz,1,hidden_size)

        # input reshape
        (high_dim_flag, num_docs), (org_doc_shape, org_query_shape), \
        (input_ids, attention_mask, token_type_ids, position_ids,), \
        (input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,) = preproc_inputs(
            input_ids, attention_mask, token_type_ids, position_ids,
            input_ids_query, attention_mask_query, token_type_ids_query, position_ids_query,) 
        # (bsz*num_doc,doc_len), (bsz,query_len)
        bsz = org_doc_shape[0]

        # encoding
        doc_outputs = self.encoding_doc(input_ids, attention_mask, return_dict=True)
        query_outputs = self.encoding_query(input_ids_query, attention_mask_query, return_dict=True)

        emb_doc, emb_query = get_representation_tensor(doc_outputs).unsqueeze(1), get_representation_tensor(query_outputs) 
        # (bsz*num_doc,1,hidden_size),(bsz,hidden_size)

        return emb_doc,emb_query # (bsz*num_doc,1,hidden_size),(bsz,hidden_size)

    
        
    def compute_loss(self, emb_doc, emb_query):
        # (bsz*num_doc,1,hidden_size), (bsz,hidden_size)
        # the inputs to this methods are full-sized, not chunked

        bsz = emb_query.size(0)
        num_docs = int(emb_doc.size(0)/bsz)
        
        dict_for_meta = {}
        dict_for_loss = {}

        if self.model_args.do_xentropy:
            dense_ib_similarities = self.calc_similarities(emb_query, emb_doc)/self.model_args.xentropy_temperature
            # (bsz*num_ranks,num_ranks*bsz*num_doc)
            xentropy_target = torch.arange(bsz, device=emb_doc.device, dtype=torch.long)*num_docs+ \
                              self.get_delta(bsz, num_docs)
            # (bsz), [i]=i*num_doc+rank*bsz*num_doc, 
            # positive doc is the first doc in the rank's block's the query's subblock
            ga_xentropy_target = self.gather_tensor(xentropy_target) # (bsz*num_ranks)
            self.calc_xentropy_loss_for_sims(
                dense_ib_similarities, ga_xentropy_target, dict_for_meta, dict_for_loss,
                loss_name="dense")
                # cross-entropy loss stored in dict_for_loss["xentropy_dense_loss"]
                # loss coefficient stored in dict_for_meta["xentropy_dense_loss_weight"]
                # statitics stored in dict_for_meta["xentropy_dense_valid_ratio"]
            dict_for_meta['dense_ib_similarities'] = dense_ib_similarities # (bsz*num_ranks,bsz*num_doc*num_ranks)
            dict_for_meta['xentropy_target'] = ga_xentropy_target # (bsz*num_ranks)

        loss = 0.
        for k in dict_for_loss:
            if k + "_weight" in dict_for_meta:
                if dict_for_meta[k + "_weight"] == 0.:
                    loss += 0.0  # save calc
                else:
                    loss += dict_for_meta[k + "_weight"] * dict_for_loss[k]
            else:
                loss += dict_for_loss[k]
        dict_for_loss["loss"] = loss
        # total loss stored in dict_for_loss["loss"]

        dict_for_meta.update(dict_for_loss) 
        
        return dict_for_meta # model output: dict_for_meta + dict_for_loss

    def calc_sims_without_inbatch(self, emb_query, emb_doc, num_docs): # (bsz,hidden_size), (bsz*num_doc,n_prompt,hidden_size)
        # emb_dim = emb_doc.shape[-1]
        # emb_doc = emb_doc.view(-1, num_docs, emb_dim) # (bsz,num_doc,hidden_size)
        # emb_query = emb_query.unsqueeze(1)  # (bsz,1,hidden_size)
        # return torch.sum(emb_doc * emb_query, dim=-1)  # (bsz,num_doc)
        _,n_prompt,hidden_size = emb_doc.shape
        emb_doc = emb_doc.view(-1,num_docs,n_prompt,hidden_size) # (bsz,num_doc,n_prompt,hidden_size)
        emb_query = emb_query.unsqueeze(1).unsqueeze(1)  # (bsz,hidden_size)->(bsz,1,1,hidden_size)
        scores = torch.sum(emb_doc*emb_query, dim=-1)  # (bsz,num_doc,n_prompt), broadcast, dot product
        return torch.max(scores,dim=-1)[0] # (bsz,num_doc)

    def calc_similarities(self,  emb_query, emb_doc): # (bsz,hidden_size), (bsz*num_doc,n_prompt,hidden_size)
        # emb_dim = emb_doc.shape[-1]
        # # method from LearnerMixin
        # ga_emb_doc = self.gather_tensor(emb_doc) # (bsz*num_doc*num_ranks,hidden_size) 
        # # gather_tensor to implement in-batch negatives
        # similarities = self.sim_fct(emb_query, ga_emb_doc.view(-1, emb_dim)) # (bsz,num_ranks*bsz*num_doc)
        # return similarities # (bsz,num_ranks*bsz*num_doc)
        _,n_prompt,emb_dim = emb_doc.shape
        ga_emb_doc = self.gather_tensor(emb_doc) # (bsz*num_doc*num_ranks,n_prompt,hidden_size)
        ga_emb_query = self.gather_tensor(emb_query) # (bsz*num_ranks,hidden_size) 
        scores = self.sim_fct(ga_emb_query, ga_emb_doc.view(-1, emb_dim))
        # (bsz*num_ranks,bsz*num_doc*num_ranks*n_prompt)
        scores = scores.view(ga_emb_query.shape[0],-1,n_prompt) 
        # (bsz*num_ranks,bsz*num_doc*num_ranks,n_prompt)
        return torch.max(scores,dim=-1)[0] # (bsz*num_ranks,bsz*num_doc*num_ranks)

    def calc_xentropy_loss_for_sims(
            self, similarities, target, dict_for_meta, dict_for_loss,
            loss_name="dense"):
            # (bsz*num_ranks,num_ranks*bsz*num_doc), (bsz*num_ranks)
        dict_for_meta[f"xentropy_{loss_name}_loss_weight"] = getattr(
            self.model_args, f"xentropy_{loss_name}_loss_weight", 1.0)
        dict_for_loss[f"xentropy_{loss_name}_loss"] = nn.CrossEntropyLoss(reduction='mean')(similarities, target)
        # (bsz*num_ranks,num_ranks*bsz*num_doc), (bsz*num_ranks) -> (bsz*num_ranks) -> (1)

        max_idxs = torch.max(similarities, 1)[1] # (bsz*num_ranks)
        correct_predictions_count = (max_idxs == target).sum() # (1)
        dict_for_meta[f"correct_predictions_count"] = correct_predictions_count # (1)


    def get_delta(self, bsz, num_docs): # compute each process's offest to index doc, used for in-batch negatives
        return dist.get_rank() * bsz * num_docs if self.my_world_size > 1 else 0
