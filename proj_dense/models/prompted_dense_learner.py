
from peach.base import *

from peach.enc_utils.general import get_representation_tensor

from peach.enc_utils.enc_learners import LearnerMixin, PromptedLearnerMixin, PrefixedLearnerMixin #, PromptedLearnerMixinWithPermut
from transformers import AutoModel, BertForSequenceClassification

import torch
import torch.nn as nn

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs


class PromptedDenseLearner(PromptedLearnerMixin): # PromptedLearnerMixinWithPermut
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
        for param in self.encoder.parameters():
            param.requires_grad = False
        if self.query_encoder is not None and self.model_args.freeze_query_encoder:
            for param in self.query_encoder.parameters():
                param.requires_grad = False

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            training_mode=None, is_query=None,
            **kwargs,
    ):

        if training_mode is None: # "retrieval_finetune" for train; None for eval (rerank and full rank)
            if is_query:
                return get_representation_tensor(self.encoding_query(input_ids, attention_mask, )).contiguous() # (bsz,hidden_size)
            else:
                return get_representation_tensor(self.encoding_doc(input_ids, attention_mask, )).contiguous() # (bsz,n_prompt,hidden_size)

        dict_for_meta = {}

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

        emb_doc, emb_query = get_representation_tensor(doc_outputs), get_representation_tensor(query_outputs) 
        # (bsz*num_doc,n_prompt,hidden_size),(bsz,hidden_size)

        return emb_doc,emb_query # (bsz*num_doc,n_prompt,hidden_size),(bsz,hidden_size)

    def compute_loss(self, emb_doc, emb_query):
        # (bsz*num_doc,n_prompt,hidden_size), (bsz,hidden_size)
        # the inputs to this methods are full-sized, not chunked

        bsz = emb_query.size(0)
        num_docs = int(emb_doc.size(0)/bsz)

        dict_for_meta = {}
        dict_for_loss = {}

        sim_out_mask = None

        if self.model_args.do_xentropy:
            # dense_ib_similarities = self.calc_similarities(emb_query, emb_doc)/self.model_args.xentropy_temperature
            # # (bsz,num_ranks*bsz*num_doc)
            dense_ib_scores = self.calc_similarities(emb_query, emb_doc) # (bsz,bsz*num_doc*num_ranks,n_prompt)
            dense_ib_similarities,reg_target = torch.max(dense_ib_scores,dim=-1) # (bsz,bsz*num_doc*num_ranks)  
            dense_ib_similarities = dense_ib_similarities/self.model_args.xentropy_temperature # (bsz,num_ranks*bsz*num_doc)     
            xentropy_target = torch.arange(bsz, device=emb_doc.device, dtype=torch.long)*num_docs+ \
                              self.get_delta(bsz, num_docs)
            # (bsz), [i]=i*num_doc+rank*bsz*num_doc, 
            # positive doc is the first doc in the rank's block's the query's subblock
            self.calc_xentropy_loss_for_sims(
                dense_ib_similarities, xentropy_target, dict_for_meta, dict_for_loss,
                loss_name="dense", sim_out_mask=sim_out_mask)
                # cross-entropy loss stored in dict_for_loss["xentropy_dense_loss"]
                # loss coefficient stored in dict_for_meta["xentropy_dense_loss_weight"]
                # statitics stored in dict_for_meta["xentropy_dense_valid_ratio"]
            if self.model_args.xentropy_reg_loss_weight>0:
                dense_ib_scores = dense_ib_scores[range(bsz),xentropy_target].contiguous() # (bsz,bsz*num_doc*num_ranks,n_prompt)->(bsz,n_prompt)
                dense_ib_scores = dense_ib_scores/self.model_args.reg_temperature # (bsz,n_prompt)
                reg_target = reg_target[range(bsz),xentropy_target] # (bsz,bsz*num_doc*num_ranks)->(bsz)
                self.calc_xentropy_loss_for_reg(
                    dense_ib_scores, reg_target, dict_for_meta, dict_for_loss,
                    loss_name="reg",)

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
        ga_emb_doc = self.gather_tensor(emb_doc) 
        # (bsz*num_doc*num_ranks,n_prompt,hidden_size) or (bsz*num_doc,n_prompt,hidden_size) when not self.training
        scores = self.sim_fct(emb_query, ga_emb_doc.view(-1, emb_dim))
        # (bsz,bsz*num_doc*num_ranks*n_prompt) or (bsz,bsz*num_doc*n_prompt) when not self.training
        scores = scores.view(emb_query.shape[0],-1,n_prompt) 
        # (bsz,bsz*num_doc*num_ranks,n_prompt) or (bsz,bsz*num_doc,n_prompt) when not self.training
        return scores
        # return torch.max(scores,dim=-1)[0] # (bsz,bsz*num_doc*num_ranks) or (bsz,bsz*num_doc) when not self.training


    def calc_xentropy_loss_for_sims(
            self, similarities, target, dict_for_meta, dict_for_loss,
            loss_name="dense", sim_out_mask=None):
            # (bsz,num_ranks*bsz*num_doc), (bsz), (bsz,num_ranks*bsz*num_doc)
        dict_for_meta[f"xentropy_{loss_name}_loss_weight"] = getattr(
            self.model_args, f"xentropy_{loss_name}_loss_weight", 1.0)
        if sim_out_mask is not None:
            proc_similarities = self.mask_out_logits(similarities, sim_out_mask) # (bsz,num_ranks*bsz*num_doc)
            with torch.no_grad():  # logging
                dict_for_meta[f"xentropy_{loss_name}_valid_ratio"] = \
                    (1 - sim_out_mask).to(similarities.dtype).mean(dim=-1).mean().detach().item() # the percentage of how many docs are kept as valid
        else:
            proc_similarities = similarities # (bsz,num_ranks*bsz*num_doc)
        dict_for_loss[f"xentropy_{loss_name}_loss"] = nn.CrossEntropyLoss(reduction='mean')(proc_similarities, target)
        # (bsz,num_ranks*bsz*num_doc), (bsz) -> (bsz) -> (1)

        max_idxs = torch.max(proc_similarities, 1)[1] # (bsz)
        correct_predictions_count = (max_idxs == target).sum() # (1)
        dict_for_meta[f"correct_predictions_count"] = correct_predictions_count # (1)


    def calc_xentropy_loss_for_reg(
            self, similarities, target, dict_for_meta, dict_for_loss,
            loss_name="reg", sim_out_mask=None):
            # (bsz,n_prompt), (bsz), (bsz,num_ranks*bsz*num_doc)
        dict_for_meta[f"xentropy_{loss_name}_loss_weight"] = getattr(
            self.model_args, f"xentropy_{loss_name}_loss_weight", 1.0)
        if False: # sim_out_mask is not None:
            proc_similarities = self.mask_out_logits(similarities, sim_out_mask) # (bsz,num_ranks*bsz*num_doc)
            with torch.no_grad():  # logging
                dict_for_meta[f"xentropy_{loss_name}_valid_ratio"] = \
                    (1 - sim_out_mask).to(similarities.dtype).mean(dim=-1).mean().detach().item() # the percentage of how many docs are kept as valid
        else:
            proc_similarities = similarities # (bsz,n_prompt)
        dict_for_loss[f"xentropy_{loss_name}_loss"] = nn.CrossEntropyLoss()(proc_similarities, target)
        # (bsz,n_prompt), (bsz) -> (bsz) -> (1)

        dict_for_meta["argmax_prompt_index"] = target # (bsz)
        dict_for_meta["prompt_dist"] = torch.nn.functional.softmax(proc_similarities,dim=-1) # (bsz,n_prompt)

    def calc_distill_loss(
            self, source_similarities, target_similarities, # (bsz,num_doc), (bsz,num_doc)
            dict_for_meta=None, dict_for_loss=None, source_name=None, # "dense"
            target_name=None, sim_neg_mask=None, tau=1.0, # "reranker" 
    ):
        kldiv_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

        if sim_neg_mask is not None:
            source_similarities = self.mask_out_logits(source_similarities/tau, sim_neg_mask)
            target_similarities = self.mask_out_logits(target_similarities/tau, sim_neg_mask)

        source_logp = torch.log_softmax(source_similarities, dim=-1) # (bsz,num_doc)
        target_prob = torch.softmax(target_similarities, dim=-1) # (bsz,num_doc)

        loss = kldiv_fct(source_logp, target_prob) # (bsz)->(1)

        if source_name is not None and target_name is not None:
            loss_str = f"distill_{target_name}2{source_name}_loss"
            dict_for_meta[f"{loss_str}_weight"] = getattr(self.model_args, f"{loss_str}_weight", 1.0) * (tau ** 2)
            dict_for_loss[loss_str] = loss
        return loss # (1)

    def generate_bi_similarities_mask_by_reranker( # denoise negatives
            self, reranker_scores, do_in_batch=False, logits_margin=1.0): # (bsz,num_doc)
        # [bs, *], [bs, num_docs]
        with torch.no_grad():
            bsz, num_docs = reranker_scores.shape
            positive_scores = reranker_scores[:, :1]  # [bs,1] # (bsz,1)
            margin_tn_mask = (reranker_scores[:, 1:] < (positive_scores - logits_margin)).to(torch.long) # (bsz,num_doc-1)
            margin_mask = torch.cat(
                [torch.ones_like(positive_scores, dtype=torch.long), margin_tn_mask], dim=1).contiguous() # (bsz,num_doc) of {0,1} indicating the mask

            if do_in_batch:
                device = reranker_scores.device
                anchors = torch.arange(bsz, device=device, dtype=torch.long) * num_docs + self.get_delta(bsz, num_docs)  
                # (bsz), [0+rank*bsz*num_doc,num_doc+rank*bsz*num_doc,2*num_doc+rank*bsz*num_doc,...(bsz-1)*num_doc+rank*bsz*num_doc]
                idxs = anchors.unsqueeze(1) + torch.arange(num_docs, device=device, dtype=torch.long).unsqueeze(0) 
                # (bsz,1)+(1,num_doc)->(bsz,num_doc), broadcast [i][j] = i*num_doc+rank*bsz*num_doc+j
                ib_margin_mask = torch.ones([bsz, self.my_world_size * bsz * num_docs], device=device, dtype=torch.long)
                # (bsz,num_ranks*bsz*num_doc)
                ib_margin_mask = torch.scatter(ib_margin_mask, dim=1, index=idxs, src=margin_mask)
                # only modify the values for your rank block, your query's num_doc block
                # How can ib_margin_mask be updated from other processes? 
                # Only mask out negatives in each queries's own doc group, and adopt all other in-batch negatives?
                return ib_margin_mask # (bsz,num_ranks*bsz*num_doc)
            else:
                return margin_mask # (bsz,num_doc)

    def get_delta(self, bsz, num_docs): # compute each process's offest to index doc, used for in-batch negatives
        return dist.get_rank() * bsz * num_docs if self.my_world_size > 1 else 0
