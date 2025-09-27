import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from peach.enc_utils.sim_metric import Similarity
from peach.enc_utils.general import preproc_inputs

from transformers.modeling_utils import PreTrainedModel

from proj_dense.modeling_cos import BertModelCos
from peach.enc_utils.general import get_representation_tensor
from functools import partial

import random
import numpy as np

import ast
from peach.base import logger

class FLOPS(nn.Module):
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)


class LearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

    @property
    def encoding_doc(self):
        return self.encoder

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # for gradient propogation
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class MELearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.me_num = model_args.me_num

        if self.model_args.pooling=='scalar_mix':
            self.scalar_parameters = nn.Parameter(torch.zeros(self.me_num), requires_grad=True)


    @property
    def encoding_doc(self):
        return partial(self._encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    def _encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        outputs = bert_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            
        sent_embeds = outputs["last_hidden_state"][:,:self.me_num,:].contiguous() # (bsz,me_num,hidden_size)

        if self.model_args.pooling=='mean':
            sent_embeds = torch.mean(sent_embeds,dim=1,keepdim=True) # (bsz,1,hidden_size)
        elif self.model_args.pooling=='scalar_mix':
            normed_weights = F.softmax(self.scalar_parameters,dim=0) # (num_layers)->(num_layers)
            sent_embeds = torch.matmul(normed_weights,sent_embeds).unsqueeze(1)
            # (num_layers)->(1,num_layers)->(1,1,num_layers)*(bsz,num_layers,hidden_size) = (bsz,hidden_size) -> (bsz,1,hidden_size)
        return {'sentence_embedding':sent_embeds} # (bsz,me_num,hidden_size) or (bsz,1,hidden_size)

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # for gradient propogation
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1



class DeepMatchLearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.layer_indices = ast.literal_eval(model_args.layers) # list
        logger.info(f'layers {self.layer_indices}, # = {len(self.layer_indices)}')

        if self.model_args.pooling=='scalar_mix':
            self.scalar_parameters = nn.Parameter(torch.zeros(len(self.layer_indices)), requires_grad=True)

    @property
    def encoding_doc(self):
        return partial(self._encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    def _encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        outputs = bert_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            # hidden_states is a Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
        sent_emb_list = [outputs['hidden_states'][i][:,0,:] for i in self.layer_indices] # (bsz,hidden_size)
        sent_embeds = torch.stack(sent_emb_list,dim=1).contiguous() # (bsz,num_layers,hidden_size)
        if self.model_args.pooling=='mean':
            sent_embeds = torch.mean(sent_embeds,dim=1,keepdim=True) # (bsz,1,hidden_size)
        elif self.model_args.pooling=='scalar_mix':
            normed_weights = F.softmax(self.scalar_parameters,dim=0) # (num_layers)->(num_layers)
            sent_embeds = torch.matmul(normed_weights,sent_embeds).unsqueeze(1)
            # (num_layers)->(1,num_layers)->(1,1,num_layers)*(bsz,num_layers,hidden_size) = (bsz,hidden_size) -> (bsz,1,hidden_size)
        return {'sentence_embedding':sent_embeds} # (bsz,num_layers,hidden_size) or (bsz,1,hidden_size)

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # for gradient propogation
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

class T5DeepMatchLearnerMixin(PreTrainedModel):
    # use mean pooling for T5 encoder to get the sentence embedding
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.layer_indices = ast.literal_eval(model_args.layers) # list
        logger.info(f'layers {self.layer_indices}, # = {len(self.layer_indices)}')

        if self.model_args.pooling=='scalar_mix':
            self.scalar_parameters = nn.Parameter(torch.zeros(len(self.layer_indices)), requires_grad=True)

    @property
    def encoding_doc(self):
        return partial(self._encoding,t5_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return partial(self._encoding_query,t5_encoder=self.encoder)
        return partial(self._encoding_query,t5_encoder=self.query_encoder)

    def _encoding(self, input_ids, attention_mask, t5_encoder, return_dict=True): # (bsz,seq_len)
        outputs = t5_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            # hidden_states is a Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
        
        def average_pool(sent_embeds): # (bsz,seq_len,hidden_size)
            sent_embeds = torch.sum(sent_embeds*(attention_mask.unsqueeze(2).expand(*sent_embeds.shape)),dim=1,keepdim=True) # (bsz,1,hidden_size)
            doc_lengths = torch.sum(attention_mask,dim=1) # (bsz,seq_len)->(bsz)
            doc_lengths = torch.max(doc_lengths,torch.ones(doc_lengths.shape[0],device=doc_lengths.device)) # (bsz)
            return (sent_embeds/(doc_lengths.view(-1,1,1))).squeeze(1) # (bsz,1,hidden_size)->(bsz,hidden_size)

        sent_emb_list = [average_pool(outputs['hidden_states'][i]) for i in self.layer_indices] # (bsz,hidden_size)
        sent_embeds = torch.stack(sent_emb_list,dim=1).contiguous() # (bsz,num_layers,hidden_size)
        if self.model_args.pooling=='mean':
            sent_embeds = torch.mean(sent_embeds,dim=1,keepdim=True) # (bsz,1,hidden_size)
        elif self.model_args.pooling=='scalar_mix':
            normed_weights = F.softmax(self.scalar_parameters,dim=0) # (num_layers)->(num_layers)
            sent_embeds = torch.matmul(normed_weights,sent_embeds).unsqueeze(1)
            # (num_layers)->(1,num_layers)->(1,1,num_layers)*(bsz,num_layers,hidden_size) = (bsz,hidden_size) -> (bsz,1,hidden_size)
        return {'sentence_embedding':sent_embeds} # (bsz,num_layers,hidden_size) or (bsz,1,hidden_size)

    def _encoding_query(self, input_ids, attention_mask, t5_encoder, return_dict=True): # (bsz,seq_len)
        outputs = t5_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            
        sent_embeds = outputs["last_hidden_state"] # (bsz,seq_len,hidden_size), in-batch padding

        # if self.model_args.pooling=='mean':
        sent_embeds = torch.sum(sent_embeds*(attention_mask.unsqueeze(2).expand(*sent_embeds.shape)),dim=1,keepdim=True) # (bsz,1,hidden_size)
        doc_lengths = torch.sum(attention_mask,dim=1) # (bsz,seq_len)->(bsz)
        doc_lengths = torch.max(doc_lengths,torch.ones(doc_lengths.shape[0],device=doc_lengths.device)) # (bsz)
        sent_embeds = (sent_embeds/(doc_lengths.view(-1,1,1))).squeeze(1) # (bsz,1,hidden_size)->(bsz,hidden_size)
        # For some questions in biencoder-trivia-train.json, hard_negative_ctxs=[] and negative_ctxs=[],
        # which will result in all 0 docs with length 0, which will result in 0-division error.
        # So we add a one to doc_lengths if the original value is 0
        return {'sentence_embedding':sent_embeds} # (bsz,hidden_size)

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # for gradient propogation
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class T5LearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

    @property
    def encoding_doc(self):
        return partial(self._encoding,t5_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return partial(self._encoding_query,t5_encoder=self.encoder)
        return partial(self._encoding_query,t5_encoder=self.query_encoder)

    def _encoding(self, input_ids, attention_mask, t5_encoder, return_dict=True): # (bsz,seq_len)
        outputs = t5_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            
        sent_embeds = outputs["last_hidden_state"] # (bsz,seq_len,hidden_size), in-batch padding

        # if self.model_args.pooling=='mean':
        sent_embeds = torch.sum(sent_embeds*(attention_mask.unsqueeze(2).expand(*sent_embeds.shape)),dim=1,keepdim=True) # (bsz,1,hidden_size)
        # logger.info(f'doc lengths: {input_ids},\n{attention_mask},\n {torch.sum(attention_mask,dim=1)}')
        doc_lengths = torch.sum(attention_mask,dim=1) # (bsz,seq_len)->(bsz)
        doc_lengths = torch.max(doc_lengths,torch.ones(doc_lengths.shape[0],device=doc_lengths.device)) # (bsz)
        sent_embeds = (sent_embeds/(doc_lengths.view(-1,1,1))).squeeze(1) # (bsz,1,hidden_size)->(bsz,hidden_size)
        # For some questions in biencoder-trivia-train.json, hard_negative_ctxs=[] and negative_ctxs=[],
        # which will result in all 0 docs with length 0, which will result in 0-division error.
        # So we add a one to doc_lengths if the original value is 0
        return {'sentence_embedding':sent_embeds} # (bsz,hidden_size)

    def _encoding_query(self, input_ids, attention_mask, t5_encoder, return_dict=True): # (bsz,seq_len)
        outputs = t5_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            
        sent_embeds = outputs["last_hidden_state"] # (bsz,seq_len,hidden_size), in-batch padding

        # if self.model_args.pooling=='mean':
        sent_embeds = torch.sum(sent_embeds*(attention_mask.unsqueeze(2).expand(*sent_embeds.shape)),dim=1,keepdim=True) # (bsz,1,hidden_size)
        # logger.info(f'doc lengths: {input_ids},\n{attention_mask},\n {torch.sum(attention_mask,dim=1)}')
        doc_lengths = torch.sum(attention_mask,dim=1) # (bsz,seq_len)->(bsz)
        doc_lengths = torch.max(doc_lengths,torch.ones(doc_lengths.shape[0],device=doc_lengths.device)) # (bsz)
        sent_embeds = (sent_embeds/(doc_lengths.view(-1,1,1))).squeeze(1) # (bsz,1,hidden_size)->(bsz,hidden_size)
        # For some questions in biencoder-trivia-train.json, hard_negative_ctxs=[] and negative_ctxs=[],
        # which will result in all 0 docs with length 0, which will result in 0-division error.
        # So we add a one to doc_lengths if the original value is 0
        return {'sentence_embedding':sent_embeds} # (bsz,hidden_size)

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor_same_size(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # for gradient propogation
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    def gather_tensor(self, target_tensor):
        # https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4
        if len(target_tensor.shape) in [1,2]:
            return self.gather_tensor_same_size(target_tensor)
        elif len(target_tensor.shape)==3: # (bsz,seq_len,hidden_size)
            # note that these tensors are already padded in each minibatch across instances,
            # here we do padding across minibatches
            target_tensor_shape = target_tensor.shape
            seq_lengths = torch.tensor([target_tensor_shape[1]], dtype=torch.int64, device=target_tensor.device) # (1)
            seq_lengths_ga = self.gather_tensor_same_size(seq_lengths) # (num_ranks)
            max_length = torch.max(seq_lengths_ga) # (1)
            padded_tensor = torch.zeros(target_tensor_shape[0],max_length,target_tensor_shape[2],dtype=target_tensor.dtype,device=target_tensor.device) # (bsz,max_len,hidden_size)
            padded_tensor[:,:target_tensor_shape[1],:] = target_tensor
            return self.gather_tensor_same_size(padded_tensor) # (bsz*num_ranks,max_len,hidden_size)

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class T5MELearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.me_num = model_args.me_num

        if self.model_args.pooling=='scalar_mix':
            self.scalar_parameters = nn.Parameter(torch.zeros(self.me_num), requires_grad=True)

    @property
    def encoding_doc(self):
        return partial(self._encoding,t5_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return partial(self._encoding_query,t5_encoder=self.encoder)
        return partial(self._encoding_query,t5_encoder=self.query_encoder)

    def _encoding(self, input_ids, attention_mask, t5_encoder, return_dict=True): # (bsz,seq_len)
        outputs = t5_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            
        sent_embeds = outputs["last_hidden_state"] # (bsz,seq_len,hidden_size), in-batch padding

        # if self.model_args.pooling=='mean':
        sent_embeds = torch.sum(sent_embeds*(attention_mask.unsqueeze(2).expand(*sent_embeds.shape)),dim=1,keepdim=True) # (bsz,1,hidden_size)
        # logger.info(f'doc lengths: {input_ids},\n{attention_mask},\n {torch.sum(attention_mask,dim=1)}')
        doc_lengths = torch.sum(attention_mask,dim=1) # (bsz,seq_len)->(bsz)
        doc_lengths = torch.max(doc_lengths,torch.ones(doc_lengths.shape[0],device=doc_lengths.device)) # (bsz)
        sent_embeds = sent_embeds/(doc_lengths.view(-1,1,1)) # (bsz,1,hidden_size)
        # For some questions in biencoder-trivia-train.json, hard_negative_ctxs=[] and negative_ctxs=[],
        # which will result in all 0 docs with length 0, which will result in 0-division error.
        # So we add a one to doc_lengths if the original value is 0
        
        sent_embeds = torch.cat([sent_embeds,outputs["last_hidden_state"][:,:self.me_num-1,:].contiguous()],dim=1) # (bsz,me_num,hidden_size)

        if self.model_args.pooling=='mean':
            sent_embeds = torch.mean(sent_embeds,dim=1,keepdim=True) # (bsz,1,hidden_size)
        elif self.model_args.pooling=='scalar_mix':
            normed_weights = F.softmax(self.scalar_parameters,dim=0) # (num_layers)->(num_layers)
            sent_embeds = torch.matmul(normed_weights,sent_embeds).unsqueeze(1)
            # (num_layers)->(1,num_layers)->(1,1,num_layers)*(bsz,num_layers,hidden_size) = (bsz,hidden_size) -> (bsz,1,hidden_size)
        
        # sent_embeds = outputs["last_hidden_state"][:,:self.me_num,:].contiguous() # (bsz,me_num,hidden_size)

        return {'sentence_embedding':sent_embeds} # (bsz,me_num,hidden_size) or (bsz,1,hidden_size)

    def _encoding_query(self, input_ids, attention_mask, t5_encoder, return_dict=True): # (bsz,seq_len)
        outputs = t5_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            
        sent_embeds = outputs["last_hidden_state"] # (bsz,seq_len,hidden_size), in-batch padding

        # if self.model_args.pooling=='mean':
        sent_embeds = torch.sum(sent_embeds*(attention_mask.unsqueeze(2).expand(*sent_embeds.shape)),dim=1,keepdim=True) # (bsz,1,hidden_size)
        # logger.info(f'doc lengths: {input_ids},\n{attention_mask},\n {torch.sum(attention_mask,dim=1)}')
        doc_lengths = torch.sum(attention_mask,dim=1) # (bsz,seq_len)->(bsz)
        doc_lengths = torch.max(doc_lengths,torch.ones(doc_lengths.shape[0],device=doc_lengths.device)) # (bsz)
        sent_embeds = (sent_embeds/(doc_lengths.view(-1,1,1))).squeeze(1) # (bsz,1,hidden_size)->(bsz,hidden_size)
        # For some questions in biencoder-trivia-train.json, hard_negative_ctxs=[] and negative_ctxs=[],
        # which will result in all 0 docs with length 0, which will result in 0-division error.
        # So we add a one to doc_lengths if the original value is 0
        return {'sentence_embedding':sent_embeds} # (bsz,hidden_size)

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor_same_size(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # for gradient propogation
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    def gather_tensor(self, target_tensor):
        # https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4
        if len(target_tensor.shape) in [1,2]:
            return self.gather_tensor_same_size(target_tensor)
        elif len(target_tensor.shape)==3: # (bsz,seq_len,hidden_size)
            # note that these tensors are already padded in each minibatch across instances,
            # here we do padding across minibatches
            target_tensor_shape = target_tensor.shape
            seq_lengths = torch.tensor([target_tensor_shape[1]], dtype=torch.int64, device=target_tensor.device) # (1)
            seq_lengths_ga = self.gather_tensor_same_size(seq_lengths) # (num_ranks)
            max_length = torch.max(seq_lengths_ga) # (1)
            padded_tensor = torch.zeros(target_tensor_shape[0],max_length,target_tensor_shape[2],dtype=target_tensor.dtype,device=target_tensor.device) # (bsz,max_len,hidden_size)
            padded_tensor[:,:target_tensor_shape[1],:] = target_tensor
            return self.gather_tensor_same_size(padded_tensor) # (bsz*num_ranks,max_len,hidden_size)

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class ColBERTLearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

    @property
    def encoding_doc(self):
        return partial(self._encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    def _encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        outputs = bert_encoder(
                input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                # position_ids=position_ids,
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)
            
        sent_embeds = outputs["last_hidden_state"] # (bsz,seq_len,hidden_size), in-batch padding

        if self.model_args.pooling=='mean':
            # sent_embeds = torch.mean(sent_embeds,dim=1,keepdim=True) # (bsz,1,hidden_size)
            sent_embeds = torch.sum(sent_embeds*(attention_mask.unsqueeze(2).expand(*sent_embeds.shape)),dim=1,keepdim=True) # (bsz,1,hidden_size)
            # logger.info(f'doc lengths: {input_ids},\n{attention_mask},\n {torch.sum(attention_mask,dim=1)}')
            doc_lengths = torch.sum(attention_mask,dim=1) # (bsz,seq_len)->(bsz)
            doc_lengths = torch.max(doc_lengths,torch.ones(doc_lengths.shape[0],device=doc_lengths.device)) # (bsz)
            sent_embeds = sent_embeds/(doc_lengths.view(-1,1,1)) # (bsz,1,hidden_size)
            # For some questions in biencoder-trivia-train.json, hard_negative_ctxs=[] and negative_ctxs=[],
            # which will result in all 0 docs with length 0, which will result in 0-division error.
            # So we add a one to doc_lengths if the original value is 0
        # scalar_mix not applicable
        # elif self.model_args.pooling=='scalar_mix':
        #     normed_weights = F.softmax(self.scalar_parameters,dim=0) # (num_layers)->(num_layers)
        #     sent_embeds = torch.matmul(normed_weights,sent_embeds).unsqueeze(1)
        #     # (num_layers)->(1,num_layers)->(1,1,num_layers)*(bsz,num_layers,hidden_size) = (bsz,hidden_size) -> (bsz,1,hidden_size)
        return {'sentence_embedding':sent_embeds} # (bsz,seq_len,hidden_size) or (bsz,1,hidden_size)

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor_same_size(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # for gradient propogation
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    def gather_tensor(self, target_tensor):
        # https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819/4
        if len(target_tensor.shape) in [1,2]:
            return self.gather_tensor_same_size(target_tensor)
        elif len(target_tensor.shape)==3: # (bsz,seq_len,hidden_size)
            # note that these tensors are already padded in each minibatch across instances,
            # here we do padding across minibatches
            target_tensor_shape = target_tensor.shape
            seq_lengths = torch.tensor([target_tensor_shape[1]], dtype=torch.int64, device=target_tensor.device) # (1)
            seq_lengths_ga = self.gather_tensor_same_size(seq_lengths) # (num_ranks)
            max_length = torch.max(seq_lengths_ga) # (1)
            padded_tensor = torch.zeros(target_tensor_shape[0],max_length,target_tensor_shape[2],dtype=target_tensor.dtype,device=target_tensor.device) # (bsz,max_len,hidden_size)
            padded_tensor[:,:target_tensor_shape[1],:] = target_tensor
            return self.gather_tensor_same_size(padded_tensor) # (bsz*num_ranks,max_len,hidden_size)

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class PromptedLearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.pre_seq_len = model_args.pre_seq_len # if model_args.prefix_tokens is None else len(model_args.prefix_tokens)
        self.prefix_tokens = [torch.arange(self.pre_seq_len).long() for _ in range(model_args.n_prompt)] # not the same as model_args.prefix_tokens, these are new tokens not in the vocab
        self.prefix_encoder = nn.ModuleList([torch.nn.Embedding(self.pre_seq_len, config.hidden_size) for _ in range(model_args.n_prompt)]) # excludes token_type and position embeddings    
        self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else None
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else None

    @property
    def encoding_doc(self):
        return partial(self._prompted_encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    def _prompted_encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        # encode doc
        batch_size = input_ids.shape[0]
        cls_embed = bert_encoder.embeddings.word_embeddings(torch.LongTensor([[self.cls_token_id]]*batch_size).to(input_ids.device)) # (bsz,1,hidden_size)
        # sep_embed = bert_encoder.embeddings.word_embeddings(torch.LongTensor([[self.sep_token_id]]*batch_size).to(input_ids.device)) # (bsz,1,hidden_size)
        raw_embed = bert_encoder.embeddings.word_embeddings(input_ids[:,1:]) # (bsz,seq_len-1,hidden_size), remove [CLS]
                

        # prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len+1).to(input_ids.device)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz,prefix_len+seq_len)

        sent_emb_list = []
        for p_idx in range(self.model_args.n_prompt):
            prefix_tokens = self.prefix_tokens[p_idx].unsqueeze(0).expand(batch_size, -1).to(input_ids.device) # (bsz,prefix_len)
            prompt_embed = self.prefix_encoder[p_idx](prefix_tokens) # (bsz,prefix_len,hidden_size), excludes token_type and position enbeddings
            # inputs_embeds = torch.cat([cls_embed,prompt_embed,sep_embed,raw_embed], dim=1) # (bsz,prefix_len+2+seq_len-1,hidden_size)
            inputs_embeds = torch.cat([cls_embed,prompt_embed,raw_embed], dim=1) # (bsz,prefix_len+1+seq_len-1,hidden_size)

            outputs = bert_encoder(
                # input_ids,
                attention_mask=attention_mask, # modified
                # token_type_ids=token_type_ids, # modified
                # position_ids=position_ids, # use default position ids in BertEmbedding
                # head_mask=head_mask,
                inputs_embeds=inputs_embeds, # modified
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                # past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,prefix_len+seq_len,hidden_size)

            sent_emb_list.append(get_representation_tensor(outputs)) # (bsz,hidden_size)
        sent_embeds = torch.stack(sent_emb_list,dim=1).contiguous() # (bsz,n_prompt,hidden_size)

        return {'sentence_embedding':sent_embeds} # (bsz,n_prompt,hidden_size)


    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # ?
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1



class PrefixedLearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.dropout = torch.nn.Dropout(encoder.config.hidden_dropout_prob)
        self.n_layer = encoder.config.num_hidden_layers
        self.n_head = encoder.config.num_attention_heads
        self.hidden_size = encoder.config.hidden_size
        assert self.hidden_size%self.n_head==0
        self.head_size = self.hidden_size//self.n_head

        self.pre_seq_len = model_args.pre_seq_len # if model_args.prefix_tokens is None else len(model_args.prefix_tokens)
        self.prefix_tokens = [torch.arange(self.pre_seq_len).long() for _ in range(model_args.n_prompt)] # not the same as model_args.prefix_tokens, these are new tokens not in the vocab
        # self.prefix_encoder = nn.ModuleList([torch.nn.Embedding(self.pre_seq_len, config.hidden_size) for _ in range(model_args.n_prompt)]) # includes token_type and position embeddings    
        self.prefix_encoder = nn.ModuleList([torch.nn.Embedding(self.pre_seq_len,self.n_layer*2*self.hidden_size) for _ in range(model_args.n_prompt)])
        # self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else None
        # self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else None
        
        self.pre_seq_len_query = model_args.pre_seq_len_query
        self.query_prefix_tokens = torch.arange(self.pre_seq_len_query).long()
        self.query_prefix_encoder = torch.nn.Embedding(self.pre_seq_len_query,self.n_layer*2*self.hidden_size)

    @property
    def encoding_doc(self):
        return partial(self._prompted_encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        return partial(self._query_prompted_encoding,bert_encoder=self.encoder)

    def _prompted_encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        # encode doc
        batch_size = input_ids.shape[0]
        
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz,prefix_len+seq_len)

        position_ids = [0]+list(range(self.pre_seq_len+1,self.pre_seq_len+input_ids.shape[-1])) # [CLS] would be in position 0
        position_ids = torch.tensor(position_ids).unsqueeze(0).expand(batch_size,-1).to(input_ids.device) # (bsz,seq_len)
        
        sent_emb_list = []
        for p_idx in range(self.model_args.n_prompt):
            prefix_tokens = self.prefix_tokens[p_idx].unsqueeze(0).expand(batch_size, -1).to(input_ids.device) # (bsz,prefix_len)
            past_key_values = self.prefix_encoder[p_idx](prefix_tokens) # (bsz,prefix_len,2*n_layers*hidden_size)
            past_key_values = past_key_values.view(batch_size,self.pre_seq_len,self.n_layer*2,self.n_head,self.head_size) # (bsz,prefix_len,2*n_layers,n_head,head_size)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2,0,3,1,4]).split(2,dim=0) # split_size_or_sections=2
            # (n_layers*2,bsz,n_heads,prefix_len,head_size)->(n_layers*(2,bsz,n_heads,prefix_len,head_size))

            outputs = bert_encoder(
                input_ids,
                attention_mask=attention_mask, # modified
                # token_type_ids=token_type_ids, # modified
                position_ids=position_ids, # modified
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds, # modified
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            # last_hidden_state is (bsz,seq_len,hidden_size)

            sent_emb_list.append(get_representation_tensor(outputs)) # (bsz,hidden_size), the first token is still [CLS]
        sent_embeds = torch.stack(sent_emb_list,dim=1).contiguous() # (bsz,n_prompt,hidden_size)

        return {'sentence_embedding':sent_embeds} # (bsz,n_prompt,hidden_size)

    def _query_prompted_encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        # encode doc
        batch_size = input_ids.shape[0]
        
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len_query).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz,prefix_len+seq_len)

        position_ids = [0]+list(range(self.pre_seq_len_query+1,self.pre_seq_len_query+input_ids.shape[-1])) # [CLS] would be in position 0
        position_ids = torch.tensor(position_ids).unsqueeze(0).expand(batch_size,-1).to(input_ids.device) # (bsz,seq_len)
        
        sent_emb_list = []
        
        prefix_tokens = self.query_prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(input_ids.device) # (bsz,prefix_len)
        past_key_values = self.query_prefix_encoder(prefix_tokens) # (bsz,prefix_len,2*n_layers*hidden_size)
        past_key_values = past_key_values.view(batch_size,self.pre_seq_len_query,self.n_layer*2,self.n_head,self.head_size) # (bsz,prefix_len,2*n_layers,n_head,head_size)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2,0,3,1,4]).split(2,dim=0) # split_size_or_sections=2
        # (n_layers*2,bsz,n_heads,prefix_len,head_size)->(n_layers*(2,bsz,n_heads,prefix_len,head_size))

        outputs = bert_encoder(
            input_ids,
            attention_mask=attention_mask, # modified
            # token_type_ids=token_type_ids, # modified
            position_ids=position_ids, # modified
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds, # modified
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            past_key_values=past_key_values,
        ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
        # last_hidden_state is (bsz,prefix_len+seq_len,hidden_size)
        return get_representation_tensor(outputs) # (bsz,hidden_size), the first token is still [CLS]

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # ?
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class DeepMatchWithPrefixLearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.dropout = torch.nn.Dropout(encoder.config.hidden_dropout_prob)
        self.n_layer = encoder.config.num_hidden_layers
        self.n_head = encoder.config.num_attention_heads
        self.hidden_size = encoder.config.hidden_size
        assert self.hidden_size%self.n_head==0
        self.head_size = self.hidden_size//self.n_head

        self.pre_seq_len = model_args.pre_seq_len # 1 
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len,self.n_layer*2*self.hidden_size)
        # self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else None
        # self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else None
        
        self.layer_indices = ast.literal_eval(model_args.layers) # list
        logger.info(f'layers {self.layer_indices}, # = {len(self.layer_indices)}')

        if self.model_args.pooling=='scalar_mix':
            self.scalar_parameters = nn.Parameter(torch.zeros(len(self.layer_indices)), requires_grad=True)

    @property
    def encoding_doc(self):
        return partial(self._prompted_encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    def _prompted_encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        # encode doc
        batch_size = input_ids.shape[0]
        
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz,prefix_len+seq_len)

        position_ids = [0]+list(range(self.pre_seq_len+1,self.pre_seq_len+input_ids.shape[-1])) # [CLS] would be in position 0
        position_ids = torch.tensor(position_ids).unsqueeze(0).expand(batch_size,-1).to(input_ids.device) # (bsz,seq_len)
        # it's necessary to manually set the position_ids here, otherwise BertEmbedding will automatically shift the position_ids by past_key_values_length 

        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(input_ids.device) # (bsz,prefix_len)
        past_key_values = self.prefix_encoder(prefix_tokens) # (bsz,prefix_len,2*n_layers*hidden_size)
        past_key_values = past_key_values.view(batch_size,self.pre_seq_len,self.n_layer*2,self.n_head,self.head_size) # (bsz,prefix_len,2*n_layers,n_head,head_size)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2,0,3,1,4]).split(2,dim=0) # split_size_or_sections=2
        # (n_layers*2,bsz,n_heads,prefix_len,head_size)->(n_layers*(2,bsz,n_heads,prefix_len,head_size))

        outputs = bert_encoder(
            input_ids,
            attention_mask=attention_mask, # modified
            # token_type_ids=token_type_ids, # modified
            position_ids=position_ids, # modified
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds, # modified
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            past_key_values=past_key_values,
        ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
        # last_hidden_state is (bsz,seq_len,hidden_size)
        # hidden_states is a Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
        sent_emb_list = [outputs['hidden_states'][i][:,0,:] for i in self.layer_indices] # (bsz,hidden_size), [CLS]'s hidden states
        sent_embeds = torch.stack(sent_emb_list,dim=1).contiguous() # (bsz,num_layers,hidden_size)
        if self.model_args.pooling=='mean':
            sent_embeds = torch.mean(sent_embeds,dim=1,keepdim=True) # (bsz,1,hidden_size)
        elif self.model_args.pooling=='scalar_mix':
            normed_weights = F.softmax(self.scalar_parameters,dim=0) # (num_layers)->(num_layers)
            sent_embeds = torch.matmul(normed_weights,sent_embeds).unsqueeze(1)
            # (num_layers)->(1,num_layers)->(1,1,num_layers)*(bsz,num_layers,hidden_size) = (bsz,hidden_size) -> (bsz,1,hidden_size)
        return {'sentence_embedding':sent_embeds} # (bsz,num_layers,hidden_size) or (bsz,1,hidden_size)


    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # backpropagate gradients
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class PrefixedMELearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.dropout = torch.nn.Dropout(encoder.config.hidden_dropout_prob)
        self.n_layer = encoder.config.num_hidden_layers
        self.n_head = encoder.config.num_attention_heads
        self.hidden_size = encoder.config.hidden_size
        assert self.hidden_size%self.n_head==0
        self.head_size = self.hidden_size//self.n_head

        self.pre_seq_len = model_args.pre_seq_len # if model_args.prefix_tokens is None else len(model_args.prefix_tokens)
        self.prefix_tokens = [torch.arange(self.pre_seq_len).long() for _ in range(model_args.n_prompt)] # not the same as model_args.prefix_tokens, these are new tokens not in the vocab
        # self.prefix_encoder = nn.ModuleList([torch.nn.Embedding(self.pre_seq_len, config.hidden_size) for _ in range(model_args.n_prompt)]) # includes token_type and position embeddings    
        self.prefix_encoder = nn.ModuleList([torch.nn.Embedding(self.pre_seq_len,self.n_layer*2*self.hidden_size) for _ in range(model_args.n_prompt)])
        # self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else None
        # self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else None
        
        self.pre_seq_len_query = model_args.pre_seq_len_query
        self.query_prefix_tokens = torch.arange(self.pre_seq_len_query).long()
        self.query_prefix_encoder = torch.nn.Embedding(self.pre_seq_len_query,self.n_layer*2*self.hidden_size)

        self.me_num = model_args.me_num

    @property
    def encoding_doc(self):
        return partial(self._prompted_encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        return partial(self._query_prompted_encoding,bert_encoder=self.encoder)

    def _prompted_encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        # encode doc
        batch_size = input_ids.shape[0]
        
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz,prefix_len+seq_len)

        position_ids = [0]+list(range(self.pre_seq_len+1,self.pre_seq_len+input_ids.shape[-1])) # [CLS] would be in position 0
        position_ids = torch.tensor(position_ids).unsqueeze(0).expand(batch_size,-1).to(input_ids.device) # (bsz,seq_len)
        
        sent_emb_list = []
        for p_idx in range(self.model_args.n_prompt):
            prefix_tokens = self.prefix_tokens[p_idx].unsqueeze(0).expand(batch_size, -1).to(input_ids.device) # (bsz,prefix_len)
            past_key_values = self.prefix_encoder[p_idx](prefix_tokens) # (bsz,prefix_len,2*n_layers*hidden_size)
            past_key_values = past_key_values.view(batch_size,self.pre_seq_len,self.n_layer*2,self.n_head,self.head_size) # (bsz,prefix_len,2*n_layers,n_head,head_size)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2,0,3,1,4]).split(2,dim=0) # split_size_or_sections=2
            # (n_layers*2,bsz,n_heads,prefix_len,head_size)->(n_layers*(2,bsz,n_heads,prefix_len,head_size))

            outputs = bert_encoder(
                input_ids,
                attention_mask=attention_mask, # modified
                # token_type_ids=token_type_ids, # modified
                position_ids=position_ids, # modified
                # head_mask=head_mask,
                # inputs_embeds=inputs_embeds, # modified
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                past_key_values=past_key_values,
            ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
            
            # last_hidden_state is (bsz,seq_len,hidden_size)
            sent_emb_list.append(outputs["last_hidden_state"][:,:self.me_num,:].contiguous()) # (bsz,me_num,hidden_size)
        
        sent_embeds = torch.cat(sent_emb_list,dim=1).contiguous() # (bsz,me_num*n_prompt,hidden_size), n_prompt=1

        return {'sentence_embedding':sent_embeds} # (bsz,me_num*n_prompt,hidden_size), n_prompt=1


    def _query_prompted_encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        # encode doc
        batch_size = input_ids.shape[0]
        
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len_query).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz,prefix_len+seq_len)

        position_ids = [0]+list(range(self.pre_seq_len_query+1,self.pre_seq_len_query+input_ids.shape[-1])) # [CLS] would be in position 0
        position_ids = torch.tensor(position_ids).unsqueeze(0).expand(batch_size,-1).to(input_ids.device) # (bsz,seq_len)
        
        sent_emb_list = []
        
        prefix_tokens = self.query_prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(input_ids.device) # (bsz,prefix_len)
        past_key_values = self.query_prefix_encoder(prefix_tokens) # (bsz,prefix_len,2*n_layers*hidden_size)
        past_key_values = past_key_values.view(batch_size,self.pre_seq_len_query,self.n_layer*2,self.n_head,self.head_size) # (bsz,prefix_len,2*n_layers,n_head,head_size)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2,0,3,1,4]).split(2,dim=0) # split_size_or_sections=2
        # (n_layers*2,bsz,n_heads,prefix_len,head_size)->(n_layers*(2,bsz,n_heads,prefix_len,head_size))

        outputs = bert_encoder(
            input_ids,
            attention_mask=attention_mask, # modified
            # token_type_ids=token_type_ids, # modified
            position_ids=position_ids, # modified
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds, # modified
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            past_key_values=past_key_values,
        ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
        # last_hidden_state is (bsz,prefix_len+seq_len,hidden_size)
        return get_representation_tensor(outputs) # (bsz,hidden_size), the first token is still [CLS]

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor # ?
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1


class MVRLearnerMixin(PreTrainedModel):
    def __init__(self, config, model_args, tokenizer, encoder, query_encoder=None, ):
        super().__init__(config)

        self.model_args = model_args
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.query_encoder = query_encoder

        self.pre_seq_len = model_args.pre_seq_len # if model_args.prefix_tokens is None else len(model_args.prefix_tokens)
        self.prefix_tokens = [torch.arange(self.pre_seq_len).long() for _ in range(model_args.n_prompt)] # not the same as model_args.prefix_tokens, these are new tokens not in the vocab
        self.prefix_encoder = nn.ModuleList([torch.nn.Embedding(self.pre_seq_len, config.hidden_size) for _ in range(model_args.n_prompt)]) # excludes token_type and position embeddings    
        self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else None
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else None

        if self.model_args.pooling=='scalar_mix':
            self.scalar_parameters = nn.Parameter(torch.zeros(self.pre_seq_len), requires_grad=True)

    @property
    def encoding_doc(self):
        return partial(self._prompted_encoding,bert_encoder=self.encoder)

    @property
    def encoding_query(self):
        if self.query_encoder is None:
            return self.encoder
        return self.query_encoder

    def _prompted_encoding(self, input_ids, attention_mask, bert_encoder, return_dict=True): # (bsz,seq_len)
        # encode doc
        batch_size = input_ids.shape[0]
        # cls_embed = bert_encoder.embeddings.word_embeddings(torch.LongTensor([[self.cls_token_id]]*batch_size).to(input_ids.device)) # (bsz,1,hidden_size)
        # sep_embed = bert_encoder.embeddings.word_embeddings(torch.LongTensor([[self.sep_token_id]]*batch_size).to(input_ids.device)) # (bsz,1,hidden_size)
        raw_embed = bert_encoder.embeddings.word_embeddings(input_ids[:,1:]) # (bsz,seq_len-1,hidden_size), remove [CLS]
                
        # token_type_ids = torch.cat(
        #     [torch.zeros(batch_size,self.pre_seq_len+2),torch.ones(batch_size,input_ids.shape[1]-1)],
        #     dim=1).long().to(input_ids.device)
        #     # (bsz,pre_seq_len+2+seq_len-1)
        
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len-1).to(input_ids.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) # (bsz,prefix_len+seq_len-1)

        
        prefix_tokens = self.prefix_tokens[0].unsqueeze(0).expand(batch_size, -1).to(input_ids.device) # (bsz,prefix_len)
        prompt_embed = self.prefix_encoder[0](prefix_tokens) # (bsz,prefix_len,hidden_size), excludes token_type and position enbeddings
        inputs_embeds = torch.cat([prompt_embed,raw_embed], dim=1) # (bsz,prefix_len+seq_len-1,hidden_size)

        position_ids = [0]*self.pre_seq_len+list(range(1,input_ids.shape[-1])) # (prefix_len+seq_len-1)
        position_ids = torch.tensor(position_ids).unsqueeze(0).expand(batch_size,-1).to(input_ids.device) # (bsz,prefix_len+seq_len-1)

        outputs = bert_encoder(
            # input_ids,
            attention_mask=attention_mask, # modified
            # token_type_ids=token_type_ids, # modified
            position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds, # modified
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            # past_key_values=past_key_values,
        ) # (last_hidden_state,pooler_output,hidden_states,attentions,cross_attentions,past_key_values)
        # last_hidden_state is (bsz,prefix_len+seq_len-1,hidden_size)

        sent_embeds = outputs["last_hidden_state"][:,:self.pre_seq_len,:].contiguous() # (bsz,prefix_len,hidden_size)

        if self.model_args.pooling=='mean':
            sent_embeds = torch.mean(sent_embeds,dim=1,keepdim=True) # (bsz,1,hidden_size)
        elif self.model_args.pooling=='scalar_mix':
            normed_weights = F.softmax(self.scalar_parameters,dim=0) # (num_layers)->(num_layers)
            sent_embeds = torch.matmul(normed_weights,sent_embeds).unsqueeze(1)
            # (num_layers)->(1,num_layers)->(1,1,num_layers)*(bsz,num_layers,hidden_size) = (bsz,hidden_size) -> (bsz,1,hidden_size)
        return {'sentence_embedding':sent_embeds} # (bsz,prefix_len,hidden_size) or (bsz,1,hidden_size) if mean & scalar mixing pooling

    @property
    def my_world_size(self):
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        return world_size

    def gather_tensor(self, target_tensor): # (bsz,hidden_size)
        if dist.is_initialized() and dist.get_world_size() > 1: # and self.training:
            target_tensor_list = [torch.zeros_like(target_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=target_tensor_list, tensor=target_tensor.contiguous())
            target_tensor_list[dist.get_rank()] = target_tensor #
            target_tensor_gathered = torch.cat(target_tensor_list, 0) # (bsz*num_ranks,hidden_size)
        else:
            target_tensor_gathered = target_tensor
        return target_tensor_gathered

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1
