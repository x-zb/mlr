from typing import List, Union, Callable, Any
from contextlib import nullcontext
# from itertools import repeat
from collections import UserDict
import logging

import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast

from proj_dense.grad_cache.context_managers import RandContext

import numpy as np

logger = logging.getLogger(__name__)


class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
            self,
            model: nn.Module,
            chunk_size: int,
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        self.model = model
        self.chunk_size = chunk_size


    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), []) # [k1t,k2t,...knt]

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def model_call(
            self,
            **model_inputs:dict,
        ):
        # with autocast() if accelerator.mixed_precision=='fp16' else nullcontext():
        return self.model(training_mode="retrieval_finetune",**model_inputs)
        # emb_doc,emb_query (bsz*num_doc,1,hidden_size),(bsz,hidden_size)
        

    
    def forward_no_grad(
            self,
            input_ids, attention_mask,
            input_ids_query, attention_mask_query,
    ) -> [Tensor, List[RandContext]]:
        """
        The first forward pass without gradient computation.
        [(chunk_size,num_doc,doc_len),(chunk_size,num_doc,doc_len),...]
        [(chunk_size,query_len),(chunk_size,query_len),...]
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = [] # [RandomContext for chunk i]_for_doc_input
        # rnd_states_q = [] # [RandomContext for chunk i]_for_query_input
        model_reps = [] # [repr tensor for chunk i]=[(chunk_size,...)]
        model_reps_q = []
        query_class = []

        with torch.no_grad():
            for x,y,xq,yq in zip(input_ids,attention_mask,input_ids_query,attention_mask_query):
                rnd_states.append(RandContext(*self.get_input_tensors((x,y,xq,yq))))
                # rnd_states_q.append(RandContext(*self.get_input_tensors((xq,yq))))
                
                outputs = self.model_call(input_ids=x, attention_mask=y,input_ids_query=xq, attention_mask_query=yq) 
                if len(outputs)==2:
                    z,zq = outputs
                else:
                    assert len(outputs)==3
                    z,zq, reg_target = outputs
                    query_class.append(reg_target)
                # (chunk_size*num_doc,1,hidden_size),(chunk_size,hidden_size),(chunk_size) or (chunk_size*num_doc,doc_len)
                model_reps.append(z)
                model_reps_q.append(zq)

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0) # (bsz*num_doc,1,hidden_size)
        model_reps_q = torch.cat(model_reps_q, dim=0) # (bsz,hidden_size)
        query_class = torch.cat(query_class, dim=0) if len(query_class)>0 else None # (bsz) or (bsz*num_doc,doc_len)
        return model_reps,model_reps_q,query_class,rnd_states #,rnd_states_q


    def build_cache(self, emb_doc, emb_query, query_class, accelerator) -> [List[Tensor], dict]:
        """
        Compute the gradient cache
        emb_doc: (bsz*num_doc,1,hidden_size)
        emb_query: (bsz,hidden_size)
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        reps = [emb_doc.detach().requires_grad_(), emb_query.detach().requires_grad_()] 
        # [(bsz*num_doc,1,hidden_size), (bsz,hidden_size)]
        
        # with autocast() if accelerator.mixed_precision=='fp16' else nullcontext():
        if query_class is None:
            dict_for_meta = self.model.module.compute_loss(*reps)
        else:
            dict_for_meta = self.model.module.compute_loss(*reps,query_class)
        loss = dict_for_meta["loss"]

        # if self.fp16:
        #     self.scaler.scale(loss).backward()
        # else:
        #     loss.backward()
        accelerator.backward(loss) # incorrect: accelerator will accumulate gradients for reps from other GPUs
        # note that in line 126 the reps have been detached from the computational graph,
        # therefore this loss.backward() will not compute gradient for the model parameters,
        # but only for the reps. 
        # r.detach() is critical.

        cache = [r.grad for r in reps] # [(bsz*num_doc,1,hidden_size), (bsz,hidden_size)]

        dict_for_meta["loss"] = loss.detach()
        return cache, dict_for_meta # [(bsz*num_doc,1,hidden_size), (bsz,hidden_size)], dict

    def forward_backward(
            self,
            input_ids,attention_mask,input_ids_query,attention_mask_query,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            accelerator,
            no_sync_except_last: bool = False
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :input_ids,attention_mask,input_ids_query,attention_mask_query: List of input chunks
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last: # only after the last chunk do we need to syncronize gradients
            sync_contexts = [self.model.no_sync for _ in range(len(input_ids) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(input_ids))]

        for x,y,xq,yq, state, (gradient,gradient_q), sync_context in zip(
            input_ids,attention_mask,input_ids_query,attention_mask_query,
            random_states,cached_gradients,sync_contexts):
            with sync_context():
                with state:
                    outputs = self.model_call(input_ids=x, attention_mask=y,
                        input_ids_query=xq, attention_mask_query=yq)
                z,zq = outputs[0],outputs[1]
                # (chunk_size*num_doc,1,hidden_size),(chunk_size,hidden_size)
                surrogate = torch.dot(z.flatten(), gradient.flatten())+torch.dot(zq.flatten(), gradient_q.flatten()) # (1) reps*grad
                accelerator.backward(surrogate)
                # surrogate_q = torch.dot(zq.flatten(), gradient_q.flatten()) # (1) reps*grad
                # accelerator.backward(surrogate_q)

    def cache_step(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
            input_ids_query=None, attention_mask_query=None, token_type_ids_query=None, position_ids_query=None,
            accelerator=None, no_sync_except_last=False,
            **kwargs,
    ):
        # (bsz,num_doc,doc_len), (bsz,query_len)

        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), [self.model])), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'

        num_docs = input_ids.size(1)

        input_ids = input_ids.split(self.chunk_size, dim=0) # (bsz,num_doc,doc_len)->[(chunk_size,num_doc,doc_len),(chunk_size,num_doc,doc_len),...]
        attention_mask = attention_mask.split(self.chunk_size, dim=0)
        input_ids_query = input_ids_query.split(self.chunk_size, dim=0) # (bsz,query_len)->[(chunk_size,query_len),(chunk_size,query_len),...]
        attention_mask_query = attention_mask_query.split(self.chunk_size, dim=0)
        
        model_reps,model_reps_q,query_class,rnd_states = self.forward_no_grad(
            input_ids,attention_mask,input_ids_query,attention_mask_query) 
            # (bsz*num_doc,1,hidden_size),(bsz,hidden_size),(bsz),[RandomContext for chunk i]
        
        cache, dict_for_meta = self.build_cache(model_reps,model_reps_q,query_class,accelerator) 
        # [(bsz*num_doc,1,hidden_size), (bsz,hidden_size)]_grad, dict

        cache = [cache[0].split(self.chunk_size*num_docs),cache[1].split(self.chunk_size)]
        # [[(chunk_size*num_doc,1,hidden_size),... grad_for_doc_encoder], 
        #  [(chunk_size,hidden_size), ... grad_for_query_encoder]]
        cache = [(c,c_q) for c,c_q in zip(cache[0],cache[1])]

        self.forward_backward(input_ids,attention_mask,input_ids_query,attention_mask_query,
            cache,rnd_states,accelerator,no_sync_except_last=no_sync_except_last)

        return dict_for_meta # results for one thread