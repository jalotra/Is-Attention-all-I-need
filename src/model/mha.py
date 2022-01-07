import torch
import numpy as np

from torch import nn
from typing import Optional

'''
See Section : 3.2.2
The paper says for (keys, values and queries) they linearly project 
the vectors d_model[keys], d_model[values] and d_model[queries] to make 
parallel h modules on which attention is run in parallel and final output is 
concatenation of these h resultant vectors. 
'''

# Linearly projects a tensor of model_dims to [heads, key_dims]
class Prepare(nn.Module):
    def __init__(self, 
        model_dims : int,
        number_heads : int, 
        key_dims : int,
        bias = True
    ):
        super().__init__()
        self.linear = nn.linear(model_dims, number_heads * key_dims, bias = bias)
        self.number_heads = number_heads
        self.key_dims = key_dims
    
    # Input : [seq_len, batch_size, model_dims]
    # Output : [seq_len, batch_size, number_heads, key_dims]
    def forward(self, x : torch.Tensor):
        remain_same = x.shape[:-1]
        x = self.linear_layer(x)
        x = x.view(*remain_same, self.number_heads, self.key_dims)
        
        return x

# Implements multi-head-attention
class MultiHeadAttenstion(nn.Module):
    def __init__(self, model_dims : int, number_heads : int, dropout_prob : float = 0.1, bias : bool = True):
        super().__init__()
        
        self.model_dims = model_dims
        self.number_heads = number_heads
        self.dropout_prob = dropout_prob
        self.bias = bias

        self.key_dims = self.model_dims // self.number_heads

        # Now get the (query, key and value) vectors
        self.key = Prepare(model_dims = model_dims, number_heads = number_heads, key_dims = self.key_dims, bias = bias)
        self.value = Prepare(model_dims = model_dims, number_heads = number_heads, key_dims = self.key_dims, bias = bias)
        self.query = Prepare(model_dims = model_dims, number_heads = number_heads, key_dims = self.key_dims, bias = bias)

        # Useful Layers 
        # Softmax accross 1 dimension 
        self.softmax = nn.Softmax(dim = 1)
        self.output = nn.Linear(model_dims, model_dims)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / np.sqrt(self.key_dims)
    
    # Calculates QK^t
    def get_scores(self, query : torch.Tensor, key = torch.Tensor):
        assert(len(query.shape) == 4)
        # After transpose : 
        # key : [seq_len, batch_size, key_dims, number_heads]
        key = torch.transpose(key, dim0= 2,dim1 = 3)
        return torch.matmul(query, key)
    
    # Masks the leftword information flow in decorder
    # Check 3.2.3[Point 3]
    def prepare_mask(self, mask : torch.Tensor):
        return mask.unsqueeze(-1)
    
    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, mask : Optional[torch.Tensor])

        seq_len, batch_size, _ = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        if mask is not None:
            raise NotImplementedError()
        
        scores = self.get_scores(query, key)
        scores *= self.scale

        if mask is not None:
            raise NotImplementedError()
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        # Multiply attn with V
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)