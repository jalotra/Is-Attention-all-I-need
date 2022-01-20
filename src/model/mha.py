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
        head_dims : int,
        bias = True
    ):
        super().__init__()
        self.linear = nn.linear(model_dims, number_heads * head_dims, bias = bias)
        self.number_heads = number_heads
        self.head_dims = head_dims

    # Input : [N, seq_len, model_dims]
    # Output : [N, seq_len, number_heads, heads_dim]
    def forward(self, x : torch.Tensor):
        remain_same = x.shape[:-1]
        x = self.linear_layer(x)
        x = x.view(*remain_same, self.number_heads, self.head_dims)
        return x

# Implements multi-head-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dims : int, number_heads : int, dropout_prob : float = 0.1, bias : bool = True):
        super().__init__()
        
        self.model_dims = model_dims
        self.number_heads = number_heads
        self.dropout_prob = dropout_prob
        self.bias = bias

        assert(self.model_dims % self.number_heads == 0)
        self.key_dims = self.model_dims // self.number_heads

        # Now get the (query, key and value) vectors
        self.key = Prepare(model_dims = model_dims, number_heads = number_heads, key_dims = self.key_dims, bias = self.bias)
        self.value = Prepare(model_dims = model_dims, number_heads = number_heads, key_dims = self.key_dims, bias = self.bias)
        self.query = Prepare(model_dims = model_dims, number_heads = number_heads, key_dims = self.key_dims, bias = self.bias)

        # Useful Layers 
        # Softmax accross 3 dimension 
        self.softmax = nn.Softmax(dim = 3)
        self.output = nn.Linear(model_dims, model_dims, bias = self.bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / np.sqrt(self.key_dims)
    
    # Calculates QK^t
    def get_scores(self, query : torch.Tensor, key = torch.Tensor):
        assert(len(query.shape) == 4)
        # key : [N, key_len, number_heads, heads_dim]
        # query : [N, query_len, number_heads, heads_dim]
        # Output : [N, heads, query_len, key_len]
        return torch.einsum("nqhd,nkhd->nhqk", [query, key])
    
    # Masks the leftword information flow in decorder
    # Check 3.2.3[Point 3]
    def prepare_mask(self, mask : torch.Tensor):
        return mask.unsqueeze(-1)
    
    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, mask : Optional[torch.Tensor])
        N, query_len, _ = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query, key)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float(-1e+20)) 
        
        scores *= self.scale
        attn = self.softmax(scores)
        attn = self.dropout(attn)


        # Multiply attn with V
        # attn shape = [N, heads, query_len, key_len]
        # V shape : [N, value_len, heads, head_dims]
        # Output shape : [N, query_len, heads, head_dims]
        x = torch.einsum("nhql,nlhd->nlhd", [attn, value])
        self.attn = attn.detach()
        x = x.reshape(N, query_len, -1)

        return self.output(x)
