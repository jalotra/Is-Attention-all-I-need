import torch
from torch import nn
from .mha import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(
        self, 
        embed_size : int,
        heads : int, 
        dropout_prob : float,
        forward_expansion : bool
        ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.attention_layer = MultiHeadAttention(embed_size=embed_size, 
                                                number_heads=heads,
                                                 dropout_prob=dropout_prob)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.Relu(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, value : torch.Tensor, key : torch.Tensor, query : torch.Tensor, mask : torch.Tensor):
        attention = self.attention(query, key, value, mask)
        # Now a dropout 
        x = self.dropout(self.norm1(attention + query))
        save = x
        # Now a Fully Connected LayerNorm
        x = self.feed_forward(x)
        # Now a add + norm + droput
        x = self.dropout(self.norm2(x + save))
        return x
