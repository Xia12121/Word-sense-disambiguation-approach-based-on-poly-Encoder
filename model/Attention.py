import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value):
        B, Lq, _ = query.size()
        _, Lk, _ = key.size()
        _, Lv, _ = value.size()
        
        # Linear projections
        Q = self.q_proj(query)  # (B, Lq, embed_dim)
        K = self.k_proj(key)    # (B, Lk, embed_dim)
        V = self.v_proj(value)  # (B, Lv, embed_dim)

        # Split into multiple heads
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lq, head_dim)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lk, head_dim)
        V = V.view(B, Lv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, Lv, head_dim)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, Lq, Lk)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, num_heads, Lq, Lk)
        
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, Lq, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)  # (B, Lq, embed_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)  # (B, Lq, embed_dim)
        
        return output