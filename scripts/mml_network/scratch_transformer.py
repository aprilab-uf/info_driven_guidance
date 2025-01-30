import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """NOT USED currently"""

    def __init__(self, d_model, dropout=0.1, max_len=10):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


class Head(nn.Module):
    """one head of self-attention
    This module computes the self-attention mechanism on a single head.
    The self-attention mechanism is a weighted sum of the values, where the
    weights are computed by a compatibility function of the keys and queries.
    The compatibility function is the dot product of the keys and queries.
    The weights are then normalized by a softmax function.

    We apply a trilled mask to the attention scores to prevent the model from
    attending to the future tokens.
    """

    def __init__(self, head_size, input_dim=2, block_size=10, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(input_dim, head_size, bias=False)
        self.query = nn.Linear(input_dim, head_size, bias=False)
        self.value = nn.Linear(input_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, input_dim=2, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size=head_size, input_dim=input_dim) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a ReLU non-linearity"""

    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10 * input_dim),
            nn.ReLU(),
            nn.Linear(10 * input_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation

    The block applies a multi-head self-attention mechanism to its input,
    followed by a feed-forward neural network. Both the attention and the
    feed-forward network have residual connections around them.
    """

    def __init__(self, input_dim, n_head):
        # input_dim: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = input_dim // n_head
        self.sa = MultiHeadAttention(n_head, head_size, input_dim=input_dim)
        self.ffwd = FeedFoward(input_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ScratchTransformer(nn.Module):
    def __init__(self, input_dim=2, block_size=10, n_embed=2, n_head=1, n_layer=1):
        """A simple transformer for time series forecasting.
        The transformer consists of a stack of blocks, each containing a multi-head self-attention
        mechanism and a simple feed-forward neural network at the end.


        Args:
            input_dim: number of features in the input
            block_size: number of time steps in the input
            n_embed: embedding dimension
            n_head: number of heads in the multi-head attention
            n_layer: number of transformer blocks

        """
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embed = nn.Linear(input_dim, n_embed).to(device)
        self.position_embedding_table = nn.Embedding(block_size, n_embed).to(device)
        # self.position_embedding_table  = PositionalEncoding(n_embed, 0.2, max_len=10)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head).to(device) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed).to(device)  # final layer norm
        self.lm_head = nn.Linear(n_embed, 2).to(
            device
        )  # output dim is same of input dim

        # better init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, targets=None):
        src = src.view(-1, 10, 2)

        # src and targets are both (B,T,C) tensors
        src = self.embed(src)
        B, T, C = src.shape  # B: batch size, T: time steps, C: embed dimension
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = src + pos_emb  # (B,T,C)
        # x = self.position_embedding_table(src)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        output = self.lm_head(x)  # (B,T,C)

        output = output[:, -1, :]  # only return the last time step
        return output

    def predict(self, x):
        with torch.no_grad():
            return (
                self.forward(torch.from_numpy(x.astype(np.float32)).to(device))
                .cpu()
                .detach()
                .numpy()
            )
