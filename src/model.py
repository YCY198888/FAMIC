"""
FAMIC Model Definition

This module contains the FAMIC model architecture implemented in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Optional, Dict, Any


# ============================================================
# Tied Linear Layer for logistic regression like behavior
# ============================================================

class tiedLinear(nn.Module):
    """
    Placeholder. Replace with your implementation.
    If your original tiedLinear had different behavior, use that.
    """
    def __init__(self, in_features, out_features=None):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.lin(x)


# ============================================================
# Plain self-attention block for position-invariant components of FAMIC (sentiment wordmask and global shifter)
# ============================================================

class SelfAttention(nn.Module):
    """
    Minimal Multi-Head Self-Attention (batch_first).
    Replace with your own SelfAttention if you have a custom one.
    """
    def __init__(self, emb: int, heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert emb % heads == 0
        self.emb = emb
        self.heads = heads
        self.head_dim = emb // heads

        self.qkv = nn.Linear(emb, 3 * emb, bias=False)
        self.proj = nn.Linear(emb, emb, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, heads, L, head_dim)
        q = q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, L, L)

        if key_padding_mask is not None:
            # key_padding_mask: (B, L) with True for real tokens (or False for pad)
            # We want to mask pad positions in keys. Convert to (B, 1, 1, L) where pad -> -inf
            # Expect key_padding_mask True=real, False=pad
            mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # pad=True at pad positions
            attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


# ============================================================
# Relative Positional Embedding (because it has to be learnt)
# words on individual head of self-attention blocks
# ============================================================

class RelativePosition(nn.Module):
    def __init__(self, num_units: int, max_relative_position: int):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.empty(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q: int, length_k: int):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)

        # distance_mat[i, j] = j - i
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = (distance_mat_clipped + self.max_relative_position).long()  # (Lq, Lk)

        embeddings = self.embeddings_table[final_mat]  # (Lq, Lk, head_dim)
        return embeddings


# ============================================================
# Cell 5 — MultiHeadAttentionLayer (cleaned: remove duplicated projections, restore softmax)
# Here we modified the original implementation of relative self-attention 
# 
# ============================================================

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, max_relative_position: int = 2, dropout: float = 0.1):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_relative_position

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        # One projection for q, k, v (fused)
        self.qkv = nn.Linear(hid_dim, 3 * hid_dim, bias=False)
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, L, D)
        key_padding_mask: (B, L) bool, True for real tokens, False for pad
        """
        B, L, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, heads, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        #  k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5

        # Content-based attention term (optional; can be disabled to mimic your original)
        #   attn1 = torch.matmul(q, k.transpose(-2, -1))  # (B, heads, L, L)

        # Relative position term (your original main term)
        # Build relative embeddings: (L, L, head_dim)
        r_k = self.relative_position_k(L, L)  # (L, L, head_dim)

        # q: (B, heads, L, head_dim)
        # want: (B, heads, L, L) via einsum over head_dim with r_k indexed by (i,j)
        attn2 = torch.einsum("bhid,ijd->bhij", q, r_k)

        attn = ( attn2 )/ scale

        if key_padding_mask is not None:
            # mask keys (last dim): pad positions -> -inf
            pad_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Relative value term
        r_v = self.relative_position_v(L, L)  # (L, L, head_dim)

        # weight1 = attn @ v
        weight1 = torch.matmul(attn, v)  # (B, heads, L, head_dim)

        # weight2 using relative values: einsum attn(b,h,i,j)*r_v(i,j,d)->(b,h,i,d)
        weight2 = torch.einsum("bhij,ijd->bhid", attn, r_v)

        out = weight1 + weight2  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        return self.fc_o(out)


class MultiHeadAttentionLayer_2(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, max_relative_position: int = 2, dropout: float = 0.1):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_relative_position

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        # One projection for q, k, v (fused)
        self.qkv = nn.Linear(hid_dim, 3 * hid_dim, bias=False)
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, L, D)
        key_padding_mask: (B, L) bool, True for real tokens, False for pad
        """
        B, L, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, heads, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        #k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5

        # Content-based attention term (optional; can be disabled to mimic your original)
        # attn1 = torch.matmul(q, k.transpose(-2, -1))  # (B, heads, L, L)

        # Relative position term (your original main term)
        # Build relative embeddings: (L, L, head_dim)
        r_k = self.relative_position_k(L, L)  # (L, L, head_dim)

        # q: (B, heads, L, head_dim)
        # want: (B, heads, L, L) via einsum over head_dim with r_k indexed by (i,j)
        attn2 = torch.einsum("bhid,ijd->bhij", q, r_k)

        attn = attn2 / scale

        if key_padding_mask is not None:
            # mask keys (last dim): pad positions -> -inf
            pad_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Relative value term
        r_v = self.relative_position_v(L, L)  # (L, L, head_dim)

        # weight1 = attn @ v
        weight1 = torch.matmul(attn, v)  # (B, heads, L, head_dim)

        # weight2 using relative values: einsum attn(b,h,i,j)*r_v(i,j,d)->(b,h,i,d)
        weight2 = torch.einsum("bhij,ijd->bhid", attn, r_v)

        out = weight1 + weight2  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        return self.fc_o(out)


class Mask_block(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        n_layers,
        max_relative_position=2,
        pivot=0.5,
        drop_prob=0.5,
        num_heads=1
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        embedding_dim = hidden_dim

        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = tiedLinear(embedding_dim, 1)
        self.fc12 = nn.Linear(embedding_dim, 1)
        self.attention1 = SelfAttention(emb=embedding_dim, heads=self.num_heads)

        self.sig = nn.Sigmoid()

        # Make pivot move with device; not trainable unless you want it to be
        self.register_buffer("pivot", torch.tensor(float(pivot)))

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout10 = nn.Dropout(0.1)

    def forward(self, embeds, mask_1, digits=None):
        """
        embeds: (B, L, D)
        mask_1: (B, L)  (bool or 0/1 float) padding/valid-token mask
        digits: unused here, kept for API compatibility

        Returns:
          mask_out: (B, L)
          p1: (B,)
          p3: (B,)
          mean_mask_out: python float
        """
        B = embeds.size(0)  # ✅ dynamic batch size (no drop_last dependency)

        # Ensure mask_1 is float for multiplication
        if mask_1.dtype != embeds.dtype:
            mask_1f = mask_1.to(dtype=embeds.dtype)
        else:
            mask_1f = mask_1

        # (Optional) enforce attention only on valid tokens if your SelfAttention supports it.
        # If your SelfAttention signature is SelfAttention(x) only, keep as-is.
        embeds = self.attention1(embeds)

        # token-wise mask logits -> (B, L, 1) then squeeze -> (B, L)
        #mask = self.fc1(embeds).squeeze(-1)
        mask = self.fc12(embeds).squeeze(-1)
        # sigmoid to (0,1)
        mask = self.sig(mask)

        # apply padding mask
        mask_out = mask * mask_1f

        # regularizers, per-sample
        p3 = torch.norm(mask * (1.0 - mask), p=1, dim=1)  # (B,)
        p1 = torch.norm(mask, p=1, dim=1)                 # (B,)

        mean_mask_out = mask_out.mean().detach().item()

        return mask_out, p1, p3, mean_mask_out


class Sentiment_block(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        n_layers,
        max_relative_position=2,
        pivot=0.5,
        drop_prob=0.5,
        num_heads=1
    ):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        embedding_dim = hidden_dim

        # dropout
        self.dropout = nn.Dropout(drop_prob)
        self.dropout10 = nn.Dropout(0.1)

        # NOTE: your original code overwrote self.fc1 twice.
        # Keeping only the actually-used layers to avoid confusion.
        self.fc12 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2  = tiedLinear(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        self.add_p12 = AddBatchScalar()
        self.norm = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.relu = nn.ReLU()

        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, embeds, mask_1, mask=None, digits=None):
        """
        embeds: (B, L, D)
        mask_1: (B, L) padding/valid-token mask (bool or 0/1 float)
        mask:   (B, L) learned mask (optional)
        digits: unused here, kept for API compatibility

        Returns:
          out: (B, L) sentiment score per token position (after reducing embedding dim)
        """
        # ✅ no global batch_size dependence; everything is shape-driven

        # Cast masks to the same dtype for safe multiplication
        if mask_1 is not None:
            mask_1f = mask_1.to(dtype=embeds.dtype) if mask_1.dtype != embeds.dtype else mask_1
        else:
            mask_1f = None

       
        if mask is not None:
            maskf = mask.to(dtype=embeds.dtype) if mask.dtype != embeds.dtype else mask
            embeds = embeds * maskf.unsqueeze(-1)  # ✅ actually apply learned mask (fix your embdes-unused bug)

        # Optional: also zero-out padding tokens (recommended)
        if mask_1f is not None:
            embeds = embeds * mask_1f.unsqueeze(-1)

        # p12 = self.norm(embeds.mean(dim=1)).squeeze(-1)   # (B, 1) regularizer on mean embedding magnitude
        p12 = self.norm(embeds.mean(dim=1)).squeeze(-1)
        # embeds = self.relu(self.fc12(embeds)) 
        # embeds = self.fc12(embeds)
        # embeds = self.dropout10(embeds)
        # t = self.relu(self.fc12(embeds)) 
        # t = self.fc12(t)
        # t = t + p12[:, None, None] 
        # embeds = t
        embeds = self.ff(embeds)
        embeds = self.add_p12(embeds, p12)
        embeds = self.dropout10(embeds)

        out = self.fc2(embeds)        # (B, L, D) if tiedLinear is square
        out = out.mean(dim=2)         # (B, L) reduce embedding dimension
        
        # Optional: keep padding positions as 0
        if mask_1f is not None:
            out = out * mask_1f
        
        return out, p12


class AddBatchScalar(nn.Module):
    def forward(self, x, s):  # x: (B,L,D), s: (B,) or (B,1)
        return x + s.view(-1, 1, 1)


class Shifter_block1(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        n_layers,
        max_relative_position=2,
        pivot=0.5,
        drop_prob=0.5,
        num_heads=1
    ):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        embedding_dim = hidden_dim

        self.dropout = nn.Dropout(drop_prob)
        self.dropout10 = nn.Dropout(0.1)

        # You only use attention1 + fc2 in forward; keep the rest minimal/clean.
        self.attention1 = SelfAttention(emb=embedding_dim, heads=self.num_heads)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.fc2 = nn.Linear(embedding_dim, 1)
        self.sig = nn.Sigmoid()

        # Device-safe constant
        self.register_buffer("pivot", torch.tensor(float(pivot)))

    def forward(self, embeds, mask_1, length=None, pos=None):
        """
        embeds: (B, L, D)
        mask_1: (B, L) padding/valid-token mask (bool or 0/1 float)
        length/pos: unused here, kept for API compatibility

        Returns:
          shifter_out: (B, L) in [0, 10] on valid tokens, 0 on padding
        """
        # ✅ no global batch_size usage; no reshapes depending on fixed batch size

        # Ensure mask is float for multiplication
        mask_1f = mask_1.to(dtype=embeds.dtype) if mask_1.dtype != embeds.dtype else mask_1

        embeds = embeds * mask_1f.unsqueeze(-1)
        embeds = self.attention1(embeds)
        embeds = self.layer_norm(embeds)
        embeds = self.dropout10(embeds)

        # (B, L, 1) -> (B, L) (IMPORTANT: use squeeze(-1), not squeeze())
        shifter = self.fc2(embeds).squeeze(-1)

        shifter = self.sig(shifter) * 10.0
        shifter_out = shifter * mask_1f  # zero-out padding

        return shifter_out


class Shifter_block2(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        n_layers,
        max_relative_position=2,
        pivot=0.5,
        drop_prob=0.5,
        num_heads=1
    ):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        embedding_dim = hidden_dim

        # This is the only attention actually used in your forward
        self.re_pos_attention1 = MultiHeadAttentionLayer_2(
            hid_dim=embedding_dim,
            n_heads=self.num_heads,
            max_relative_position=max_relative_position
        )

        self.dropout = nn.Dropout(drop_prob)
        self.dropout10 = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.fc2 = nn.Linear(embedding_dim, 1)
        self.tanh = nn.Tanh()

        # Device-safe constant
        self.register_buffer("pivot", torch.tensor(float(pivot)))

    def forward(self, embeds, mask_1, length=None, pos=None):
        """
        embeds: (B, L, D)
        mask_1: (B, L) padding/valid-token mask (bool or 0/1 float)
        length/pos: unused here, kept for API compatibility

        Returns:
          shifter_out: (B, L) in [-1, 1] on valid tokens, 0 on padding
          ps32: (B,)   regularizer term per sample
        """
        # ✅ no fixed batch_size; safe for last batch and B=1

        mask_1f = mask_1.to(dtype=embeds.dtype) if mask_1.dtype != embeds.dtype else mask_1

        embeds = embeds * mask_1f.unsqueeze(-1)

        embeds = self.re_pos_attention1(embeds)   # (B, L, D)
        embeds = self.layer_norm(embeds)
        embeds = self.dropout10(embeds)

        # (B, L, 1) -> (B, L) (IMPORTANT: squeeze only last dim)
        shifter = self.fc2(embeds).squeeze(-1)

        shifter_out = self.tanh(shifter) * mask_1f

        # regularizer per sample
        ps32 = torch.norm((1.0 - shifter_out**2), p=1, dim=1)  # (B,)

        return shifter_out, ps32


class Synthesizer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        n_layers,
        max_relative_position=2,
        pivot=0.5,
        drop_prob=0.5,
        num_heads=1,
        digits_dim=1
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(drop_prob)

        # Keep your components (even if not used everywhere yet)
        self.fc1 = tiedLinear(hidden_dim, 1)
        self.fc2 = tiedLinear(hidden_dim)
        self.fc3 = nn.Linear(1 + digits_dim, 1)

        self.attention = SelfAttention(emb=hidden_dim, heads=self.num_heads)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.register_buffer("pivot", torch.tensor(float(pivot)))

    def forward(
        self,
        sent,
        digits,
        mask,
        shift1,
        shift2,
        norm,
        use_mask=False,
        use_shift1=False,
        use_shift2=False
    ):
        """
        sent:   (B, L) or (B, L, *) depending on your pipeline; expected (B, L) here
        mask:   (B, L)
        shift1: (B, L)
        shift2: (B, L)
        digits: whatever you pass through (unused here; kept for compatibility)

        Returns:
          sig_out: (B,) probabilities
          p2:      same as input sent (for your p2 regularizer)
          save_out:(B, L) token-level values before pooling
        """
        # ✅ dynamic batch size; safe for last batch and B=1
        B = sent.size(0)

        p2 = sent
        # Apply optional modifiers (same semantics)
        if use_shift1 and shift1 is not None:
            sent = sent * shift1
        if use_shift2 and shift2 is not None:
            sent = sent * shift2
        if use_mask and mask is not None:
            sent = sent * mask

        save_out = sent  # keep token-level values for debugging/analysis

        # Pool to document-level
        out = self.dropout(sent)
        out = out.mean(dim=1)         # (B,) + stablizaer

        # # Sigmoid probability
        # sig_out = self.sig(out)        # (B,)

        # ✅ Do NOT reshape using global batch_size, and do NOT slice [:, -1]
        # sig_out is already (B,) which works with BCELoss vs labels (B,)

        return out, p2, save_out


class Embeds(nn.Module):

    def __init__(self, vocab_size, hidden_dim, weight_matrix):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        # embedding False = trainable
        # self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)

        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weight_matrix, False)

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        #  mask = torch.transpose(mask,0,1)
        mask_1 = x.ge(0.1)
        #beta
        # x = (x * mask).int()
        embeds = self.embedding(x)
        #    embeds = self.pos_enc(embeds)
        # return last sigmoid output and hidden state
        return embeds, mask_1


def create_emb_layer(weights_matrix: torch.Tensor, freeze: bool = True, padding_idx: int = 0):
    """
    weights_matrix: torch.FloatTensor of shape [V, D]
    """
    assert weights_matrix.dim() == 2, "weights_matrix must be [V, D]"
    emb = nn.Embedding.from_pretrained(weights_matrix, freeze=freeze, padding_idx=padding_idx)
    V, D = weights_matrix.size()
    return emb, V, D


# Default model hyperparameters
MAX_LEN = 100  # word-level tokenization requires smaller token limit
SEED = 2025
EMBEDDING_DIMENSIONS = 300
VOCAB_LENGTH = 250000  # nice round number that is larger than the Vocab size
INPUT_LENGTH = MAX_LEN
DIGITS_DIM = 1  # 9-1 reviewers dummies, 1 lengths
PIVOT = 0.5
N_LAYERS = 2  # no use
NUM_HEADS = 10


def create_embedding_matrix(vocab_length: int = VOCAB_LENGTH, embedding_dim: int = EMBEDDING_DIMENSIONS) -> torch.Tensor:
    """
    Create an empty embedding matrix.
    
    Args:
        vocab_length: Size of vocabulary
        embedding_dim: Embedding dimension
        
    Returns:
        torch.Tensor of shape (vocab_length + 1, embedding_dim)
    """
    embedding_matrix = np.zeros((vocab_length + 1, embedding_dim), dtype=np.float32)
    pad_embedding_matrix = torch.from_numpy(np.asarray(embedding_matrix, dtype=np.float32))
    return pad_embedding_matrix


def initialize_model_blocks(
    embedding_matrix: torch.Tensor,
    hidden_dim: int = EMBEDDING_DIMENSIONS,
    n_layers: int = N_LAYERS,
    max_relative_position_mask: int = 2,
    max_relative_position_shift: int = 5,
    pivot: float = PIVOT,
    num_heads: int = NUM_HEADS,
    drop_prob: float = 0.5,
    digits_dim: int = DIGITS_DIM
):
    """
    Initialize all FAMIC model blocks.
    
    Args:
        embedding_matrix: Pre-trained embedding matrix
        hidden_dim: Hidden dimension (default: 300)
        n_layers: Number of layers (not used, kept for compatibility)
        max_relative_position_mask: Max relative position for mask blocks
        max_relative_position_shift: Max relative position for shifter blocks
        pivot: Pivot value
        num_heads: Number of attention heads
        drop_prob: Dropout probability
        digits_dim: Dimension for digits/auxiliary features
        
    Returns:
        Dictionary containing all model blocks
    """
    vocab_size = embedding_matrix.shape[0]
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    mb0 = Embeds(vocab_size, hidden_dim, embedding_matrix)
    mb1 = Sentiment_block(vocab_size, hidden_dim, n_layers, max_relative_position=max_relative_position_mask, 
                          pivot=pivot, num_heads=num_heads, drop_prob=drop_prob)
    mb2 = Mask_block(vocab_size, hidden_dim, n_layers, max_relative_position=max_relative_position_mask, 
                     pivot=pivot, num_heads=num_heads, drop_prob=drop_prob)
    mb31 = Shifter_block1(vocab_size, hidden_dim, n_layers, max_relative_position=max_relative_position_shift, 
                          pivot=pivot, num_heads=num_heads, drop_prob=drop_prob)
    mb32 = Shifter_block2(vocab_size, hidden_dim, n_layers, max_relative_position=max_relative_position_shift, 
                          pivot=pivot, num_heads=num_heads, drop_prob=drop_prob)
    mb4 = Synthesizer(vocab_size, hidden_dim, n_layers, max_relative_position=max_relative_position_mask, 
                      pivot=pivot, num_heads=num_heads, drop_prob=drop_prob, digits_dim=digits_dim)
    
    return {
        'embeds': mb0,
        'sentiment': mb1,
        'mask': mb2,
        'shifter1': mb31,
        'shifter2': mb32,
        'synthesizer': mb4
    }


class FAMIC(nn.Module):
    """
    FAMIC Model Architecture
    
    Complete FAMIC model combining all blocks.
    """
    
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = EMBEDDING_DIMENSIONS,
        n_layers: int = N_LAYERS,
        max_relative_position_mask: int = 2,
        max_relative_position_shift: int = 5,
        pivot: float = PIVOT,
        num_heads: int = NUM_HEADS,
        drop_prob: float = 0.5,
        digits_dim: int = DIGITS_DIM,
        **kwargs
    ):
        """
        Initialize FAMIC model.
        
        Args:
            embedding_matrix: Pre-trained embedding matrix
            hidden_dim: Hidden dimension (default: 300)
            n_layers: Number of layers (not used, kept for compatibility)
            max_relative_position_mask: Max relative position for mask blocks
            max_relative_position_shift: Max relative position for shifter blocks
            pivot: Pivot value
            num_heads: Number of attention heads
            drop_prob: Dropout probability
            digits_dim: Dimension for digits/auxiliary features
            **kwargs: Additional model-specific parameters
        """
        super(FAMIC, self).__init__()
        
        # Initialize all model blocks
        blocks = initialize_model_blocks(
            embedding_matrix=embedding_matrix,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            max_relative_position_mask=max_relative_position_mask,
            max_relative_position_shift=max_relative_position_shift,
            pivot=pivot,
            num_heads=num_heads,
            drop_prob=drop_prob,
            digits_dim=digits_dim
        )
        
        self.embeds = blocks['embeds']
        self.sentiment = blocks['sentiment']
        self.mask = blocks['mask']
        self.shifter1 = blocks['shifter1']
        self.shifter2 = blocks['shifter2']
        self.synthesizer = blocks['synthesizer']
        
    def forward(self, x: torch.Tensor, use_mask: bool = False, use_shift1: bool = False, use_shift2: bool = False):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length) containing token indices
            use_mask: Whether to use learned mask
            use_shift1: Whether to use first shifter
            use_shift2: Whether to use second shifter
            
        Returns:
            Dictionary containing model outputs
        """
        # Get embeddings and padding mask
        embeds, mask_1 = self.embeds(x)
        
        # Mask block
        mask_out, p1, p3, mean_mask_out = self.mask(embeds, mask_1)
        
        # Sentiment block
        sent, p12 = self.sentiment(embeds, mask_1, mask=mask_out if use_mask else None)
        
        # Shifter blocks
        shift1 = self.shifter1(embeds, mask_1)
        shift2, ps32 = self.shifter2(embeds, mask_1)
        
        # Synthesizer
        out, p2, save_out = self.synthesizer(
            sent, 
            digits=None,  # TODO: Add digits if available
            mask=mask_out,
            shift1=shift1,
            shift2=shift2,
            norm=None,
            use_mask=use_mask,
            use_shift1=use_shift1,
            use_shift2=use_shift2
        )
        
        return {
            'output': out,
            'mask': mask_out,
            'sentiment': sent,
            'shift1': shift1,
            'shift2': shift2,
            'regularizers': {
                'p1': p1,
                'p2': p2,
                'p3': p3,
                'p12': p12,
                'ps32': ps32
            }
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], embedding_matrix: torch.Tensor) -> 'FAMIC':
        """
        Create model instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing model parameters
            embedding_matrix: Pre-trained embedding matrix
            
        Returns:
            Initialized FAMIC model
        """
        model_params = config.get('model', {})
        return cls(embedding_matrix=embedding_matrix, **model_params)
    
    @classmethod
    def from_pretrained(
        cls,
        weights_path: str,
        embedding_matrix: torch.Tensor,
        device: Optional[torch.device] = None,
        **model_kwargs
    ) -> 'FAMIC':
        """
        Load model from pretrained weights (legacy method for single weight file).
        
        Args:
            weights_path: Path to the model weights file (.pth, .pt, or .ckpt)
            embedding_matrix: Pre-trained embedding matrix
            device: Device to load model on (default: cuda if available, else cpu)
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Model loaded with pretrained weights
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = cls(embedding_matrix=embedding_matrix, **model_kwargs)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model
    
    @classmethod
    def from_pretrained_huggingface(
        cls,
        dataset_name: str,
        embedding_matrix: torch.Tensor,
        cache_dir: Optional[str] = None,
        version: Optional[str] = None,
        device: Optional[torch.device] = None,
        **model_kwargs
    ) -> 'FAMIC':
        """
        Load model from pretrained weights downloaded from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset ("twitter" or "wine")
            embedding_matrix: Pre-trained embedding matrix
            cache_dir: Directory to cache downloaded weights (default: "models")
            version: Model version (defaults to "v2" from registry)
            device: Device to load model on (default: cuda if available, else cpu)
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Model loaded with pretrained weights from HuggingFace
        """
        from src.download_weights import load_pretrained_weights
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = cls(embedding_matrix=embedding_matrix, **model_kwargs)
        
        # Load pretrained weights into model blocks
        model_blocks = {
            'embeds': model.embeds,
            'sentiment': model.sentiment,
            'mask': model.mask,
            'shifter1': model.shifter1,
            'shifter2': model.shifter2,
            'synthesizer': model.synthesizer
        }
        
        # Load weights
        load_pretrained_weights(
            model_blocks=model_blocks,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            version=version,
            device=device
        )
        
        model.to(device)
        model.eval()
        
        return model
