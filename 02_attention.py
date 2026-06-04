"""
Phase 2a: Multi-Head Self-Attention
-------------------------------------
The core of every transformer (and our dLLM backbone).

Attention answers: "which other tokens should I look at when
processing this token?"

  python 02_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── 1. What is attention? ─────────────────────────────────────────────────────
# For each token, attention computes a weighted sum over all other tokens.
# Tokens that are "relevant" get high weight, others get near-zero.
#
# The formula:  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
#
#   Q (Query)  = "what am I looking for?"
#   K (Key)    = "what do I contain?"
#   V (Value)  = "what do I actually send if selected?"

class SelfAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads   # each head sees a slice

        # single linear that projects to Q, K, V all at once
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape   # batch, sequence length, hidden size

        # project to Q, K, V and split into heads
        qkv = self.qkv(x)                           # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)              # each [B, T, C]

        # reshape: [B, T, C] → [B, n_heads, T, head_dim]
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale  # [B, H, T, T]

        if mask is not None:
            # mask=True means "ignore this position" (e.g. padding)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)             # [B, H, T, T]
        out  = attn @ v                              # [B, H, T, head_dim]

        # merge heads back: [B, H, T, head_dim] → [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# ── 2. Test it ───────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

hidden, n_heads = 64, 4
attn = SelfAttention(hidden, n_heads).to(device)

B, T = 2, 10   # batch=2, sequence length=10
x = torch.randn(B, T, hidden).to(device)
out = attn(x)

print(f"Input:  {x.shape}")    # [2, 10, 64]
print(f"Output: {out.shape}")  # [2, 10, 64]  — same shape, richer representation

# ── 3. Masking — critical for dLLMs ──────────────────────────────────────────
# In a dLLM, some tokens are [MASK]. We want the model to attend to
# all non-masked tokens when predicting what the masked tokens should be.
# (Unlike autoregressive LMs that can only look left.)

# Example: positions 2 and 5 are masked
masked_positions = torch.zeros(B, T, dtype=torch.bool).to(device)
masked_positions[:, [2, 5]] = True
print(f"\nMasked positions: {masked_positions[0].tolist()}")

# The attention mask prevents attending FROM masked positions TO other masked positions
# (optional refinement — basic dLLM just attends to everything including masks)
out_masked = attn(x)   # for now, attend to all — masking is handled in the loss
print(f"Output with masking awareness: {out_masked.shape}")

# ── 4. Key insight: dLLM vs autoregressive attention ─────────────────────────
print("""
Key difference:
  Autoregressive LLM  → causal mask (each token only sees past tokens)
  dLLM                → NO causal mask (each token sees ALL tokens, even future)

This is why dLLMs can plan in any order — full bidirectional attention.
The model sees the whole sequence and decides what to fill in where.
""")

print("✅ Phase 2a complete — attention block built")
print("Next: 03_transformer.py — stack attention + FFN into a full transformer")
