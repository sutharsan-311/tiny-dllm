"""
Phase 2b: Full Transformer Backbone
--------------------------------------
Stack attention + feed-forward into transformer blocks,
then build the complete tiny model (~10M params).

  python 03_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Re-use attention from phase 2 ────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out  = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# ── Feed-Forward Network ──────────────────────────────────────────────────────
# After attention mixes information across tokens,
# FFN processes each token independently (position-wise).
# Rule of thumb: FFN hidden = 4 × model hidden.

class FeedForward(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),               # smooth activation (better than ReLU for LMs)
            nn.Linear(4 * hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block = Attention + FFN + LayerNorm + Residuals ──────────────
# Residual connections ("x + ...") let gradients flow cleanly during training.
# LayerNorm stabilizes activations so training doesn't explode or vanish.

class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = SelfAttention(hidden, n_heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = FeedForward(hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # attention with residual
        x = x + self.ff(self.norm2(x))     # FFN with residual
        return x


# ── The Full Tiny dLLM Backbone ───────────────────────────────────────────────
# Architecture:
#   Token embedding → Position embedding → N transformer blocks → LayerNorm → head
#
# "head" = final linear that maps hidden → vocab_size (predicts which token)

class TinyDLLM(nn.Module):
    def __init__(self, vocab_size, hidden=256, n_layers=4, n_heads=4, max_seq=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb   = nn.Embedding(max_seq, hidden)
        self.blocks    = nn.Sequential(*[TransformerBlock(hidden, n_heads)
                                         for _ in range(n_layers)])
        self.norm      = nn.LayerNorm(hidden)
        self.head      = nn.Linear(hidden, vocab_size, bias=False)

        # weight tying: share token embedding and output projection weights
        # (common trick — saves params, improves quality)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos  = torch.arange(T, device=token_ids.device)

        x = self.token_emb(token_ids) + self.pos_emb(pos)   # [B, T, hidden]
        x = self.blocks(x)                                   # [B, T, hidden]
        x = self.norm(x)                                     # [B, T, hidden]
        return self.head(x)                                  # [B, T, vocab_size]


# ── Instantiate and inspect ───────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB   = 65      # Shakespeare character-level vocab (a-z, A-Z, punctuation...)
HIDDEN  = 256
LAYERS  = 4
HEADS   = 4
MAX_SEQ = 128

model = TinyDLLM(VOCAB, HIDDEN, LAYERS, HEADS, MAX_SEQ).to(device)

total = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total:,}")   # ~10M

# test forward pass
x = torch.randint(0, VOCAB, (2, MAX_SEQ)).to(device)   # fake token ids
logits = model(x)
print(f"Input:   {x.shape}")       # [2, 128]
print(f"Logits:  {logits.shape}")  # [2, 128, 65]  — score for each vocab item at each position

print("""
Architecture summary:
  Token IDs [B, T]
      ↓ embedding
  [B, T, 256]
      ↓ × 4 transformer blocks (attention + FFN)
  [B, T, 256]
      ↓ LayerNorm + Linear
  Logits [B, T, 65]   ← "what token should be here?"
""")

print("✅ Phase 2b complete — transformer backbone built")
print("Next: 04_diffusion.py — add masking + denoising (the actual diffusion part)")
