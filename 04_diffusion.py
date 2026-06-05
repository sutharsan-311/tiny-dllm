"""
Phase 3: Masked Diffusion — the dLLM core
-------------------------------------------
This is what makes it a *diffusion* LM, not just a regular transformer.

Concept:
  Forward process  (add noise):   text → gradually mask tokens → fully masked
  Backward process (denoise):     [MASK]...[MASK] → gradually unmask → text

The model learns: given a partially masked sequence, predict what the masked tokens are.

  python 04_diffusion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Minimal model (copy from 03) ─────────────────────────────────────────────
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
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = SelfAttention(hidden, n_heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = FeedForward(hidden)
    def forward(self, x):
        return x + self.ff(self.norm2(x + self.attn(self.norm1(x))))


# ── The dLLM model ────────────────────────────────────────────────────────────
class TinyDLLM(nn.Module):
    def __init__(self, vocab_size, hidden=256, n_layers=4, n_heads=4, max_seq=128):
        super().__init__()
        # vocab_size + 1 because we add a special [MASK] token at index vocab_size
        self.mask_token_id = vocab_size
        full_vocab = vocab_size + 1

        self.token_emb = nn.Embedding(full_vocab, hidden)
        self.pos_emb   = nn.Embedding(max_seq, hidden)
        self.blocks    = nn.Sequential(*[TransformerBlock(hidden, n_heads)
                                         for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)  # predict real tokens only
        self.head.weight = nn.Parameter(self.token_emb.weight[:vocab_size])

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        x = self.norm(self.blocks(x))
        return self.head(x)   # [B, T, vocab_size]


# ── Masked Diffusion ──────────────────────────────────────────────────────────
class MaskedDiffusion:
    """
    Forward process: randomly mask tokens with probability t ∈ [0, 1].
      t=0 → no masking (original text)
      t=1 → fully masked

    We sample t uniformly each training step — the model must learn
    to denoise at ALL noise levels simultaneously.
    """

    def __init__(self, mask_token_id):
        self.mask_id = mask_token_id

    def add_noise(self, tokens, t):
        """
        Mask each token independently with probability t.
        tokens: [B, T]  original token ids
        t:      [B]     noise level per sample (0=clean, 1=all masked)
        Returns: noisy_tokens [B, T], mask [B, T] (True = was masked)
        """
        B, T = tokens.shape
        # t is per-sample; broadcast to [B, T]
        mask_prob = t.unsqueeze(1).expand(B, T)
        mask = torch.bernoulli(mask_prob).bool()       # True = mask this token
        noisy = tokens.clone()
        noisy[mask] = self.mask_id
        return noisy, mask

    def loss(self, model, tokens):
        """
        Training loss:
          1. Sample random noise level t ~ Uniform(0, 1) per sample
          2. Apply forward process (mask tokens)
          3. Model predicts original tokens at masked positions
          4. Cross-entropy only on masked positions (nothing to learn at visible ones)
        """
        B, T = tokens.shape
        device = tokens.device

        t = torch.rand(B, device=device)               # random noise level
        noisy_tokens, mask = self.add_noise(tokens, t)

        logits = model(noisy_tokens)                   # [B, T, vocab]

        # only compute loss where we masked
        logits_masked = logits[mask]                   # [n_masked, vocab]
        targets       = tokens[mask]                   # [n_masked]

        if logits_masked.numel() == 0:
            return torch.tensor(0.0, device=device)

        return F.cross_entropy(logits_masked, targets)

    @torch.no_grad()
    def sample(self, model, seq_len, n_steps=20, device='cpu'):
        """
        Generate text from scratch:
          Start fully masked → iteratively unmask tokens over n_steps.
          Each step unmasks a fraction of the most confident predictions.
        """
        model.eval()
        B = 1
        # start: everything masked
        tokens = torch.full((B, seq_len), self.mask_id, dtype=torch.long, device=device)

        for step in range(n_steps):
            # how many tokens to unmask this step
            # unmask gradually: step 0 reveals fewest, step n_steps-1 reveals rest
            frac_unmasked = (step + 1) / n_steps
            target_unmasked = int(frac_unmasked * seq_len)

            logits = model(tokens)                          # [1, T, vocab]
            probs  = F.softmax(logits, dim=-1)              # [1, T, vocab]

            # sample token predictions everywhere
            predicted = torch.multinomial(
                probs.view(B * seq_len, -1), num_samples=1).view(B, seq_len)

            # confidence = max probability at each position
            confidence, _ = probs.max(dim=-1)              # [1, T]

            # only consider currently masked positions
            still_masked = (tokens == self.mask_id)
            confidence[~still_masked] = -1.0               # ignore already-unmasked

            # unmask the most confident positions up to target_unmasked total
            currently_unmasked = (~still_masked).sum().item()
            to_unmask = max(0, target_unmasked - currently_unmasked)

            if to_unmask > 0 and still_masked.any():
                _, top_idx = confidence.view(-1).topk(min(to_unmask, still_masked.sum().item()))
                flat_tokens = tokens.view(-1)
                flat_pred   = predicted.view(-1)
                flat_tokens[top_idx] = flat_pred[top_idx]
                tokens = flat_tokens.view(B, seq_len)

        return tokens


# ── Demo ──────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB   = 65
HIDDEN  = 256
LAYERS  = 4
HEADS   = 4
MAX_SEQ = 128

model    = TinyDLLM(VOCAB, HIDDEN, LAYERS, HEADS, MAX_SEQ).to(device)
diffusion = MaskedDiffusion(mask_token_id=VOCAB)

# test loss on fake data
fake_tokens = torch.randint(0, VOCAB, (4, MAX_SEQ)).to(device)
loss = diffusion.loss(model, fake_tokens)
print(f"Loss on random data (untrained): {loss.item():.4f}")
print(f"  (expected ~ln({VOCAB}) = {math.log(VOCAB):.2f} for random model)")

# test forward process
tokens = torch.randint(0, VOCAB, (1, 10)).to(device)
t      = torch.tensor([0.5]).to(device)
noisy, mask = diffusion.add_noise(tokens, t)
print(f"\nOriginal: {tokens[0].tolist()}")
print(f"Noisy (t=0.5): {noisy[0].tolist()}  ({VOCAB}=MASK)")
print(f"Masked positions: {mask[0].tolist()}")

# test sampling (untrained = random output, but pipeline works)
generated = diffusion.sample(model, seq_len=20, n_steps=10, device=device)
print(f"\nGenerated (untrained): {generated[0].tolist()}")

print("""
What just happened:
  1. add_noise()  — masked 50% of tokens randomly (forward process)
  2. model()      — predicted all token positions (bidirectional attention!)
  3. sample()     — started fully masked, unmasked most-confident tokens step by step

✅ Phase 3 complete — diffusion process built
Next: 05_train.py — actually train this on Shakespeare text
""")
