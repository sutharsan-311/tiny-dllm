"""
Phase 6: Train on Tamil Text (Thirukkural + Sangam poetry)
------------------------------------------------------------
Same dLLM architecture as 05_train.py — only the tokenizer and data change.

Tamil uses Unicode (U+0B80–U+0BFF). Character-level tokenizer handles it
naturally since Python strings are Unicode-aware.

Run:
  python tamil_dataset.py          # download data first
  python 08_train_tamil.py         # train

Checkpoint saved to checkpoints/tamil_dllm_stepN.pt
"""

import os, math, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tamil-aware character tokenizer ──────────────────────────────────────────
class TamilTokenizer:
    """
    Character-level tokenizer for Tamil Unicode text.
    Tamil block: U+0B80–U+0BFF (vowels, consonants, compound glyphs).
    We keep Tamil chars + basic punctuation + space/newline.
    """
    def __init__(self, text):
        # keep Tamil block + useful ASCII
        def keep(c):
            cp = ord(c)
            return (0x0B80 <= cp <= 0x0BFF) or c in ' \n.,!?;:\'"()-–'
        chars = sorted(set(c for c in text if keep(c)))
        self.vocab_size = len(chars)
        self.c2i = {c: i for i, c in enumerate(chars)}
        self.i2c = {i: c for i, c in enumerate(chars)}
        print(f"Tamil vocab size: {self.vocab_size} unique characters")
        tamil_only = [c for c in chars if 0x0B80 <= ord(c) <= 0x0BFF]
        print(f"  Tamil chars: {len(tamil_only)}  |  Other: {self.vocab_size - len(tamil_only)}")

    def encode(self, text):
        return [self.c2i[c] for c in text if c in self.c2i]

    def decode(self, ids):
        return ''.join(self.i2c.get(i, '') for i in ids)

    def save(self, path):
        json.dump({k: v for k, v in self.c2i.items()}, open(path, 'w', encoding='utf-8'),
                  ensure_ascii=False)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.c2i = json.load(open(path, encoding='utf-8'))
        obj.i2c = {int(v): k for k, v in obj.c2i.items()}
        obj.vocab_size = len(obj.c2i)
        return obj


# ── Model (same architecture as 05_train.py) ─────────────────────────────────
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
        def split(t): return t.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        out = (F.softmax(scores,-1) @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 4*hidden), nn.GELU(), nn.Linear(4*hidden, hidden))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden); self.attn = SelfAttention(hidden, n_heads)
        self.norm2 = nn.LayerNorm(hidden); self.ff   = FeedForward(hidden)
    def forward(self, x):
        return x + self.ff(self.norm2(x + self.attn(self.norm1(x))))

class TinyDLLM(nn.Module):
    def __init__(self, vocab_size, hidden=256, n_layers=4, n_heads=4, max_seq=128):
        super().__init__()
        self.mask_token_id = vocab_size
        full_vocab = vocab_size + 1
        self.token_emb = nn.Embedding(full_vocab, hidden)
        self.pos_emb   = nn.Embedding(max_seq, hidden)
        self.blocks    = nn.Sequential(*[TransformerBlock(hidden, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = nn.Parameter(self.token_emb.weight[:vocab_size])
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.head(self.norm(self.blocks(x)))


# ── Masked Diffusion ──────────────────────────────────────────────────────────
class MaskedDiffusion:
    def __init__(self, mask_token_id):
        self.mask_id = mask_token_id

    def loss(self, model, tokens):
        B, T = tokens.shape
        t = torch.rand(B, device=tokens.device)
        mask = torch.bernoulli(t.unsqueeze(1).expand(B, T)).bool()
        noisy = tokens.clone()
        noisy[mask] = self.mask_id
        logits = model(noisy)
        logits_m, targets = logits[mask], tokens[mask]
        if logits_m.numel() == 0:
            return torch.tensor(0.0, device=tokens.device)
        return F.cross_entropy(logits_m, targets)

    @torch.no_grad()
    def sample(self, model, seq_len, n_steps=20, device='cpu'):
        model.eval()
        tokens = torch.full((1, seq_len), self.mask_id, dtype=torch.long, device=device)
        for step in range(n_steps):
            target = int((step + 1) / n_steps * seq_len)
            probs  = F.softmax(model(tokens), dim=-1)
            pred   = torch.multinomial(probs.view(seq_len, -1), 1).view(1, seq_len)
            conf, _ = probs.max(dim=-1)
            masked  = (tokens == self.mask_id)
            conf[~masked] = -1.0
            to_unmask = max(0, target - (~masked).sum().item())
            if to_unmask > 0 and masked.any():
                _, idx = conf.view(-1).topk(min(to_unmask, masked.sum().item()))
                flat = tokens.view(-1)
                flat[idx] = pred.view(-1)[idx]
                tokens = flat.view(1, seq_len)
        model.train()
        return tokens


def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    return torch.stack([data[i:i+seq_len] for i in ix]).to(device)


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    TAMIL_PATH = 'data/tamil.txt'
    if not os.path.exists(TAMIL_PATH):
        print("Tamil dataset not found.")
        print("Run first: python tamil_dataset.py")
        return

    # config — slightly smaller batch since Tamil vocab is larger
    HIDDEN   = 256
    N_LAYERS = 4
    N_HEADS  = 4
    SEQ_LEN  = 128
    BATCH    = 24
    LR       = 3e-4
    STEPS    = 5000
    EVAL_INT = 100
    SAVE_INT = 500
    WARMUP   = 200

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    with open(TAMIL_PATH, encoding='utf-8') as f:
        text = f.read()

    tokenizer = TamilTokenizer(text)
    data      = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split     = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    os.makedirs('checkpoints', exist_ok=True)
    tokenizer.save('checkpoints/tamil_vocab.json')

    model     = TinyDLLM(tokenizer.vocab_size, HIDDEN, N_LAYERS, N_HEADS, SEQ_LEN).to(device)
    diffusion = MaskedDiffusion(tokenizer.mask_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    def lr_schedule(step):
        if step < WARMUP: return step / WARMUP
        p = (step - WARMUP) / (STEPS - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}\n")

    model.train()
    t0 = time.time()

    for step in range(1, STEPS + 1):
        batch = get_batch(train_data, SEQ_LEN, BATCH, device)
        loss  = diffusion.loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_INT == 0:
            with torch.no_grad():
                val_loss = diffusion.loss(model, get_batch(val_data, SEQ_LEN, BATCH, device))
            elapsed = time.time() - t0
            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss.item():.4f} "
                  f"| {elapsed:.0f}s")
            ids  = diffusion.sample(model, seq_len=60, n_steps=20, device=device)
            sample_text = tokenizer.decode(ids[0].tolist())
            print(f"  sample: {sample_text}")
            t0 = time.time()

        if step % SAVE_INT == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'vocab_size': tokenizer.vocab_size,
                'hidden': HIDDEN, 'n_layers': N_LAYERS,
                'n_heads': N_HEADS, 'seq_len': SEQ_LEN,
                'lang': 'tamil',
            }, f'checkpoints/tamil_dllm_step{step}.pt')
            print(f"  ✅ Saved checkpoint at step {step}")


if __name__ == '__main__':
    train()
