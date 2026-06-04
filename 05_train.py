"""
Phase 4: Train on TinyShakespeare
-----------------------------------
Downloads Shakespeare text, builds a character-level vocab,
and trains the dLLM to denoise masked sequences.

  python 05_train.py

Checkpoints saved to checkpoints/dllm.pt every 500 steps.
Training ~1-2 hrs on RTX 3050 for 5000 steps.
"""

import os, math, time, json, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Download data ─────────────────────────────────────────────────────────────
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"

def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Done.")
    with open(DATA_PATH) as f:
        return f.read()


# ── Character-level tokenizer ─────────────────────────────────────────────────
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.c2i = {c: i for i, c in enumerate(chars)}
        self.i2c = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self.c2i[c] for c in text]

    def decode(self, ids):
        return ''.join(self.i2c.get(i, '?') for i in ids)

    def save(self, path):
        json.dump(self.c2i, open(path, 'w'))

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.c2i = json.load(open(path))
        obj.i2c = {int(v): k for k, v in obj.c2i.items()}
        obj.vocab_size = len(obj.c2i)
        return obj


# ── Model (same as 04) ────────────────────────────────────────────────────────
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
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = SelfAttention(hidden, n_heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = FeedForward(hidden)
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
        self.head.weight = self.token_emb.weight[:vocab_size]
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.head(self.norm(self.blocks(x)))


# ── Diffusion ─────────────────────────────────────────────────────────────────
class MaskedDiffusion:
    def __init__(self, mask_token_id):
        self.mask_id = mask_token_id

    def loss(self, model, tokens):
        B, T = tokens.shape
        t = torch.rand(B, device=tokens.device)
        mask_prob = t.unsqueeze(1).expand(B, T)
        mask = torch.bernoulli(mask_prob).bool()
        noisy = tokens.clone()
        noisy[mask] = self.mask_id
        logits = model(noisy)
        logits_masked = logits[mask]
        targets = tokens[mask]
        if logits_masked.numel() == 0:
            return torch.tensor(0.0, device=tokens.device)
        return F.cross_entropy(logits_masked, targets)

    @torch.no_grad()
    def sample(self, model, seq_len, n_steps=20, device='cpu'):
        model.eval()
        tokens = torch.full((1, seq_len), self.mask_id, dtype=torch.long, device=device)
        for step in range(n_steps):
            frac = (step + 1) / n_steps
            target = int(frac * seq_len)
            logits = model(tokens)
            probs  = F.softmax(logits, dim=-1)
            predicted = torch.multinomial(probs.view(seq_len, -1), 1).view(1, seq_len)
            confidence, _ = probs.max(dim=-1)
            still_masked = (tokens == self.mask_id)
            confidence[~still_masked] = -1.0
            already = (~still_masked).sum().item()
            to_unmask = max(0, target - already)
            if to_unmask > 0 and still_masked.any():
                _, idx = confidence.view(-1).topk(min(to_unmask, still_masked.sum().item()))
                flat = tokens.view(-1)
                flat[idx] = predicted.view(-1)[idx]
                tokens = flat.view(1, seq_len)
        model.train()
        return tokens


# ── Data loader ───────────────────────────────────────────────────────────────
def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x  = torch.stack([data[i:i+seq_len] for i in ix])
    return x.to(device)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    # config
    HIDDEN    = 256
    N_LAYERS  = 4
    N_HEADS   = 4
    SEQ_LEN   = 128
    BATCH     = 32
    LR        = 3e-4
    STEPS     = 5000
    EVAL_INT  = 100
    SAVE_INT  = 500
    WARMUP    = 200

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # data
    text      = download_data()
    tokenizer = CharTokenizer(text)
    data      = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split     = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    print(f"Vocab size:  {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,}")

    os.makedirs("checkpoints", exist_ok=True)
    tokenizer.save("checkpoints/vocab.json")

    # model
    model     = TinyDLLM(tokenizer.vocab_size, HIDDEN, N_LAYERS, N_HEADS, SEQ_LEN).to(device)
    diffusion = MaskedDiffusion(tokenizer.mask_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # linear warmup + cosine decay schedule
    def lr_schedule(step):
        if step < WARMUP:
            return step / WARMUP
        progress = (step - WARMUP) / (STEPS - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

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
            # val loss
            with torch.no_grad():
                val_batch = get_batch(val_data, SEQ_LEN, BATCH, device)
                val_loss  = diffusion.loss(model, val_batch)

            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0] * LR
            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss.item():.4f} "
                  f"| lr {lr_now:.2e} | {elapsed:.0f}s")

            # sample
            ids  = diffusion.sample(model, seq_len=80, n_steps=20, device=device)
            text_out = tokenizer.decode(ids[0].tolist())
            print(f"  sample: {repr(text_out[:80])}")
            t0 = time.time()

        if step % SAVE_INT == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'vocab_size': tokenizer.vocab_size,
                'hidden': HIDDEN, 'n_layers': N_LAYERS,
                'n_heads': N_HEADS, 'seq_len': SEQ_LEN,
            }, f"checkpoints/dllm_step{step}.pt")
            print(f"  ✅ Saved checkpoint at step {step}")


if __name__ == '__main__':
    train()
