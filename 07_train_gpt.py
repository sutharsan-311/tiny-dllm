"""
Phase 6: Train a tiny GPT on TinyShakespeare
---------------------------------------------
Same architecture, same data, same steps as 05_train.py (dLLM).
The only differences vs the dLLM:
  1. Causal mask in attention — each token only sees past tokens
  2. Loss is next-token prediction, not masked denoising
  3. No mask token needed

Run:
  python 07_train_gpt.py

Checkpoints saved to checkpoints/gpt_step<N>.pt every 1000 steps.
"""

import os
import glob
import math
import time
import json
import urllib.request
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


# ── Model ─────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, hidden, n_heads, max_seq):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out  = nn.Linear(hidden, hidden, bias=False)
        # causal mask — lower triangle of 1s, upper triangle blocked
        self.register_buffer('mask', torch.tril(torch.ones(max_seq, max_seq)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, max_seq):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = CausalSelfAttention(hidden, n_heads, max_seq)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = FeedForward(hidden)
    def forward(self, x):
        return x + self.ff(self.norm2(x + self.attn(self.norm1(x))))

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, hidden=384, n_layers=6, n_heads=6, max_seq=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb   = nn.Embedding(max_seq, hidden)
        self.blocks    = nn.Sequential(*[TransformerBlock(hidden, n_heads, max_seq) for _ in range(n_layers)])
        self.norm      = nn.LayerNorm(hidden)
        self.head      = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.head(self.norm(self.blocks(x)))

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=0, device='cpu'):
        self.eval()
        tokens = prompt_ids.clone()
        for _ in range(max_new_tokens):
            tokens_cropped = tokens[:, -128:]
            logits = self(tokens_cropped)[:, -1, :] / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                threshold, _ = logits.topk(k, dim=-1)
                logits = logits.masked_fill(logits < threshold[:, -1:], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        self.train()
        return tokens


# ── Data loader ───────────────────────────────────────────────────────────────
def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x  = torch.stack([data[i:i + seq_len] for i in ix])
    y  = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    HIDDEN   = 384
    N_LAYERS = 6
    N_HEADS  = 6
    SEQ_LEN  = 128
    BATCH    = 32
    LR       = 3e-4
    STEPS    = 50000
    EVAL_INT = 100
    SAVE_INT = 1000
    WARMUP   = 400

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    text      = download_data()
    tokenizer = CharTokenizer(text)
    data      = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split     = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    print(f"Vocab size:   {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,}")

    os.makedirs("checkpoints", exist_ok=True)
    tokenizer.save("checkpoints/vocab.json")

    model     = TinyGPT(tokenizer.vocab_size, HIDDEN, N_LAYERS, N_HEADS, SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    # resume from latest checkpoint if available
    ckpts = sorted(glob.glob('checkpoints/gpt_step*.pt'), key=lambda p: int(p.split('step')[1].split('.')[0]))
    start_step = 0
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        print(f"Resuming from step {start_step}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {total_params:,}")

    def lr_schedule(step):
        if step < WARMUP:
            return step / WARMUP
        progress = (step - WARMUP) / (STEPS - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    t0 = time.time()

    for step in range(start_step + 1, STEPS + 1):
        x, y = get_batch(train_data, SEQ_LEN, BATCH, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_INT == 0:
            with torch.no_grad():
                xv, yv = get_batch(val_data, SEQ_LEN, BATCH, device)
                val_loss = F.cross_entropy(model(xv).view(-1, tokenizer.vocab_size), yv.view(-1))

            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0] * LR
            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss.item():.4f} "
                  f"| lr {lr_now:.2e} | {elapsed:.0f}s")

            # sample — seed with newline so it starts like a new speech
            seed = torch.tensor([[tokenizer.c2i['\n']]], dtype=torch.long, device=device)
            ids  = model.generate(seed, max_new_tokens=80, temperature=0.8, top_k=5, device=device)
            print(f"  sample: {repr(tokenizer.decode(ids[0].tolist()[1:81]))}")
            t0 = time.time()

        if step % SAVE_INT == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'vocab_size': tokenizer.vocab_size,
                'hidden': HIDDEN, 'n_layers': N_LAYERS,
                'n_heads': N_HEADS, 'seq_len': SEQ_LEN,
            }, f"checkpoints/gpt_step{step}.pt")
            print(f"  Saved checkpoint at step {step}")


def generate_from_checkpoint(prompt="", max_new_tokens=200, temperature=0.8, top_k=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    ckpts = sorted(glob.glob('checkpoints/gpt_step*.pt'), key=lambda p: int(p.split('step')[1].split('.')[0]))
    if not ckpts:
        print("No checkpoints found in checkpoints/. Train first.")
        return
    ckpt_path = ckpts[-1]
    print(f"Loading {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    tokenizer = CharTokenizer.load("checkpoints/vocab.json")

    hidden   = ckpt.get('hidden',   384)
    n_layers = ckpt.get('n_layers', 6)
    n_heads  = ckpt.get('n_heads',  6)
    seq_len  = ckpt.get('seq_len',  128)

    model = TinyGPT(tokenizer.vocab_size, hidden, n_layers, n_heads, seq_len).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    if prompt:
        seed_ids = tokenizer.encode(prompt)
    else:
        seed_ids = [tokenizer.c2i['\n']]

    seed = torch.tensor([seed_ids], dtype=torch.long, device=device)
    ids  = model.generate(seed, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k, device=device)
    text = tokenizer.decode(ids[0].tolist())
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate text from the latest checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Seed text for generation')
    parser.add_argument('--tokens', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    if args.generate:
        generate_from_checkpoint(args.prompt, args.tokens, args.temperature, args.top_k)
    else:
        train()
