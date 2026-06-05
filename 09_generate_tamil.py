"""
Generate Tamil text from a trained checkpoint.

  python 09_generate_tamil.py
  python 09_generate_tamil.py --steps 30 --len 100
"""

import sys, math, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.token_emb = nn.Embedding(vocab_size + 1, hidden)
        self.pos_emb   = nn.Embedding(max_seq, hidden)
        self.blocks    = nn.Sequential(*[TransformerBlock(hidden, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = nn.Parameter(self.token_emb.weight[:vocab_size])
    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.head(self.norm(self.blocks(x)))


@torch.no_grad()
def sample(model, seq_len, n_steps, temp=1.0, device='cpu'):
    mask_id = model.mask_token_id
    tokens  = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    for step in range(n_steps):
        target = int((step + 1) / n_steps * seq_len)
        probs  = F.softmax(model(tokens) / temp, dim=-1)
        pred   = torch.multinomial(probs.view(seq_len, -1), 1).view(1, seq_len)
        conf, _ = probs.max(dim=-1)
        masked  = (tokens == mask_id)
        conf[~masked] = -1.0
        to_unmask = max(0, target - (~masked).sum().item())
        if to_unmask > 0 and masked.any():
            _, idx = conf.view(-1).topk(min(to_unmask, masked.sum().item()))
            flat = tokens.view(-1)
            flat[idx] = pred.view(-1)[idx]
            tokens = flat.view(1, seq_len)
        bar = '█' * int((step+1)/n_steps*20) + '░' * (20-int((step+1)/n_steps*20))
        print(f"\r  [{bar}] step {step+1}/{n_steps}", end='', flush=True)
    print()
    return tokens


def main():
    args    = sys.argv[1:]
    n_steps = int(args[args.index('--steps')+1]) if '--steps' in args else 20
    seq_len = int(args[args.index('--len')+1])   if '--len'   in args else 80
    temp    = float(args[args.index('--temp')+1]) if '--temp'  in args else 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpts = sorted(glob.glob('checkpoints/tamil_dllm_step*.pt'))
    if not ckpts:
        print("No Tamil checkpoint found. Run 08_train_tamil.py first.")
        return

    ckpt  = torch.load(ckpts[-1], map_location=device)
    vocab = json.load(open('checkpoints/tamil_vocab.json', encoding='utf-8'))
    i2c   = {int(v): k for k, v in vocab.items()}

    model = TinyDLLM(ckpt['vocab_size'], ckpt['hidden'],
                     ckpt['n_layers'], ckpt['n_heads'], ckpt['seq_len']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint: step {ckpt['step']}")

    tokens = sample(model, seq_len, n_steps, temp, device)
    text   = ''.join(i2c.get(i, '') for i in tokens[0].tolist())

    print("\n" + "─" * 50)
    print(text)
    print("─" * 50)


if __name__ == '__main__':
    main()
