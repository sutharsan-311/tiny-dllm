"""
Generate a demo GIF showing iterative denoising — for LinkedIn/Twitter.
Run this on your RTX 3050 laptop (needs checkpoint in checkpoints/).

  pip install pillow
  python make_gif.py

Produces: demo_denoising.gif
"""

import sys, math, json, os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


# ── Model (same as 06_generate.py) ───────────────────────────────────────────
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
    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        return self.head(self.norm(self.blocks(x)))


# ── Sampling with frame capture ───────────────────────────────────────────────
@torch.no_grad()
def sample_with_frames(model, seq_len, n_steps, temperature=0.8, top_k=5, device='cpu'):
    mask_id = model.mask_token_id
    tokens  = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    frames  = []

    for step in range(n_steps):
        frac   = (step + 1) / n_steps
        target = int(frac * seq_len)

        logits = model(tokens) / temperature
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold, _ = logits.topk(k, dim=-1)
            logits = logits.masked_fill(logits < threshold[..., -1:], float('-inf'))
        probs     = F.softmax(logits, dim=-1)
        predicted = torch.multinomial(probs.view(seq_len, -1), 1).view(1, seq_len)
        confidence, _ = probs.max(dim=-1)

        still_masked = (tokens == mask_id)
        confidence[~still_masked] = -1.0
        already   = (~still_masked).sum().item()
        to_unmask = max(0, target - already)

        if to_unmask > 0 and still_masked.any():
            _, idx = confidence.view(-1).topk(min(to_unmask, still_masked.sum().item()))
            flat = tokens.view(-1)
            flat[idx] = predicted.view(-1)[idx]
            tokens = flat.view(1, seq_len)

        frames.append((step + 1, tokens.clone(), still_masked.clone()))

    return frames


# ── Frame rendering ───────────────────────────────────────────────────────────
W, H = 720, 200
BG      = (10, 10, 10)
MASK_C  = (100, 90, 180)   # purple — still masked
REVEAL_C= (81, 207, 102)   # green  — just revealed
DONE_C  = (200, 200, 200)  # white  — already done
TITLE_C = (170, 170, 170)
DIM_C   = (80, 80, 80)

def try_font(size):
    for name in ['DejaVuSansMono', 'Courier', 'monospace', 'FreeMono']:
        try:
            return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', size)
        except Exception:
            pass
    return ImageFont.load_default()

def render_frame(step, n_steps, tokens_t, mask_t, prev_mask_t, i2c, seq_len):
    img  = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    font_sm = try_font(11)
    font_md = try_font(14)

    # Title
    draw.text((W//2, 14), 'tiny-dllm  ·  iterative denoising',
              fill=TITLE_C, font=font_sm, anchor='mm')

    # Step indicator
    pct = int(step / n_steps * 100)
    bar_w = int((W - 80) * step / n_steps)
    draw.rectangle([40, 30, W-40, 38], fill=(30, 30, 30))
    draw.rectangle([40, 30, 40 + bar_w, 38], fill=(100, 90, 180))
    draw.text((W//2, 50), f'step {step}/{n_steps}  ({pct}% unmasked)',
              fill=DIM_C, font=font_sm, anchor='mm')

    # Render tokens in a grid
    chars_per_row = 40
    char_w, char_h = 14, 22
    x0 = (W - chars_per_row * char_w) // 2
    y0 = 65

    newly_revealed = prev_mask_t & ~mask_t

    for i in range(min(seq_len, chars_per_row * 3)):
        row = i // chars_per_row
        col = i % chars_per_row
        x = x0 + col * char_w
        y = y0 + row * char_h

        tok = tokens_t[0, i].item()
        is_masked = mask_t[0, i].item()
        is_new    = newly_revealed[0, i].item()

        if is_masked:
            ch    = '▒'
            color = MASK_C
        elif is_new:
            ch    = i2c.get(tok, '?')
            color = REVEAL_C
        else:
            ch    = i2c.get(tok, '?')
            color = DONE_C

        draw.text((x, y), ch, fill=color, font=font_md)

    # Legend
    lx = 20
    ly = H - 22
    for label, col in [('masked', MASK_C), ('just revealed', REVEAL_C), ('settled', DONE_C)]:
        draw.rectangle([lx, ly+4, lx+10, ly+14], fill=col)
        draw.text((lx+14, ly), label, fill=DIM_C, font=font_sm)
        lx += 110

    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    ckpts = sorted(glob.glob('checkpoints/dllm_step*.pt'),
                   key=lambda p: int(p.split('step')[1].split('.')[0]))
    if not ckpts:
        print("No checkpoint found — run 05_train.py first.")
        sys.exit(1)
    ckpt_path = ckpts[-1]
    print(f"Loading {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    vocab = json.load(open('checkpoints/vocab.json'))
    i2c   = {int(v): k for k, v in vocab.items()}

    model = TinyDLLM(
        ckpt['vocab_size'], ckpt['hidden'],
        ckpt['n_layers'], ckpt['n_heads'], ckpt['seq_len']
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    seq_len = min(80, ckpt['seq_len'])  # keep it readable in GIF
    n_steps = 30

    print(f"Sampling {seq_len} chars over {n_steps} steps...")
    frames_data = sample_with_frames(model, seq_len, n_steps, temperature=0.8, top_k=5, device=device)

    imgs   = []
    prev_mask = torch.ones(1, seq_len, dtype=torch.bool)

    for step, tokens_t, mask_t in frames_data:
        img = render_frame(step, n_steps, tokens_t, mask_t, prev_mask, i2c, seq_len)
        imgs.append(img)
        prev_mask = mask_t

    # Hold last frame longer
    imgs += [imgs[-1]] * 8

    out = 'demo_denoising.gif'
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 optimize=False, duration=180, loop=0)
    print(f"Saved: {out}  ({len(imgs)} frames)")


if __name__ == '__main__':
    main()
