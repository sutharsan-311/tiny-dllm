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

        frames.append((step + 1, tokens.cpu().clone(), still_masked.cpu().clone()))

    return frames


# ── Frame rendering ───────────────────────────────────────────────────────────
W, H    = 700, 260
BG      = (13, 13, 13)
MASK_C  = (60, 55, 100)    # dim purple — masked placeholder
REVEAL_C= (81, 207, 102)   # bright green — just revealed
DONE_C  = (204, 204, 204)  # light grey — settled
TITLE_C = (150, 150, 150)
BAR_C   = (80, 70, 180)
DIM_C   = (65, 65, 65)

FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
]

def try_font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def render_frame(step, n_steps, tokens_t, mask_t, prev_mask_t, i2c, seq_len):
    img  = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    f_title = try_font(13)
    f_text  = try_font(16)
    f_small = try_font(11)

    # measure actual char width from font so spacing is exact
    bbox = f_text.getbbox('A')
    cw   = bbox[2] - bbox[0]   # character width
    ch   = bbox[3] - bbox[1] + 4  # character height + line gap

    # ── header ───────────────────────────────────────────────────────────────
    draw.text((W // 2, 16), 'tiny-dllm  ·  iterative denoising',
              fill=TITLE_C, font=f_title, anchor='mm')

    # progress bar
    bar_x1, bar_x2, bar_y = 30, W - 30, 30
    draw.rectangle([bar_x1, bar_y, bar_x2, bar_y + 6], fill=(28, 28, 28))
    filled = bar_x1 + int((bar_x2 - bar_x1) * step / n_steps)
    draw.rectangle([bar_x1, bar_y, filled, bar_y + 6], fill=BAR_C)

    pct = int(step / n_steps * 100)
    draw.text((W // 2, 46), f'step {step} / {n_steps}   —   {pct}% revealed',
              fill=DIM_C, font=f_small, anchor='mm')

    # ── text grid ────────────────────────────────────────────────────────────
    max_cols = (W - 60) // cw
    x0 = (W - max_cols * cw) // 2
    y0 = 62

    newly_revealed = prev_mask_t & ~mask_t

    for i in range(min(seq_len, max_cols * 4)):
        row = i // max_cols
        col = i % max_cols
        x   = x0 + col * cw
        y   = y0 + row * ch

        tok        = tokens_t[0, i].item()
        is_masked  = mask_t[0, i].item()
        is_new     = newly_revealed[0, i].item()

        if is_masked:
            ch_str = '_'
            color  = MASK_C
        elif is_new:
            ch_str = i2c.get(tok, '?')
            color  = REVEAL_C
        else:
            ch_str = i2c.get(tok, '?')
            color  = DONE_C

        draw.text((x, y), ch_str, fill=color, font=f_text)

    # ── legend ────────────────────────────────────────────────────────────────
    ly = H - 18
    items = [('_  masked', MASK_C), ('█  just revealed', REVEAL_C), ('█  settled', DONE_C)]
    lx = 24
    for label, col in items:
        draw.text((lx, ly), label, fill=col, font=f_small)
        lx += 160

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

    # Print final output text (copy this into make_loss_curve.py)
    last_tokens, last_mask = frames_data[-1][1], frames_data[-1][2]
    final_text = ''.join(i2c.get(last_tokens[0, i].item(), '?') for i in range(seq_len))
    print(f"\nFinal output:\n{final_text}\n")

    # Hold last frame longer
    imgs += [imgs[-1]] * 12

    out = 'demo_denoising.gif'
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 optimize=False, duration=180, loop=0)
    print(f"Saved: {out}  ({len(imgs)} frames)")


if __name__ == '__main__':
    main()
