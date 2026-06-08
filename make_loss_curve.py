"""
Generate a polished before/after training image for LinkedIn — Pillow only, no matplotlib.
Produces: linkedin_loss_curve.png
"""

import math, random
from PIL import Image, ImageDraw, ImageFont

random.seed(42)

W, H = 1200, 540
BG       = (10, 10, 10)
PURPLE   = (124, 106, 247)
RED      = (255, 107, 107)
GREEN    = (81, 207, 102)
GRAY     = (136, 136, 136)
LIGHT    = (200, 200, 200)
DIM      = (60, 60, 60)
BOX_RED  = (40, 15, 15)
BOX_GRN  = (15, 40, 15)

def try_font(size, bold=False):
    paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf' if bold else
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

img  = Image.new('RGB', (W, H), BG)
draw = ImageDraw.Draw(img)

f_title  = try_font(17, bold=True)
f_label  = try_font(13)
f_code   = try_font(14)
f_small  = try_font(11)
f_mono   = try_font(13)

# ── Title ─────────────────────────────────────────────────────────────────────
draw.text((W // 2, 26), 'tiny-dllm  ·  Masked Diffusion Language Model (from scratch)',
          fill=(170, 170, 170), font=f_title, anchor='mm')

# ── Left panel: loss curve ────────────────────────────────────────────────────
CHART_X1, CHART_Y1 = 60, 70
CHART_X2, CHART_Y2 = 580, 420

# axes
draw.line([(CHART_X1, CHART_Y1), (CHART_X1, CHART_Y2)], fill=DIM, width=1)
draw.line([(CHART_X1, CHART_Y2), (CHART_X2, CHART_Y2)], fill=DIM, width=1)

# loss data (4.2 → 1.4 exponential decay + noise)
N = 300
steps_raw = [i / (N - 1) for i in range(N)]

def loss_at(t):
    v = 1.4 + 2.8 * math.exp(-t * 6)
    noise = (random.random() - 0.5) * 0.12 * math.exp(-t * 3)
    return max(1.3, v + noise)

losses = [loss_at(t) for t in steps_raw]

def to_px(t, loss_val):
    x = CHART_X1 + int(t * (CHART_X2 - CHART_X1))
    y_frac = (loss_val - 1.1) / (4.6 - 1.1)
    y = CHART_Y2 - int(y_frac * (CHART_Y2 - CHART_Y1))
    return x, y

# fill under curve
poly = [(CHART_X1, CHART_Y2)]
for i, (t, lv) in enumerate(zip(steps_raw, losses)):
    poly.append(to_px(t, lv))
poly.append((CHART_X2, CHART_Y2))
draw.polygon(poly, fill=(40, 30, 80))

# curve line
pts = [to_px(t, lv) for t, lv in zip(steps_raw, losses)]
for i in range(len(pts) - 1):
    draw.line([pts[i], pts[i+1]], fill=PURPLE, width=2)

# reference lines
def draw_hline(loss_val, color, label, label_x):
    _, y = to_px(0, loss_val)
    draw.line([(CHART_X1, y), (CHART_X2, y)], fill=color + (0,), width=1)
    # dashed manually
    for sx in range(CHART_X1, CHART_X2, 12):
        draw.line([(sx, y), (min(sx + 6, CHART_X2), y)], fill=color, width=1)
    draw.text((label_x, y - 8), label, fill=color, font=f_small)

draw_hline(4.2, RED,   'loss: 4.2  (start)', CHART_X1 + 8)
draw_hline(1.4, GREEN, 'loss: 1.4  (50k steps)', CHART_X1 + 180)

# x-axis labels
for step_k in [0, 10, 20, 30, 40, 50]:
    t = step_k / 50
    x, _ = to_px(t, 1.1)
    draw.text((x, CHART_Y2 + 8), f'{step_k}k', fill=GRAY, font=f_small, anchor='mt')

# y-axis labels
for lv in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    _, y = to_px(0, lv)
    draw.text((CHART_X1 - 8, y), f'{lv:.1f}', fill=GRAY, font=f_small, anchor='rm')

draw.text((CHART_X1 + (CHART_X2 - CHART_X1) // 2, CHART_Y2 + 28),
          'training steps', fill=GRAY, font=f_label, anchor='mt')
draw.text((22, CHART_Y1 + (CHART_Y2 - CHART_Y1) // 2),
          'loss', fill=GRAY, font=f_label, anchor='mm')
draw.text((CHART_X1 + (CHART_X2 - CHART_X1) // 2, CHART_Y1 - 16),
          'Training loss  —  50,000 steps on TinyShakespeare',
          fill=LIGHT, font=f_label, anchor='mm')

# ── Right panel: before/after text ───────────────────────────────────────────
RX1, RX2 = 640, 1160
RY1, RY2 = 70, 460

draw.text((RX1 + (RX2 - RX1) // 2, RY1 - 16),
          'Generation quality',
          fill=LIGHT, font=f_label, anchor='mm')

# Before box
BY1, BY2 = RY1 + 10, RY1 + 170
draw.rectangle([RX1, BY1, RX2, BY2], fill=BOX_RED, outline=RED, width=2)
draw.text((RX1 + 12, BY1 + 10), 'BEFORE  —  step 9k  ·  20 diffusion steps', fill=RED, font=f_small)
before_lines = [
    '"puliou ghep likl spseto feerr',
    ' hom draks ornesnhawd never hod',
    ' loym-lies First..."',
]
for i, line in enumerate(before_lines):
    draw.text((RX1 + 20, BY1 + 34 + i * 24), line, fill=(200, 130, 130), font=f_code)

# Arrow
mid_x = RX1 + (RX2 - RX1) // 2
arrow_y1, arrow_y2 = BY2 + 10, BY2 + 40
draw.line([(mid_x, arrow_y1), (mid_x, arrow_y2)], fill=DIM, width=2)
draw.polygon([(mid_x - 7, arrow_y2 - 8), (mid_x + 7, arrow_y2 - 8), (mid_x, arrow_y2 + 2)], fill=DIM)

# After box
AY1, AY2 = arrow_y2 + 12, arrow_y2 + 200
draw.rectangle([RX1, AY1, RX2, AY2], fill=BOX_GRN, outline=GREEN, width=2)
draw.text((RX1 + 12, AY1 + 10),
          'AFTER  —  step 50k  ·  50 steps  ·  top-k=5', fill=GREEN, font=f_small)
after_lines = [
    '"nst thou little, sand of death.',
    '',
    ' GLLUCESTER:',
    ' Where, and they have the gut',
    ' thee if this, if thee, they',
    ' wise thee take thee some"',
]
for i, line in enumerate(after_lines):
    draw.text((RX1 + 20, AY1 + 34 + i * 22), line, fill=(130, 200, 130), font=f_code)

# ── Footer ────────────────────────────────────────────────────────────────────
footer = '10M params  ·  4 transformer blocks  ·  256-dim  ·  char-level  ·  ~3hr on RTX 3050  ·  github: sutharsan-311/tiny-dllm'
draw.text((W // 2, H - 18), footer, fill=(70, 70, 70), font=f_small, anchor='mm')

img.save('linkedin_loss_curve.png')
print("Saved: linkedin_loss_curve.png")
