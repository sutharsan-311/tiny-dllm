"""
Phase 1: PyTorch Tensors + Autograd
------------------------------------
Goal: understand the building blocks before we build the model.
Run this file top to bottom. Each section teaches one concept
you will use directly in the dLLM.

  python 01_tensors.py
"""

import torch
import torch.nn as nn

# ── 1. What is a tensor? ─────────────────────────────────────────────────────
# A tensor is just a multi-dimensional array — like numpy, but GPU-capable
# and able to track gradients automatically.

x = torch.tensor([1.0, 2.0, 3.0])
print("1D tensor:", x)
print("shape:", x.shape)        # torch.Size([3])
print("dtype:", x.dtype)        # torch.float32

# 2D tensor (like a matrix)
mat = torch.tensor([[1.0, 2.0],
                    [3.0, 4.0]])
print("\n2D tensor:\n", mat)
print("shape:", mat.shape)      # torch.Size([2, 2])

# ── 2. The shapes you'll see in a language model ──────────────────────────────
# Every piece of text becomes a 3D tensor:  [batch, sequence, hidden]
#   batch    = how many sentences at once
#   sequence = how many tokens (words/characters) in each sentence
#   hidden   = how many numbers represent each token

batch, seq, hidden = 2, 8, 16
x = torch.randn(batch, seq, hidden)   # random normal values
print(f"\nTypical LM tensor shape: {x.shape}")   # [2, 8, 16]

# ── 3. Moving to GPU ─────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")     # 'cuda' on your RTX 3050

x = x.to(device)
print("Tensor on GPU:", x.device)

# ── 4. Autograd — automatic gradients ────────────────────────────────────────
# This is the magic behind training. PyTorch tracks every operation
# on tensors that have requires_grad=True, then can compute d(loss)/d(param)
# automatically via .backward().

w = torch.tensor(2.0, requires_grad=True)   # a learnable weight
x = torch.tensor(3.0)

y = w * x           # forward pass: y = 2 * 3 = 6
loss = (y - 10)**2  # pretend 10 is our target: loss = (6-10)^2 = 16

loss.backward()     # compute gradients

print(f"\nw={w.item()}, y={y.item()}, loss={loss.item()}")
print(f"d(loss)/d(w) = {w.grad.item()}")   # -16.0
# Math check: d/dw [(wx - 10)^2] = 2(wx-10)*x = 2*(6-10)*3 = -24 ✓

# ── 5. nn.Module — how every model is built ──────────────────────────────────
# Every layer, block, and full model inherits from nn.Module.
# You define __init__ (create layers) and forward (define computation).

class TinyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias   # matrix multiply + bias

model = TinyLinear(16, 8).to(device)
x = torch.randn(2, 16).to(device)          # batch=2, in=16
out = model(x)
print(f"\nTinyLinear: {x.shape} → {out.shape}")   # [2, 16] → [2, 8]

# Parameters are auto-tracked
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params}")        # 16*8 + 8 = 136

# ── 6. A minimal training loop ────────────────────────────────────────────────
# This is the skeleton of every training loop you'll ever write in PyTorch:
#   1. forward  → compute prediction
#   2. loss     → how wrong are we?
#   3. backward → compute gradients
#   4. step     → update weights
#   5. zero_grad → clear gradients for next step

model = TinyLinear(4, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# fake data: learn y = sum(x)
X = torch.randn(100, 4).to(device)
Y = X.sum(dim=1, keepdim=True)

print("\nTraining TinyLinear to learn y = sum(x):")
for step in range(200):
    pred = model(X)           # 1. forward
    loss = loss_fn(pred, Y)   # 2. loss
    loss.backward()           # 3. backward
    optimizer.step()          # 4. update weights
    optimizer.zero_grad()     # 5. clear gradients

    if step % 50 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}")

print("\n✅ Phase 1 complete. You now understand:")
print("   - Tensors and shapes [batch, seq, hidden]")
print("   - GPU placement (.to(device))")
print("   - Autograd (loss.backward())")
print("   - nn.Module (how all models are built)")
print("   - Training loop (forward → loss → backward → step → zero_grad)")
print("\nNext: 02_attention.py — build multi-head attention from scratch")
