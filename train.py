# train.py (torch)
from model.gpt_model import GPTModel
import os
import torch

# ====== hyperparams ======
batch_size = 16
block_size = 64
steps = 100000
resume_path = None  # e.g. "checkpoints/step_9000.pt"

# optimizer hyperparams
lr = 1e-3
weight_decay = 0.0

os.makedirs("checkpoints", exist_ok=True)

# 1) init model
model = GPTModel(vocab_size=256, embed_dim=32)  # auto picks cuda if available

# create optimizer in train.py (mainstream practice)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# 2) optional resume
start_step = 0
if resume_path is not None and os.path.exists(resume_path):
    meta = model.load(resume_path)
    start_step = int(meta.get("step", 0))
    print(f"[Resume] Loaded from {resume_path}, start from {start_step}")
else:
    if not os.path.exists("checkpoints/init.pt"):
        model.save("checkpoints/init.pt", step=0)
    print("[Start] Training from scratch")

# 3) read data
with open("data/input.txt", "r") as f:
    text = f.read()

tokens = model.tokenizer.encode(text)

# 4) train loop
model.train()
end_step = start_step + steps
for step in range(start_step, end_step):
    x, y = model.get_batch(tokens, batch_size, block_size)  # returns tensors on model.device

    logits = model(x)                 # [B, T, V]
    loss = model.loss(logits, y)      # scalar tensor

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        # grad norm of lm_head (projection) as a proxy
        grad_norm = None
        if model.lm_head.weight.grad is not None:
            grad_norm = model.lm_head.weight.grad.norm().item()
        print(f"step {step:6d} | loss {loss.item():.4f} | ||grad_lm_head|| {0.0 if grad_norm is None else grad_norm:.4f}")

    if step > 0 and step % 1000 == 0:
        model.save(f"checkpoints/step_{step}.pt", step=step)

model.save("checkpoints/final.pt", step=end_step)
print("Training completed.")