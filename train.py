# train.py
from model.gpt_model import GPTModel
import numpy as np
import os

batch_size = 16
block_size = 64
steps = 100000
resume_path = None

os.makedirs("checkpoints", exist_ok=True)

model = GPTModel()
start_step = 0

if resume_path is not None and os.path.exists(resume_path):
    meta = model.load(resume_path)
    start_step = int(meta.get("step", 0))
    print(f"[Resume] Loaded from {resume_path}, start from {start_step}")
else:
    if not os.path.exists("checkpoints/init.npz"):
        model.save("checkpoints/init.npz", step=0)
    print("[Start] Training from scratch")

text = open('data/input.txt', 'r').read()
tokens = model.tokenizer.encode(text)

end_step = start_step + steps
for step in range(start_step, end_step):
    x, y = model.get_batch(tokens, batch_size, block_size)

    logits = model.forward(x)
    loss = model.cross_entropy(logits, y)

    model.backward(loss)
    model.optimizer.step()

    if step % 100 == 0:
        grad_w = np.linalg.norm(model.optimizer.grads[1])
        print(f"step {step:6d} | loss {loss:.4f} | ||grad_proj|| {grad_w:.4f}")

    if step > 0 and step % 1000 == 0:
        model.save(f"checkpoints/step_{step}.npz", step=step)

model.save("checkpoints/final.npz", step=end_step)
print("Training completed.")