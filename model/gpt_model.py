import numpy as np  
import os

class GPTModel:
    def __init__(self):
        self.tokenizer = MyTokenizer()
        vocab_size = 256  # 词表大小，需与tokenizer一致
        embed_dim = 32    # 嵌入维度，可自定义
        self.embedding = np.random.randn(vocab_size, embed_dim)  # [V, D]
        self.proj = np.random.randn(embed_dim, vocab_size)  # [D, V]
        # embedding 与 proj 解耦，从而观察学习特点
        self.optimizer = SGD([self.embedding, self.proj], lr=0.01)

        # 缩放，避免softmax过尖
        self.embedding *= 0.02
        self.proj *= 0.02

    def forward(self, x):
        embeddings = []
        logits = []

        B = len(x)
        T = len(x[0])
        
        for i in range(B):
            seq_ids = x[i]  # [T]
            seq_embeddings = [self.embedding[token_id] for token_id in seq_ids] # [T, D]
            seq_embeddings = self.self_attention(seq_embeddings)
            # 保存embeddings, 用于backward
            embeddings.append(seq_embeddings)  # [T, D]
            seq_logits = []
            for t in range(T):
                logit_t = np.dot(seq_embeddings[t], self.proj) # [V]
                seq_logits.append(logit_t)
            logits.append(seq_logits)  # [T, V]
        self.cache = {
            'embeds': embeddings,
            'logits': logits,
            'x': x
        }
        return logits  # [B, T, V]
    

    def self_attention(self, seq_embeds):
        # 直接实现 p(x_{t+1} | x_0, x_1, …, x_t) 的简单版本
        # 使用累积平均来聚合上下文：每个位置的输出是前缀的平均嵌入
        # 输入: seq_embeds [T, D]
        # 输出: [T, D] (每个位置基于前缀平均)  
        T = len(seq_embeds)
        D = len(seq_embeds[0]) 
        output_embeds = []
        for t in range(T):
            if t == 0:
                output_embeds.append(seq_embeds[0])  # 第一个位置不变
            else:
                prefix_sum = np.zeros(D)
                for k in range(t + 1):
                    prefix_sum += seq_embeds[k]
                prefix_avg = prefix_sum / (t + 1)
                output_embeds.append(prefix_avg)
        return output_embeds  # [T, D]
    

    def softmax(self, logits):
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
    

    def cross_entropy(self, logits, targets):
        # 简单的交叉熵损失实现，返回一个标量loss
        loss = 0.0
        for i in range(len(targets)):
            for j in range(len(targets[0])):
                probs = self.softmax(logits[i][j])  # [V]
                target_id = targets[i][j]
                loss -= np.log(probs[target_id] + 1e-9)  
        loss = loss / (len(targets) * len(targets[0]))  
        # 保存用于 backward
        self.cache['targets'] = targets
        self.cache['logits'] = logits
        return loss  
     

    def backward(self, loss):
        # 实现反向传播
        if 'logits' not in self.cache or 'targets' not in self.cache:
            return
        
        embeds = self.cache['embeds']  # [B, T, D] after attention
        logits = self.cache['logits']  # [B, T, V]
        targets = self.cache['targets']  # [B, T]
        x = self.cache['x']  # [B, T]
        
        B, T, V = len(logits), len(logits[0]), len(logits[0][0])
        D = len(embeds[0][0])
        
        # 初始化梯度
        self.optimizer.zero_grad()
        
        # 计算 d_loss/d_logits
        d_logits = np.zeros((B, T, V))
        for i in range(B):
            for j in range(T):
                probs = self.softmax(logits[i][j])
                one_hot = np.zeros(V)
                one_hot[targets[i][j]] = 1
                d_logits[i][j] = (probs - one_hot) / (B * T)
               
        
        # d_logits/d_proj = embeds.T, 所以 proj.grad += embeds.T @ d_logits
        for i in range(B):
            for j in range(T):
                outer = np.outer(embeds[i][j], d_logits[i][j])
                self.optimizer.grads[1] += outer
        
        # d_logits/d_embeds = proj.T, 所以 d_embeds = d_logits @ proj.T
        d_embeds = np.zeros((B, T, D))
        for i in range(B):
            for j in range(T):
                d_embeds[i][j] = np.dot(d_logits[i][j], self.proj.T)
        
        # 现在反向通过 self_attention
        # self_attention 是累积平均，需要反向
        for i in range(B):
            seq_d_embeds = d_embeds[i]  # [T, D]
            # 反向累积平均
            d_seq_embeds = np.zeros((T, D))
            for t in range(T-1, -1, -1):
                if t == 0:
                    d_seq_embeds[0] += seq_d_embeds[0]
                else:
                    # 每个前缀贡献 1/(t+1)
                    for k in range(t+1):
                        d_seq_embeds[k] += seq_d_embeds[t] / (t+1)
        
            # 现在 d_seq_embeds 到 embedding
            for j in range(T):
                token_id = x[i][j]
                self.optimizer.grads[0][token_id] += d_seq_embeds[j]


    def get_batch(self, tokens, batch_size, block_size):
        x, y = [], []
        n = len(tokens)
        for _ in range(batch_size):
            start = np.random.randint(0, n - block_size - 1)
            x.append(tokens[start:start+block_size])
            y.append(tokens[start+1:start+block_size+1])
        return x, y
    

    def save(self, path, step=0, extra_meta=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {
            "step": int(step),
            "vocab_size": int(self.embedding.shape[0]),
            "embed_dim": int(self.embedding.shape[1]),
            "lr": float(self.optimizer.lr),
            "weight_decay": float(getattr(self.optimizer, "weight_decay", 0.0)),
        }
        if extra_meta:
            meta.update(extra_meta)

        # np.savez 会把多个数组打包到一个文件里
        np.savez(
            path,
            embedding=self.embedding,
            proj=self.proj,
            meta=np.array([meta], dtype=object)  # 用 object 存 dict
        )


    def load(self, path, load_optimizer_lr=True):
        ckpt = np.load(path, allow_pickle=True)
        self.embedding[...] = ckpt["embedding"]
        self.proj[...] = ckpt["proj"]

        meta = ckpt["meta"].item() if "meta" in ckpt else {}
        if load_optimizer_lr and "lr" in meta:
            self.optimizer.lr = float(meta["lr"])
        return meta



class MyTokenizer:
    def __init__(self):
        # TODO: 构建 vocab（词到id）和 itos（id到词）
        # 这里用最简单的字符级映射
        chars = [chr(i) for i in range(256)]  # 假设只用ASCII
        self.vocab = {ch: i for i, ch in enumerate(chars)}  # str -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int -> str


    def encode(self, text):
        # TODO: 文本转token id列表
        # return [self.vocab[c] for c in text]
        return [self.vocab.get(c, 0) for c in text]  # 未知字符映射为0


    def decode(self, ids):
        # TODO: token id列表转文本
        # return ''.join([self.itos[i] for i in ids])
        return ''.join([self.itos[i] for i in ids])
    


class SGD:
    def __init__(self, params, lr=0.01, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.grads = [np.zeros_like(p) for p in params]


    def zero_grad(self):
        for g in self.grads:
            g.fill(0.0)


    def step(self):
        for param, grad in zip(self.params, self.grads):
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param  # L2 正则（可选）
            param -= self.lr * grad