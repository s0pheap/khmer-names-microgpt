"""
Trains the MicroGPT model on khmer_names_training.txt.
Implements the same architecture as microgpt.py (Karpathy) adapted for
named datasets and checkpoint saving.

Dataset trained:
  data/khmer_names_training.txt → data/checkpoint_khmer_names.json

Usage:
  python train.py
"""
import json
import math
import multiprocessing as mp
import os
import random
import sys

# ── Hyperparameters (mirrors microgpt.py) ────────────────────────────────────
N_LAYER    = 1
N_EMBD     = 16
BLOCK_SIZE = 16
N_HEAD     = 4
HEAD_DIM   = N_EMBD // N_HEAD

NUM_STEPS  = 3000

LR         = 0.01
BETA1      = 0.85
BETA2      = 0.99
EPS_ADAM   = 1e-8
TEMPERATURE = 0.5
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASETS = {
    "khmer_names": "khmer_names_training.txt",
}

# ── Autograd (same Value class as microgpt.py) ────────────────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):  return Value(math.log(self.data),  (self,), (1/self.data,))
    def exp(self):  return Value(math.exp(self.data),  (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data),    (self,), (float(self.data > 0),))
    def __neg__(self):            return self * -1
    def __radd__(self, other):    return self + other
    def __sub__(self, other):     return self + (-other)
    def __rsub__(self, other):    return other + (-self)
    def __rmul__(self, other):    return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other):return other * self**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# ── Model helpers ─────────────────────────────────────────────────────────────
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    mv = max(val.data if isinstance(val, Value) else val for val in logits)
    exps = [(v - mv).exp() if isinstance(v, Value) else Value(math.exp(v - mv)) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_plain(logits):
    mv = max(logits); exps = [math.exp(v - mv) for v in logits]; s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    return [xi * (ms + 1e-5) ** -0.5 for xi in x]

def make_matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def build_state_dict(vocab_size):
    sd = {
        'wte': make_matrix(vocab_size, N_EMBD),
        'wpe': make_matrix(BLOCK_SIZE, N_EMBD),
        'lm_head': make_matrix(vocab_size, N_EMBD),
    }
    for i in range(N_LAYER):
        sd[f'layer{i}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)
    return sd

def gpt(token_id, pos_id, keys, values, sd):
    x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(N_LAYER):
        x_res = x; x = rmsnorm(x)
        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs:hs+HEAD_DIM]
            k_h = [ki[hs:hs+HEAD_DIM] for ki in keys[li]]
            v_h = [vi[hs:hs+HEAD_DIM] for vi in values[li]]
            al = [sum(q_h[j]*k_h[t][j] for j in range(HEAD_DIM))/HEAD_DIM**0.5 for t in range(len(k_h))]
            aw = softmax(al)
            x_attn.extend([sum(aw[t]*v_h[t][j] for t in range(len(v_h))) for j in range(HEAD_DIM)])
        x = linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = [a+b for a, b in zip(x, x_res)]
        x_res = x; x = rmsnorm(x)
        x = linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a+b for a, b in zip(x, x_res)]
    return linear(x, sd['lm_head'])

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(name: str, data_file: str):
    data_path   = os.path.join(DATA_DIR, data_file)
    output_path = os.path.join(DATA_DIR, f"checkpoint_{name}.json")

    if not os.path.exists(data_path):
        print(f"[{name}] SKIP — file not found: {data_path}")
        return

    random.seed(42)
    docs = [line.strip() for line in open(data_path, encoding="utf-8") if line.strip()]
    random.shuffle(docs)

    uchars     = sorted(set("".join(docs)))
    BOS        = len(uchars)
    vocab_size = len(uchars) + 1

    print(f"[{name}] pid={os.getpid()} | docs={len(docs)} | vocab={vocab_size} | steps={NUM_STEPS}")

    sd     = build_state_dict(vocab_size)
    params = [p for mat in sd.values() for row in mat for p in row]
    m_buf  = [0.0] * len(params)
    v_buf  = [0.0] * len(params)

    for step in range(NUM_STEPS):
        doc    = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n      = min(BLOCK_SIZE, len(tokens) - 1)

        keys   = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, sd)
            probs  = softmax(logits)
            losses.append(-probs[target_id].log())

        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = LR * (1 - step / NUM_STEPS)
        for i, p in enumerate(params):
            m_buf[i] = BETA1 * m_buf[i] + (1 - BETA1) * p.grad
            v_buf[i] = BETA2 * v_buf[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_buf[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_buf[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0

        print(f"[{name}] step {step+1:4d}/{NUM_STEPS} | loss {loss.data:.4f}", end="\r", flush=True)

    # Save checkpoint — JSON
    weights = {k: [[p.data for p in row] for row in mat] for k, mat in sd.items()}
    checkpoint = {
        "weights": weights, "uchars": uchars, "vocab_size": vocab_size,
        "n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD,
        "block_size": BLOCK_SIZE, "BOS": BOS, "dataset": name,
    }
    with open(output_path, "w") as f:
        json.dump(checkpoint, f)
    print(f"\n[{name}] saved → {output_path}")

    # Sanity check — 5 samples
    sd_plain = {k: [[p.data for p in row] for row in mat] for k, mat in sd.items()}
    print(f"[{name}] samples:")
    for _ in range(5):
        keys2 = [[] for _ in range(N_LAYER)]
        vals2 = [[] for _ in range(N_LAYER)]
        tok, out = BOS, []
        for pos_id in range(BLOCK_SIZE):
            x = [t + p for t, p in zip(sd_plain['wte'][tok], sd_plain['wpe'][pos_id])]
            x = [xi * (sum(v*v for v in x)/len(x) + 1e-5)**-0.5 for xi in x]
            for li in range(N_LAYER):
                x_res = x[:]
                ms = sum(v*v for v in x)/len(x); x = [xi*(ms+1e-5)**-0.5 for xi in x]
                q  = [sum(w*xi for w, xi in zip(row, x)) for row in sd_plain[f'layer{li}.attn_wq']]
                k  = [sum(w*xi for w, xi in zip(row, x)) for row in sd_plain[f'layer{li}.attn_wk']]
                vv = [sum(w*xi for w, xi in zip(row, x)) for row in sd_plain[f'layer{li}.attn_wv']]
                keys2[li].append(k); vals2[li].append(vv)
                x_attn = []
                for h in range(N_HEAD):
                    hs = h * HEAD_DIM
                    q_h = q[hs:hs+HEAD_DIM]
                    k_h = [ki[hs:hs+HEAD_DIM] for ki in keys2[li]]
                    v_h = [vi[hs:hs+HEAD_DIM] for vi in vals2[li]]
                    al  = [sum(q_h[j]*k_h[t][j] for j in range(HEAD_DIM))/HEAD_DIM**0.5 for t in range(len(k_h))]
                    aw  = softmax_plain(al)
                    x_attn.extend([sum(aw[t]*v_h[t][j] for t in range(len(v_h))) for j in range(HEAD_DIM)])
                x = [sum(w*xi for w, xi in zip(row, x_attn)) for row in sd_plain[f'layer{li}.attn_wo']]
                x = [a+b for a, b in zip(x, x_res)]
                x_res = x[:]
                ms = sum(v*v for v in x)/len(x); x = [xi*(ms+1e-5)**-0.5 for xi in x]
                x = [sum(w*xi for w, xi in zip(row, x)) for row in sd_plain[f'layer{li}.mlp_fc1']]
                x = [max(0, xi) for xi in x]
                x = [sum(w*xi for w, xi in zip(row, x)) for row in sd_plain[f'layer{li}.mlp_fc2']]
                x = [a+b for a, b in zip(x, x_res)]
            logits = [sum(w*xi for w, xi in zip(row, x)) for row in sd_plain['lm_head']]
            probs  = softmax_plain([l / TEMPERATURE for l in logits])
            tok    = random.choices(range(len(probs)), weights=probs)[0]
            if tok == BOS: break
            out.append(uchars[tok])
        print(f"  {name}: {''.join(out)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key not in DATASETS:
            print(f"Unknown dataset '{key}'. Choose from: {list(DATASETS)}")
            sys.exit(1)
        train_model(key, DATASETS[key])
    else:
        n_jobs = min(len(DATASETS), mp.cpu_count())
        print(f"Training {len(DATASETS)} models using {n_jobs} parallel processes…\n")
        with mp.Pool(processes=n_jobs) as pool:
            pool.starmap(train_model, [(n, f) for n, f in DATASETS.items()])
        print("\nAll models trained.")
