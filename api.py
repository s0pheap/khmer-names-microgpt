"""
FastAPI server for MicroGPT inference.
Run: uvicorn api:app --reload
"""
import math
import os
import random

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")

CHECKPOINTS = {
    "khmer_names": "checkpoint_khmer_names.npz",
}

# ── Inference helpers (plain floats, no autograd needed) ─────────────────────
def _linear(x, w):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def _softmax(logits):
    mv = max(logits); exps = [math.exp(v - mv) for v in logits]; s = sum(exps)
    return [e / s for e in exps]

def _rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    return [xi * (ms + 1e-5) ** -0.5 for xi in x]

def _forward(token_id, pos_id, keys, values, m):
    sd, n_layer, n_head, head_dim = m["sd"], m["n_layer"], m["n_head"], m["head_dim"]
    x = [t + p for t, p in zip(sd["wte"][token_id], sd["wpe"][pos_id])]
    x = _rmsnorm(x)
    for li in range(n_layer):
        x_res = x; x = _rmsnorm(x)
        q = _linear(x, sd[f"layer{li}.attn_wq"])
        k = _linear(x, sd[f"layer{li}.attn_wk"])
        v = _linear(x, sd[f"layer{li}.attn_wv"])
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs  = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            al  = [sum(q_h[j]*k_h[t][j] for j in range(head_dim))/head_dim**0.5 for t in range(len(k_h))]
            aw  = _softmax(al)
            x_attn.extend([sum(aw[t]*v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)])
        x = _linear(x_attn, sd[f"layer{li}.attn_wo"])
        x = [a+b for a, b in zip(x, x_res)]
        x_res = x; x = _rmsnorm(x)
        x = _linear(x, sd[f"layer{li}.mlp_fc1"])
        x = [max(0, xi) for xi in x]
        x = _linear(x, sd[f"layer{li}.mlp_fc2"])
        x = [a+b for a, b in zip(x, x_res)]
    return _linear(x, sd["lm_head"])

# ── Load checkpoint ───────────────────────────────────────────────────────────
def _load(path: str) -> dict:
    data       = np.load(path, allow_pickle=True)
    uchars     = data["_uchars"].tolist()
    BOS        = int(data["_BOS"])
    n_layer    = int(data["_n_layer"])
    n_head     = int(data["_n_head"])
    n_embd     = int(data["_n_embd"])
    block_size = int(data["_block_size"])
    sd         = {k: data[k].tolist() for k in data.files if not k.startswith("_")}
    return {
        "uchars":     uchars,
        "BOS":        BOS,
        "n_layer":    n_layer,
        "n_head":     n_head,
        "head_dim":   n_embd // n_head,
        "block_size": block_size,
        "sd":         sd,
    }

MODELS: dict[str, dict] = {}
for name, fname in CHECKPOINTS.items():
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        MODELS[name] = _load(path)
        print(f"Loaded: {name}")
    else:
        print(f"[warn] not found, skipping: {fname}")

if not MODELS:
    raise RuntimeError("No checkpoints found in data/. Run train.py first.")

# ── Generation ────────────────────────────────────────────────────────────────
def _generate(model_name: str, prefix: str, temperature: float, count: int) -> list[str]:
    m      = MODELS[model_name]
    uchars = m["uchars"]
    BOS    = m["BOS"]
    tokens = [BOS] + [uchars.index(ch) for ch in prefix if ch in uchars]
    results, attempts = [], 0

    while len(results) < count and attempts < count * 20:
        attempts += 1
        keys   = [[] for _ in range(m["n_layer"])]
        values = [[] for _ in range(m["n_layer"])]

        for pos_id, token_id in enumerate(tokens):
            if pos_id >= m["block_size"]: break
            _forward(token_id, pos_id, keys, values, m)

        name     = list(prefix)
        pos_id   = len(tokens)
        token_id = tokens[-1]

        for _ in range(m["block_size"] - len(tokens)):
            logits   = _forward(token_id, pos_id, keys, values, m)
            probs    = _softmax([l / temperature for l in logits])
            token_id = random.choices(range(len(probs)), weights=probs)[0]
            if token_id == BOS: break
            name.append(uchars[token_id])
            pos_id += 1

        name_str = "".join(name)
        if name_str not in results and len(name_str) > len(prefix):
            results.append(name_str)

    return results

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MicroGPT Name Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prefix:      str   = Field(default="")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    count:       int   = Field(default=10,  ge=1,   le=50)
    model:       str   = Field(default="latin_full")

class GenerateResponse(BaseModel):
    results: list[str]
    prefix:  str
    model:   str

@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys())}

@app.get("/models")
def list_models():
    return {"models": list(MODELS.keys())}

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if req.model not in MODELS:
        raise HTTPException(404, f"Model '{req.model}' not found. Available: {list(MODELS.keys())}")
    try:
        results = _generate(req.model, req.prefix.lower(), req.temperature, req.count)
        return GenerateResponse(results=results, prefix=req.prefix, model=req.model)
    except Exception as e:
        raise HTTPException(500, str(e))

if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))
