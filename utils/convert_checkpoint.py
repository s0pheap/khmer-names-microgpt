"""
Converts an existing checkpoint_*.json to a compressed .npz file.

Usage:
  python utils/convert_checkpoint.py                        # converts checkpoint_khmer_names.json
  python utils/convert_checkpoint.py my_checkpoint.json    # converts a specific file
"""
import json
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def convert(json_path: Path):
    npz_path = json_path.with_suffix(".npz")

    print(f"Reading  {json_path} ...")
    with open(json_path, encoding="utf-8") as f:
        saved = json.load(f)

    arrays = {}

    # weights — each is a 2D list → numpy array
    for key, mat in saved["weights"].items():
        arrays[key] = np.array(mat, dtype=np.float32)

    # metadata
    arrays["_uchars"]     = np.array(saved["uchars"])
    arrays["_BOS"]        = np.array(saved["BOS"])
    arrays["_vocab_size"] = np.array(saved["vocab_size"])
    arrays["_n_embd"]     = np.array(saved["n_embd"])
    arrays["_n_layer"]    = np.array(saved["n_layer"])
    arrays["_n_head"]     = np.array(saved["n_head"])
    arrays["_block_size"] = np.array(saved["block_size"])

    np.savez_compressed(npz_path, **arrays)

    json_size = json_path.stat().st_size / 1024
    npz_size  = npz_path.stat().st_size  / 1024
    print(f"Saved    {npz_path}")
    print(f"Size     {json_size:.1f} KB (json) → {npz_size:.1f} KB (npz)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_absolute():
            path = DATA_DIR / path
    else:
        path = DATA_DIR / "checkpoint_khmer_names.json"

    if not path.exists():
        print(f"[error] File not found: {path}")
        sys.exit(1)

    convert(path)
