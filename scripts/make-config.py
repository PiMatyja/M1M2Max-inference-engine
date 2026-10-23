#!/usr/bin/env python3
"""make-config.py [--context N] [--port N] [--out FILE]: writes the server's run config (guide section 6)."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--context", type=int, default=131072, help="the longest chat in tokens (default 131072)")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--out", default=os.path.join(ROOT, "strata-cached.json"))
ap.add_argument("--models", default=os.environ.get("STRATA_MODELS") or os.path.expanduser("~/strata-models"))
a = ap.parse_args()

m = os.path.abspath(os.path.expanduser(a.models))
pack = os.path.join(m, "packs", "iq3_xxs")
shard1 = os.path.join(m, "IQ3_XXS", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
shard2 = os.path.join(m, "IQ3_XXS", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf")
args = ["--pack", pack, "--native", shard1, "--ple-gguf", shard2,
        "--expert-profile", os.path.join(ROOT, "data", "expert-profile.bin"),
        "--expert-cache", "64", "--prefill", "auto", "--spec", "8", "--mtp", os.path.join(m, "mtp", "rt"),
        "--mtp-max-t", "2", "--max-context", str(a.context), "--kv", "int8", "--mmap-experts",
        "--root-slot-tokens", "16384", "--ssd-cache-mib", "20480", "--ssd-cache-dir", os.path.expanduser("~/.strata-cache"),
        "--conversation-cache-mib", "0", "--conversation-cache-min-tokens", "4096", "--conversation-cache-min-free-mib", "1024"]
cfg = {"exe": os.path.join(ROOT, "build-metal", "strata"), "args": args, "cwd": ROOT,
       "tokenizer": os.path.join(pack, "tokenizer"), "model_name": "qwen3.8-flash-next-iq3_xxs",
       "log": os.path.join(ROOT, "strata-cached.log"), "lib_dirs": [], "port": a.port, "backend": "metal",
       "env": {"STRATA_METAL_STREAM": "0"},
       "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
       "reasoning_effort": "medium", "reasoning_budget_tokens": 8000, "effort_position": "end"}
with open(a.out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=1)
    f.write("\n")
print(f"wrote {a.out}: context {a.context}, port {a.port}, models in {m}")
missing = [p for p in (cfg["exe"], shard1, shard2, pack, os.path.join(m, "mtp", "rt")) if not os.path.exists(p)]
for p in missing:
    print(f"note: {p} does not exist yet", file=sys.stderr)
