# The model this repository serves

One model, one size: **Qwen3.8-Flash-Next** by the Qwen team, in ISTA-DASLab's GSQ-RCO quantization at
**IQ3_XXS**. Nothing else is set up here. The model files are not part of this repository; their own licenses
apply to them.

## The files

Hugging Face repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, folder `IQ3_XXS`:

| Shard | File | Bytes | SHA-256 |
|---|---|---:|---|
| 1 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47,039,860,096 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| 2 | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

The hashes are the Hugging Face LFS oids of that revision
(`https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_XXS`).
`scripts/download-model.sh` checks both after the download; check them yourself with `shasum -a 256` when you
download by hand. The files go in `$STRATA_MODELS/IQ3_XXS/` (default `~/strata-models/IQ3_XXS/`).

- **Shard 1 (47.0 GB)** is the model: the experts, the attention and SSM layers, the routers, the hyper-connections,
  the embeddings and the output head. The GPU maps it and reads every weight where it lies in unified memory
  (`--native <shard 1>`), as overlapping Metal buffers because one buffer cannot cover it
  ([TUNING.md](TUNING.md)).
- **Shard 2 (28.8 GB)** is the per-layer token embedding table (`--ple-gguf <shard 2>`). The engine reads a few
  rows of it per token from the SSD; it does not have to sit in memory.

## What is inside

48 layers, hidden size 2560, 512 routed experts per layer with 10 used per token, MoE intermediate 640, 36 GDN
and 12 QSA attention layers, 4-stream hyper-connections. The tensor formats in this file:

- routed experts: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and Q2_0 (GGUF type 42);
- attention and `ssm_out`: IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K;
- shared expert: IQ3_S, IQ4_XS, Q4_K, IQ4_NL;
- output head: Q5_K;
- routers, hyper-connections, indexer and SSM projections: BF16. The hyper-connection matrices are read as Q8_0
  copies by default on the Mac (`STRATA_HC_Q8=0` gives the exact BF16 read).

## The MTP draft layer

The GGUF ships no draft head. The model's own multi-token-prediction (MTP) layer lets the engine guess the next
tokens and check them in one pass, which is where much of the decode speed comes from. Its 31 `mtp.*` tensors are
read from the original BF16 checkpoint (`Qwen/Qwen3.8-Flash-Next`, pinned to one revision) by
`tools/mtp_fetch.py`: it reads the safetensors headers, then only the byte ranges of those tensors, and checks
every tensor against a pinned SHA-256. Nothing else of the 360 GB checkpoint is downloaded.

```bash
.venv/bin/python tools/mtp_fetch.py fetch --out ~/strata-models/mtp
.venv/bin/python tools/mtp_fetch.py verify --out ~/strata-models/mtp
```

`tools/mtp_pack.py` then writes the draft layer as one GGUF, `mtp-q2_0.gguf`: the dense tensors stay BF16 (F32 for
the norms), the routed experts are quantized to Q2_0 (about 0.71 GB), the same format as the main model's experts.

```bash
.venv/bin/python tools/mtp_pack.py --src ~/strata-models/mtp --experts q2_0 --out ~/strata-models/mtp/mtp-q2_0.gguf
mkdir -p ~/strata-models/mtp/rt && cp data/draft_vocab.bin ~/strata-models/mtp/rt/
```

The engine takes `--mtp ~/strata-models/mtp/rt`: it reads the draft head's token subset from
`rt/draft_vocab.bin` and the draft layer itself from `mtp-q2_0.gguf` in the folder above `rt`
(`STRATA_METAL_MTP_GGUF=<path>` names another file). With `--mtp-max-t 2` it drafts 2 tokens per round.

The draft only proposes; the full model checks every token. The output with MTP is the same as without it.

## The pack

`tools/iq_pack.py` writes the expert pack the engine loads, from shard 1:

```bash
STRATA_GGUF_PY=third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
  --gguf ~/strata-models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --out ~/strata-models/packs/iq3_xxs
```

The i-quant experts stay in their GGUF form; the pack is an index over them plus the small float tensors in the
form the engine reads them:

- `index.txt`: the table the engine loads. Quantized dense tensors, the token embedding and the output head are
  served from the GGUF itself (`--native`); their rows carry shape only.
- `native_experts.txt`: one line per layer with the experts' types and offsets in the GGUF. Written last, so a
  pack without it is not finished.
- `dense.bin`: the BF16 / F16 / F32 tensors as the engine reads them.
- `conversions.json`: every tensor that was converted, with the method and the source bytes' SHA-256.
- `tokenizer/`: exported from the GGUF, with the model's chat template.

**The pack is tied to the exact GGUF.** It records the shards' names, sizes and offsets, and the engine reads the
experts from the GGUF at those offsets. If the model file changes (another revision, another download), rebuild
the pack.

The pack tools and llama.cpp's `gguf-py` import numpy; `scripts/python-env.sh` installs it from a pinned source
archive.

## Memory

With the model resident the GPU keeps the 47 GB of shard 1 in memory, plus the prompt buffers, the KV cache
(`--kv int8`, 131072 context) and the draft layer. That is why the GPU limit is raised to 58,000 MB
(`sudo sysctl iogpu.wired_limit_mb=58000`) and why one engine runs at a time.

## Credits

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
- Quantization: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
- The GGUF formats and `gguf-py`: [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT).
