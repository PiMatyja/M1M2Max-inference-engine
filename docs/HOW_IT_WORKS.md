# How does Strata work on a Mac?

Models like Qwen3.8-Flash-Next normally run on servers with hundreds of gigabytes of memory. An M1 Max or M2 Max
with 64 GB runs it because the model's main file is 47 GB, the Mac's memory is shared between the processor and
the GPU, and the model only needs a small part of itself for each word. Back to the [README](../README.md).

## Who does what

- **The model is a team of 24,576 small specialists ("experts"),** 512 in each of its 48 layers, and each word it
  writes needs only 10 of them per layer. So most of the model sits still while a word is written.
- **The GPU** does all the work. On a Mac the GPU and the processor share one memory, so there is nothing to
  copy: the engine maps the 47 GB model file once and the GPU reads every weight where it lies.
- **The SSD** holds the model's second file, a 28.8 GB lookup table (the per-layer token embeddings). The model
  reads a few small rows of it per word.

The same in technical terms:

- **Unified memory:** the model file is mapped and wrapped as Metal buffers without a copy (several overlapping
  buffers, because one buffer may be at most 38.88 GiB on these chips). The attention and DeltaNet mixers, the
  hyper-connections, the routers, the shared experts, the output head, the MTP draft layer, the KV cache and the
  routed experts are all read in place. `STRATA_METAL_STREAM=0` keeps it that way (resident); the engine can also
  stream the experts from the SSD on a Mac whose memory cannot hold the file, but that is not this setup.
- **The GPU memory limit:** macOS caps how much memory the GPU may wire. `sudo sysctl iogpu.wired_limit_mb=58000`
  raises it so the whole model plus the KV cache and the prompt buffers fit. It resets on reboot.
- **The SSD:** the 28.8 GB table (`--ple-gguf`), read a few rows per token.

## Guess, then check

- **Guess, then check.** A small, fast helper built into the model guesses the next few words, and the big model
  checks all the guesses in one go. It keeps the ones it agrees with and writes the next word itself, so one step
  often produces several words. The helper only guesses; the big model decides every word, so you get the same
  answer, sooner.
- In technical terms: the model's own MTP layer drafts 2 tokens per round (`--mtp-max-t 2`; 3 or more was slower
  on these chips); one pass over all 48 layers checks them. When the reply repeats the context (code edits, quoted
  text), **prompt lookup** drafts up to 8 tokens from the earlier copy (`--spec 8`). The drafts are checked like
  the MTP's, so the output is the same.

## Reading long texts

**Long texts are read in big pieces** (chunks of thousands of tokens at a time) through the prompt kernels, not
one token at a time, which is why a long document or code base is read at 280-307 tokens per second on an M2 Max,
flat up to 78K tokens. After the first message, Strata keeps the conversation and reads only what is new, so
follow-ups start in seconds.

Three caches make that hold up under an agent:

- **Checkpoints** of the conversation's state, taken at each new assistant turn and every 16K prompt tokens, so
  a request that continues a chat reads only its new part.
- **The root slot:** the shared system prompt and tool list an agent sends with every request (about 14K tokens)
  is read once and kept in a pinned slot that no other request evicts.
- **The SSD tier:** a conversation displaced by another chat or a side request is parked on the SSD and restored
  when it comes back, also across restarts. Going back to an older chat takes about 5 s instead of 60-75 s.

## More

The API and every server setting: [DETAILS.md](DETAILS.md). What was tuned for these chips and why:
[TUNING.md](TUNING.md). The model files: [MODEL.md](MODEL.md).

## Credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; the compressed
  version by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF). Their licenses
  apply to the model files.
- Strata by [Niko1221](https://github.com/Niko1221/Strata) (MIT). Built with parts of
  [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Ideas from
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen).

## License

Strata is open source under the [MIT License](../LICENSE). A few parts carry their own licenses: `third_party/ggml`
(MIT, llama.cpp / ggml) and the web app's font (SIL Open Font License 1.1). The models are not part of this
repository; each model's own license applies to its files.
