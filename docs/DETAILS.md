# Strata on the Mac: the API and the server settings

How to talk to the server and what its settings do. The setup itself is in [GUIDE.md](GUIDE.md), the model in
[MODEL.md](MODEL.md), the engine changes in [TUNING.md](TUNING.md), common problems in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

> **On this page:** [Endpoints](#endpoints) - [Thinking](#thinking-levels) - [Streaming and tools](#streaming-and-tool-calls) -
> [Chat apps and agents](#chat-apps-and-agents) - [Context and sampling](#context-and-sampling) -
> [The conversation cache](#the-conversation-cache) - [The Responses API and Codex](#the-responses-api-and-codex-cli) -
> [JSON response formats](#json-response-formats) - [Network](#network) - [Other settings](#other-settings)

The server is `serve/server.py`. It reads one run config (`strata-cached.json`, written by `scripts/make-config.py`),
starts the engine and serves the API. The config's `args` are the engine's command line; the keys outside `args`
are the server's own, and this page names them as `"key"`.

## Endpoints

The server listens on `http://127.0.0.1:8000` (the port is the server's `--port`; the host is always 127.0.0.1 in
this build).

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#the-responses-api-and-codex-cli)) | `POST /v1/responses` |
| Model list / health | `GET /v1/models`, `GET /models`, `GET /health` |
| Model properties | `GET /props` |
| What the model is doing right now | `GET /status`, `GET /slots` (busy or idle, with `n_prompt_tokens`) |
| Everything the Monitor tab shows (engine, live state, last requests, hardware) | `GET /metrics` |
| The same for Prometheus, with vLLM's metric names (`Accept: text/plain` or `?format=prometheus`) | `GET /metrics` |
| Request totals in Prometheus' text format, under llama-server's metric names | `GET /metrics/prometheus` |

`/models` and `/v1/models` list only the loaded model, with its context limit. `/props` exposes the chat template,
the context limit, the configured generation defaults, the model path and the engine version. `/slots` uses
llama.cpp's names: `n_ctx` and `n_prompt_tokens` (the running request's prompt size, kept after it ends: what a
front-end's context meter divides by `n_ctx`). `chat_template_caps` in `/props` uses llama.cpp's field names
(`supports_tools`, `supports_tool_calls`, `supports_parallel_tool_calls`, `supports_system_role`,
`supports_preserve_reasoning`); they are compatibility hints for clients, not a guarantee that the model follows a
request. The chat page is at `http://127.0.0.1:8000/`, with a Monitor tab.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.8-flash-next-iq3_xxs", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
r = client.chat.completions.create(model="qwen3.8-flash-next-iq3_xxs", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

Any model name is served; the loaded model answers. `"aliases": ["qwen", "local-model"]` in the config lists the
model under those names too in `/v1/models`.

## Thinking levels

The model thinks before it answers (streamed as `reasoning_content`; Anthropic: `thinking` blocks). Choose how much
per request, in the chat page (the "Thinking" menu) or over the API:

| API | how |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

Without a setting the model uses its own default, the template's highest level (xhigh), which on this setup
overthinks and often never answers. So the tuned config sets three keys:

- **`"reasoning_effort": "medium"`**: the level for every request that names none. A request's own level wins.
- **`"reasoning_budget_tokens": 8000`**: a hard cap on the thinking. When a request gets there, the server ends the
  thinking with a short wrap-up line and `</think>`, and the model answers from there (the engine continues from
  what it already holds, so nothing is read again). The wrap-up is part of the thinking the client sees and counts
  as output tokens. A request's own `reasoning_budget_tokens` wins; `0` means no budget.
- **`"effort_position": "end"`**: the effort's instruction is normally the first thing in the prompt, so a request
  that only changes the effort (an agent's "think harder" switch, `none` for a quick tool step) would read the whole
  conversation again. With `"end"` every prompt starts as the default effort's and a `low` / `medium` effort goes in
  a short system turn right before the answer; the engine keeps its checkpoint in front of that turn, so the next
  request reuses the conversation whatever its effort.

`none` answers at once (fastest); `low` keeps the thinking short. The levels are instructions the model was
trained with, not a hard token limit: on easy questions all levels think briefly.

More thinking keys, all off by default:

- `"reasoning_close_retry": true`: a reply that ends inside its thinking with no answer and no tool call is closed
  once (as the budget does) and continued.
- `"repeat_stop_tokens": N`: a reply that repeats the same token N times in a row (default 256) is ended with
  `finish_reason` `"length"`; `0` turns it off.
- `"reasoning_loop_recovery": "stop" | "recover"`: a reply whose reasoning repeats whole passages is ended, or
  continued once with the low-effort instruction in place of the default one. A policy, not a guarantee of an
  answer; it can mistake repeated useful code for a loop.
- `"anthropic_thinking": "on_request"`: a `/v1/messages` request with no `"thinking"`, effort or budget is
  rendered without thinking (Anthropic's own rule, and what Claude Code's short helper calls need).

## Streaming and tool calls

- **Streaming.** With `"stream": true` everything arrives as it is made: the thinking, the answer, and tool calls
  (the tool's name first, then its arguments piece by piece). While the model reads a long prompt the stream sends
  keep-alives, so agents do not time out; `GET /status` says what it is doing (`reading the prompt`, `answering`,
  tokens so far). Closing the connection or pressing stop in your app really stops the model, so the next request
  starts at once.
- **Prefill progress in the stream** (`"return_progress": true` in the request): the progress goes on the stream
  as one extra field on a chunk with an empty delta, `prompt_progress` with `total`, `cache`, `processed` and
  `time_ms` (llama.cpp's four fields, with the same meaning), so a client that draws a prefill bar for llama.cpp
  draws one here too. The progress is this request's own: a ping while queued, or during another request's read,
  carries none.
- **Tool calls in other forms** (`"tool_call_recovery": true`): a call the model writes as `<parameter=NAME>`
  instead of `<function=NAME>`, as JSON inside `<tool_call>`, as a `<function=NAME>` block without `<tool_call>`, or
  as a second call inside the same `<tool_call>` is read as a call of a declared tool. Off by default, so what a
  client gets back is unchanged unless you turn it on.
- **Token biases**: `logit_bias` per request, see [LOGIT_BIAS.md](LOGIT_BIAS.md).

## Chat apps and agents

- **Any app with an "OpenAI-compatible" provider:** base URL `http://127.0.0.1:8000/v1`, any API key. Zed's
  llama.cpp provider reads `/props` and offers tools from `chat_template_caps`; its generic OpenAI-compatible
  provider does not: add `"capabilities": {"tools": true, "chat_completions": true}` on that model in Zed's settings.
- **OpenCode.** A starting point for `opencode.jsonc` (the field names are OpenCode's; check its docs if your
  version differs):

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "strata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Strata (local)",
        "options": { "baseURL": "http://127.0.0.1:8000/v1", "apiKey": "none" },
        "models": {
          "qwen3.8-flash-next-iq3_xxs": {
            "name": "Qwen3.8-Flash-Next (Strata)",
            "limit": { "context": 131072, "output": 32768 },
            "options": { "reasoningEffort": "medium" },
            "variants": {
              "low": { "reasoningEffort": "low" },
              "high": { "reasoningEffort": "high" },
              "none": { "reasoningEffort": "none" }
            }
          }
        }
      }
    },
    "model": "strata/qwen3.8-flash-next-iq3_xxs"
  }
  ```

  Keep `limit.output` well under the context: a request whose prompt plus `max_tokens` runs past the context is
  refused, or add `"fit_max_tokens": true` to the config (below).
  `"env_to_user": true` in the config moves opencode's `<env>` block (working directory, date) from the system
  message to the front of the first user message, so the system turn is the same for every session and the
  cache restores it.
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`, `ANTHROPIC_MODEL` set to a Claude model name it
  knows (it refuses names it does not; the name is ignored here), plus any `ANTHROPIC_AUTH_TOKEN`. Claude Code sends
  `"output_config": {"effort": "high"}` with every request unless told otherwise; `CLAUDE_CODE_EFFORT_LEVEL=medium`
  (or `low`) in its environment changes that. Claude Code stamps its system prompt with a billing header that
  changes on every request; the server pins those stamps, so the system prompt and the tool list are the same
  prompt on every turn and the cache holds.
- **Codex CLI:** see [the Responses API](#the-responses-api-and-codex-cli) below.

## Context and sampling

- **Context.** 131072 tokens in the tuned config (`--max-context`; `scripts/make-config.py --context N` changes
  it). Requests longer than that are refused, never silently cut. A request whose `max_tokens` would run past the
  context is refused too (400); agents that always ask for their full output cap can instead get it shortened to
  the room left: `"fit_max_tokens": true` in the config. A prompt that leaves no room at all is still refused.
- **Temperature / top_p / top_k / min_p / seed** are honored per request (OpenAI and Anthropic fields), and so
  are stop strings (OpenAI `stop`, a string or up to 4; Anthropic `stop_sequences`). The config's optional
  `sampling` block sets the defaults for requests that leave the fields out
  (`"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`); a request's own fields always win, and with
  no block at all a request without sampling keys decodes greedy. `top_k` keeps at most 64 candidates.
- **The penalties** (`presence_penalty`, `frequency_penalty`, `repetition_penalty`, with `penalty_last_n` capping how
  many recent tokens they count over, default 64 when any penalty is set; llama.cpp's `repeat_penalty` and
  `repeat_last_n` work too) count the tokens the request has consumed, and apply to every token the speculative
  decoding checks, exactly as if it decoded one token at a time. Requests with penalties are a little slower: the
  draft layer guesses without them, so more guesses are rejected.
- **Greedy output is reproducible** for a fixed drafting setup. Drafting history changes near-tie answers, so a
  comparison fixes it ([TUNING.md](TUNING.md)).

## The conversation cache

A request that continues a chat reads only the part after what the engine already holds: the live session, or
one of the checkpoints it keeps (taken at the start of each new assistant turn and every 16K prompt tokens). A
checkpoint is used only when the prompt starts with exactly its tokens. The oldest checkpoint, in practice the
end of the system prompt that every chat of the same client shares, is kept while the rest rotates by least
recent use, so a new chat that shares that prefix starts reading after it instead of from token 0. Engine
options: `--prompt-cache N` (0 = off), `--prompt-cache-every N`, `--prompt-cache-root N` (0 = no system-prompt
checkpoint), `--turn-token ID`, and [`--prompt-cache-tail`](PROMPT_CACHE_TAIL.md).

On top of that, the tuned config turns on two Metal tiers ([TUNING.md](TUNING.md)):

- **The root slot** (`--root-slot-tokens 16384`): the shared system prompt and tool list is read once into a
  pinned slot that no other request evicts.
- **The SSD tier** (`--ssd-cache-mib 20480 --ssd-cache-dir ~/.strata-cache`): a conversation displaced by another
  chat or a side request is parked on the SSD and restored on the next request that continues it, also across
  restarts. The host-memory tier is off (`--conversation-cache-mib 0`); conversations under 4096 tokens are not
  parked (`--conversation-cache-min-tokens 4096`), and a park needs 1024 MiB of free RAM
  (`--conversation-cache-min-free-mib 1024`).

Two request fields steer the cache:

- `"strata_checkpoint": false`: a one-shot request that no later request continues (a classification call, a
  probe) saves no checkpoint and is not kept or parked for a next request. It still starts from a checkpoint it
  matches.
- `"strata_prefix": {"messages": 1}` (or `{"message": 0, "chars": 210000}`, or `{"tokens": 12345}`): one long
  document, many questions. The engine reads the prompt in two parts there and keeps the checkpoint at the
  boundary pinned, so the next question starts at the end of the document and 20 questions cost one read of it.
  One pinned prefix at a time; it needs `--prompt-cache 3` or more. A prefix that cannot be marked is said in the
  server's window and ignored; a malformed field is a 400.

The Monitor tab's **Conversation cache** card shows the parked conversations, how many were parked, restored and
evicted, and how many prompt tokens the cache gave back. `/metrics` has the same under `"conversation_cache"`.

## The Responses API and Codex CLI

`POST /v1/responses` speaks OpenAI's Responses API, which Codex CLI uses. It runs on the same path as
`/v1/chat/completions`, so the thinking levels, the budget and the conversation cache apply.

It is **stateless**: nothing is stored, so the client sends the whole conversation in `input` every time (Codex
does, with `store: false`). `previous_response_id`, `conversation`, `background` and the retrieve/delete/cancel
endpoints are refused with an error that says so.

| Request | What Strata does |
| --- | --- |
| `input` as a string, or as items | `message` items (`user`, `assistant`, `system`, `developer`), `reasoning`, `function_call`, `function_call_output`, `custom_tool_call(_output)` |
| `instructions` | The system message |
| `tools` | `function` tools, `namespace` tools, `custom` tools. Hosted tools (`web_search`, `file_search`, ...) are left out: the model cannot run them |
| `tool_choice` | `"none"` hides the tools; anything else lets the model choose |
| `reasoning.effort` | `none`/`minimal`, `low`, `medium`, `high`/`xhigh`; without it the config's `reasoning_effort` |
| `max_output_tokens` | The output cap (thinking included). Running out ends the response `incomplete` |
| `text.format` | `json_schema` and `json_object` use the [JSON response formats](#json-response-formats) |
| `temperature`, `top_p`, `reasoning_budget_tokens`, ... | As on the chat path |

The model's thinking comes back as a `reasoning` output item with `reasoning_text` content (streamed as
`response.reasoning_text.delta`). Send the reasoning items back with the rest of the conversation, as Codex does,
so the prompt matches what the model wrote and the cache is reused. With `"stream": true` the events are the
official ones, in order, and a `response.in_progress` event goes out every 15 s while a long prompt is read.

**Codex CLI.** In `~/.codex/config.toml`:

```toml
model = "qwen3.8-flash-next-iq3_xxs"   # any name; Strata answers with its model
model_provider = "strata"
model_context_window = 131072          # the context you chose: Codex compacts before it gets there
show_raw_agent_reasoning = true        # show the model's thinking (it writes no summaries)
# model_reasoning_effort = "medium"    # none, low, medium or high

[model_providers.strata]
name = "Strata (local)"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000        # a first, long prompt can take minutes to read
```

Codex warns `Model metadata for ... not found` for a local model name; that is expected. Two Codex keys in the
config, both off by default: `"codex_thread_titles": true` answers Codex's thread-title request without the engine
(the title is the user's line, cut to 36 characters), so the conversation keeps its cache; `"codex_compaction_cache":
true` renders Codex's compaction request with the tools of the conversation it compacts, so that request shares
its prompt with what the engine holds instead of being read again from the start.

## JSON response formats

`POST /v1/chat/completions` accepts `response_format: {"type": "json_object"}` or
`{"type": "json_schema", "json_schema": {"name": ..., "strict": true, "schema": {...}}}`. The schema must accept
only JSON objects at its root. `json_schema` is checked with the Python package `jsonschema` when it is installed;
without it the answer is only checked to be one JSON object, and the server says so once.

This is **schema prompting followed by server validation**, not grammar-constrained decoding. One generation is
made per request, with no hidden retry. Malformed JSON, schema violations and incomplete generations return
**502** with `error.code: structured_output_failed`; invalid request schemas return **400**. JSON formats combined
with tools are refused. Structured SSE buffers the answer while sending keep-alive comments and emits the content
only after validation.

## Network

The server listens on 127.0.0.1 only; this build forces it whatever `--host` or the config says. A few checks
still apply to what reaches it:

- **Host names (DNS rebinding).** A web page of another site can point its own name at `127.0.0.1` and then reach
  this server as if it were its own, so without an API key the server answers only requests whose `Host` is a name
  it knows: `localhost` (and `*.localhost`), any IP address, the address it listens on; any port. Others get
  **403** naming the setting. To reach it under another name (a reverse proxy, [llama-swap](LLAMA_SWAP.md)), add
  it: `"allowed_hosts": ["name"]` in the config or `STRATA_ALLOWED_HOSTS=name` (comma-separated); `["*"]` turns the
  check off, and so does setting an `"api_key"`.
- **API key.** `"api_key": "some-long-secret"` in the config (or `STRATA_API_KEY`): clients then send it as their
  API key. Several keys, one per client: `"api_key": "key1,key2"` or a list.
- **Web pages (CORS).** Off by default. `"cors_origins": ["https://chat.example.com"]` lets pages of those origins
  call `/v1/*` from the browser. Without an API key, a `POST` to `/v1/*` that carries an `Origin` header is
  answered only for Strata's own page, pages on `localhost` or an allowed host, the origins in `trusted_origins` or
  `cors_origins`, and browser extensions; clients that send no `Origin` (curl, the SDKs) are not affected.

## Other settings

- **Model settings in the web page.** The About tab's Model settings card shows and changes a few config keys:
  the `sampling` defaults, `reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`, `effort_position`,
  `aliases`, `idle_unload_s`, `lazy_load`, `engine_silence_s` and `api_monitor`. The earlier file is kept as
  `strata-cached.json.bak`, and the model uses the change from its next start.
- **Start the API without loading the model:** `serve/server.py ... --lazy` (or `"lazy_load": true`). The first
  generation request, or `POST /load`, loads the engine; `POST /unload` stops it. `"idle_unload_s": N` unloads
  the model after N seconds without requests.
- **An engine that says nothing:** the server ends the engine after `"engine_silence_s"` (default 300) seconds
  without a line from it during a request, and the next request starts it again. An engine that is silent and has
  also used no CPU time and moved no disk bytes for 90 s is ended sooner (`STRATA_ENGINE_STALL_S` sets it, `0` turns
  it off; it needs `psutil`).
- **API request monitor** (`"api_monitor": true`, off by default since it keeps prompts and answers in memory):
  `/api-monitor` shows the model state, active and queued requests, request bodies, output and reasoning;
  `GET /api/requests` returns summaries. The newest 100 requests are kept until restart.
- **An empty earlier reply is left out** of the prompt, so the next reply does not copy it
  (`STRATA_KEEP_EMPTY_TURNS=1` renders them as before).

## Not in this build

- One request at a time: batch slots (`"parallel"`) are not ported to Metal; several clients queue.
- MCP servers: the config's `"mcp_servers"` is ignored; the server prints one line saying so.
- Pictures: the image encoder is not set up, and an `image_url` that is an `http(s)://` address is refused (send
  the image itself as a `data:` URL only if an encoder is configured; none is here).
- Session files on disk (`--slot-save-path`) and the host-RAM conversation tier are upstream features that are
  not measured on the Mac build; the tuned config uses the root slot and the SSD tier instead.
