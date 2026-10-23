# Running Strata behind llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) puts several model servers behind one address and starts the one a
request names, stopping the others to free the GPU. Strata works as one of its models. This page is what upstream needed to
get it running; it was tested there with llama-swap's default stop behaviour and clients reaching the machine by its
host name. It is not measured again on the Mac build.

## The entry

Set Strata up as usual first ([GUIDE.md](GUIDE.md)), then point llama-swap at the server directly, with llama-swap's
port instead of the guide's 8000:

```yaml
healthCheckTimeout: 300            # loading the model takes a while; give it room

models:
  strata:
    cmd: >-
      sh -c 'cd /path/to/the/repository && exec .venv/bin/python serve/server.py
      --engine strata --config strata-cached.json --port ${PORT}'
    env: ["STRATA_ALLOWED_HOSTS=my-server"]     # the host name your clients use -- see below
    checkEndpoint: /health
    ttl: 1800
```

- `exec` makes the Python server the process llama-swap stops. On llama-swap's stop (SIGTERM) the server stops its
  engine, which frees the GPU memory.
- `checkEndpoint: /health` -- Strata's `/health` answers only once the model is loaded.
- If other models share the GPU, put them in a llama-swap group or let swapping unload them; one engine on the GPU at a
  time (the model is 47 GB resident).

## The trap: "Host ... is not allowed"

The first request through llama-swap failed at once with:

```text
403 Host 'my-server:8040' is not allowed (DNS rebinding protection)
```

Strata only answers to the host names it knows, so a web page cannot reach it by pointing a DNS name at 127.0.0.1.
llama-swap passes the client's Host header through, so a client that reaches the proxy by name (`http://my-server:8040`)
is refused. Requests by IP address and to `localhost` always pass, which is why it can work in a quick test and fail
from another machine. (The Strata server itself always listens on 127.0.0.1 in this build; llama-swap is what other
machines reach.) Any one of these fixes it:

- `STRATA_ALLOWED_HOSTS=my-server` in the entry's `env` (comma-separated names; ports are ignored), or
- `"allowed_hosts": ["my-server"]` in `strata-<model>.json`, or
- an `"api_key"`, which turns the check off -- then every client must send the key.

## Strata's own pages through llama-swap

llama-swap forwards `/upstream/<model>/...` unchanged, so Strata's status and health pages stay reachable, and calling one
also loads the model without sending a chat request:

```bash
curl http://my-server:8040/upstream/strata/health
curl http://my-server:8040/upstream/strata/v1/status
```

## Claude Code and other Anthropic-API clients: the thinking level

Claude Code (and the Claude Agent SDK) send `"output_config": {"effort": "high"}` with every request unless told
otherwise, so Strata thinks at its highest level (seen by logging the requests). `CLAUDE_CODE_EFFORT_LEVEL=medium` (or
`low`) in the client's environment changes it. The tuned config in [GUIDE.md](GUIDE.md) also puts the effort at the end
of the prompt (`"effort_position": "end"`), so a request that changes only the effort keeps the conversation cache.
