# ollama-task-router

A zero-dependency prompt router for Ollama. A small model classifies each prompt and routes it to
the best of three local specialists - a coder, a reasoner, and a quick all-rounder - instead of one
overloaded model. Needs Ollama and ~31 GB of RAM.

## Models

| Variant | Base | Role | Size |
|---|---|---|---|
| `qwen-fast`    | `qwen3-coder:30b` | code, daily work | ~18 GB |
| `gpt-oss-fast` | `gpt-oss:20b`     | reasoning, step-by-step logic | ~13 GB |
| `gemma-fast`   | `gemma4:e4b`      | quick questions | ~10 GB |

## Quick start

```bash
./setup.sh                                            # pull bases + build the -fast variants (once)
./ask.py "write an is_prime function in Python"       # auto  -> qwen-fast
./ask.py --reason "prove that sqrt(2) is irrational"  # force -> gpt-oss-fast
./ask.py --quick  "capital of Australia?"             # force -> gemma-fast
```

## Routing

`gemma-fast` classifies each prompt (`code` / `reason` / `quick`) at `temperature 0`, so routing is
accurate and deterministic, then dispatches to the matching model. If gemma is down it falls back to
a keyword heuristic. Force a model with `--code` / `--reason` / `--quick`, skip the classifier with
`--keyword`, or point elsewhere with `OLLAMA_HOST`. Keep models warm for instant switching:
`OLLAMA_KEEP_ALIVE=30m ollama serve`.

Tuned `Modelfile.*` params come from [ollama-bench](https://github.com/justi/ollama-bench).
