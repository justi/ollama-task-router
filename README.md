# ollama-task-router

A zero-dependency prompt router for Ollama. A small model classifies each prompt and routes it to
the best of three local specialists - a dual-mode coder+reasoner, a quick all-rounder, and an opt-in
hard-reasoning model - instead of one overloaded model. Needs Ollama; the three models are ~43 GB on disk, and at runtime each prompt loads
the gemma classifier plus one specialist (the code route peaks around ~30 GB of RAM) - not all three at
once unless you keep them warm.

## Models

| Variant | Base | Role | Size |
|---|---|---|---|
| `qwen-fast`    | `qwen3.6:35b-a3b` | code + reasoning (dual-mode: think off / on) | ~20 GB |
| `gpt-oss-fast` | `gpt-oss:20b`     | hard-reasoning escalation (`--reason-hard`) | ~13 GB |
| `gemma-fast`   | `gemma4:e4b`      | quick questions | ~10 GB |

## Quick start

```bash
./setup.sh                                            # pull bases + build the -fast variants (once)
./ask.py "write an is_prime function in Python"       # auto  -> qwen-fast
./ask.py --reason "prove that sqrt(2) is irrational"  # force reasoning -> qwen-fast (think on)
./ask.py --reason-hard "hardest logic puzzle ..."     # escalate         -> gpt-oss-fast
./ask.py --quick  "capital of Australia?"             # force            -> gemma-fast
```

## Routing

`gemma-fast` classifies each prompt (`code` / `reason` / `quick`) into a JSON label at
`temperature 0` - language-independent (it reads meaning, not keywords) and stable - then dispatches
to the matching model. `qwen-fast` (qwen3.6, dual-mode) handles both code (thinking OFF - thinking
hurts code) and reasoning (thinking ON - it reasons as well as gpt-oss and more reliably; see
[docs/reason-routing.md](docs/reason-routing.md)). Gemma runs with thinking OFF - both as the
classifier and on the quick route. `gpt-oss-fast` is the opt-in hard-reasoning escalation
(`--reason-hard`), not the default reason route. If gemma is unreachable it routes to the coder
(`qwen-fast`); telling `reason` from `quick` is semantic, so the offline fallback does not guess it.
Force a route with `--code` / `--reason` / `--reason-hard` / `--quick`, skip the classifier with
`--no-classify`, or point elsewhere with
`OLLAMA_HOST`. Keep models warm for instant switching: `OLLAMA_KEEP_ALIVE=30m ollama serve`.

Tuned `Modelfile.*` params come from [ollama-bench](https://github.com/justi/ollama-bench). The
router also sets `num_predict` per route (code 4000 / reason 8000 / reason-hard 10000 / quick 1500), which overrides the
Modelfile default so each task gets the budget it needs; the Modelfile value is the `ollama run` floor.

## Tests

```bash
python3 test_ask.py
```

Offline tests (routing logic, classifier failure handling, flag parsing) need nothing - the
network is mocked. The live tests verify your own setup end to end - that the `-fast` models
answer and that classification works across languages - and skip automatically until you have
built the models with `./setup.sh`.
