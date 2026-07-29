# ollama-task-router

A zero-dependency prompt router for Ollama. A small model classifies each prompt and routes it to
the best of three local specialists - a coder, a reasoner, and a quick all-rounder - instead of one
overloaded model. Needs Ollama and ~43 GB of RAM (all three models; gemma is the classifier).

## Models

| Variant | Base | Role | Size |
|---|---|---|---|
| `qwen-fast`    | `qwen3.6:35b-a3b` | code + daily driver (dual-mode) | ~20 GB |
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

`gemma-fast` classifies each prompt (`code` / `reason` / `quick`) into a JSON label at
`temperature 0` - language-independent (it reads meaning, not keywords) and stable - then dispatches
to the matching model - the code route runs `qwen-fast` (qwen3.6, dual-mode) with thinking forced OFF
(thinking hurts code), `gpt-oss-fast` always thinks, and Gemma runs with thinking OFF too - both as
the classifier and on the quick route. If gemma is unreachable it routes to the
coder (`qwen-fast`); telling `reason`
from `quick` is semantic, so the offline fallback does not guess it. Force a model with `--code` /
`--reason` / `--quick`, skip the classifier with `--no-classify`, or point elsewhere with
`OLLAMA_HOST`. Keep models warm for instant switching: `OLLAMA_KEEP_ALIVE=30m ollama serve`.

Tuned `Modelfile.*` params come from [ollama-bench](https://github.com/justi/ollama-bench). The
router also sets `num_predict` per route (code 4000 / reason 6000 / quick 1500), which overrides the
Modelfile default so each task gets the budget it needs; the Modelfile value is the `ollama run` floor.

## Tests

```bash
python3 test_ask.py
```

Offline tests (routing logic, classifier failure handling, flag parsing) need nothing - the
network is mocked. The live tests verify your own setup end to end - that the `-fast` models
answer and that classification works across languages - and skip automatically until you have
built the models with `./setup.sh`.
