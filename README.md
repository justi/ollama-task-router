# ollama-task-router

A prompt router for Ollama: a small model **classifies the task** and sends it to the best of
**three local specialists** - a fast coder for everyday work, a thinking model for hard reasoning,
and a light all-rounder for quick questions. A team of models instead of one overloaded giant. For
a Mac with 64 GB RAM (or any machine that can hold ~31 GB in memory).

Why three, not one: in measurements the models' strengths invert - the best at code is not the best
at reasoning (benchmarks: [github.com/justi/ollama-bench](https://github.com/justi/ollama-bench)).
No single model covers both axes well, so the router sends each task where it performs best.

## Models

| Variant | Base | Role | Size |
|---|---|---|---|
| `qwen-fast`    | `qwen3-coder:30b` | code, daily work, agent loops | ~18 GB |
| `gpt-oss-fast` | `gpt-oss:20b`     | algorithms, logic debugging, step-by-step | ~13 GB |
| `gemma-fast`   | `gemma4:e4b`      | quick questions, small stuff | ~10 GB |

## Quick start

```bash
# 1. Ollama must be installed: https://ollama.com
# 2. Pull the bases + build the -fast variants (once):
./setup.sh

# 3. Ask - the router picks the model by task type:
./ask.py "write an is_prime function in Python"          # auto -> qwen-fast
./ask.py --reason "prove that sqrt(2) is irrational"     # forced -> gpt-oss-fast
./ask.py --quick  "capital of Australia?"                 # forced -> gemma-fast
```

## How the router works

`ask.py` is pure stdlib (zero dependencies). First the small **gemma-fast classifies** the task
into one of three categories (`code` / `reason` / `quick`) at `temperature 0` - this makes the
choice accurate for subtle prompts (e.g. a logic puzzle with no "prove" keyword) and at the same
time deterministic: the same prompt always lands on the same model. Classification is one short
call with thinking disabled, so it is fast. The router then dispatches: `code` -> `qwen-fast`,
`reason` -> `gpt-oss-fast`, `quick` -> `gemma-fast`.

If gemma is unavailable or returns something outside the three labels, the router **falls back to a
keyword heuristic** (`route_keyword`). You can always force a model manually with `--code` /
`--reason` / `--quick`, or skip gemma and use keywords only with `--keyword`. Override the endpoint
with `OLLAMA_HOST`.

Want stronger routing (vector semantics, fallbacks, an OpenAI-compatible proxy)? The same
"team of models instead of one giant" pattern scales up with
[LiteLLM](https://github.com/BerriAI/litellm) or [RouteLLM](https://github.com/lm-sys/RouteLLM).

## Tips

- Keep the models warm so switching is instant: `OLLAMA_KEEP_ALIVE=30m ollama serve`.
- `gpt-oss-fast` thinks before answering - on hard questions give it a moment (tens of seconds).
- The parameters in `Modelfile.*` come from the benchmark (link above) - reproduce and tune them yourself.

Full setup walkthrough and rationale (in Polish): the article "Setup lokalnego LLM na macOS z 64 GB RAM".
