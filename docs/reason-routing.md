# Decision: what should the `reason` route dispatch to?

**TL;DR:** measured (n=30) — `gpt-oss` (thinking=high) shows **no reasoning advantage** over
`qwen3.6` (thinking=on), and gpt-oss additionally suffers **runaway-thinking truncation** on hard
puzzles roughly twice as often. Since qwen3.6 is already the dual-mode base, routing `reason` to
gpt-oss buys nothing and adds a 13 GB model plus a worse failure mode. Decision: route `reason` to
`qwen-fast` with thinking ON, and keep gpt-oss as an explicit escalation, not the default.

## The question

The router's original policy sent every `reason` prompt to `gpt-oss-fast`. That rested on an
**unmeasured assumption**: that gpt-oss reasons better than qwen3.6 with thinking on. After
"decision B" made qwen3.6 the dual-mode daily driver (code = think off, reason = think on), the
assumption was worth testing.

## Method (reused, not invented)

- Benchmark: `ollama-bench` reasoning suite — **6 curated PL logic puzzles** with one unambiguous
  correct answer each (knight/knave, Monty Hall, wolf-goat-cabbage, ASCII code, family counting,
  string reverse). `bench_reasoning.py` generates, `grade_reasoning.py` grades.
- Judge: **`phi4-best`** — a strong reasoner that is NOT one of the two contestants (avoids
  self-judging bias).
- Host: **darwine (RTX 5090)** — reasoning *quality* is machine-independent, and the 5090 has no
  CPU spill and doesn't heat the Mac.
- Contestants: `qwen3.6:35b-a3b` think=on vs `gpt-oss:20b` think=high (temp 1.0, its official
  reasoning sampling).

## Results

Hardened to **n=30** at equal budget (num_predict 10000); the initial n=5 sweep agrees.

| Config | n=5 | **n=30 (equal budget 10000)** |
|---|---:|---:|
| qwen3.6 think=on | 5.60/6 @10000 | **5.47/6** (~164/180) |
| gpt-oss think=high | 5.40 @6000 / 5.20 @10000 | **5.33/6** (~160/180) |

Truncation on q3 (Monty Hall) at num_predict=10000: qwen **11/30**, gpt-oss **20/30**.

Two findings:

1. **No quality advantage.** qwen is ahead in *every* comparison (n=5 and n=30). At n=30 equal
   budget: 5.47 vs 5.33 (~91% vs 89%). The n=5 gap narrowed at n=30 (regression to the mean), but
   the direction is consistent — gpt-oss shows no reasoning edge over qwen3.6-think-on.
2. **Runaway-thinking truncation.** At 10000 tokens **both** models occasionally fail to finish the
   hardest puzzles, but gpt-oss does so **~1.8x more often** (20/30 vs 11/30, concentrated on Monty
   Hall). A difference of degree, not kind — gpt-oss is the less reliable reasoner here.

## Conclusion

Routing `reason` to gpt-oss:
- buys **no** measurable reasoning quality over qwen3.6-think-on,
- is **less reliable** on hard problems (more truncation),
- adds a 13 GB model and heavy hidden thinking (slow visible output),
- while qwen3.6 is **already the base** (decision B).

**Decision:** the `reason` route dispatches to `qwen-fast` with thinking ON. `gpt-oss` stays
available as an explicit escalation (a `--reason-hard` flag), not the default reason route.

## Caveats

The quality margin is small (~2 percentage points) and the judge is an LLM (fallible); the puzzle
set is 6 items. So the honest claim is "gpt-oss has no reasoning advantage and worse reliability,"
not "qwen is decisively better." The direction is consistent across n=5 and n=30, and the
truncation asymmetry is large and reproducible.
