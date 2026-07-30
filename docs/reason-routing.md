# Decision: what should the `reason` route dispatch to?

**TL;DR:** measured — `gpt-oss` (thinking=high) shows **no reasoning advantage** over
`qwen3.6` (thinking=on), and gpt-oss additionally suffers **runaway-thinking truncation** on hard
puzzles. Since qwen3.6 is already the dual-mode base, routing `reason` to gpt-oss buys nothing and
adds a 13 GB model plus a failure mode. Plan: route `reason` to `qwen-fast` with thinking ON, and
keep gpt-oss as an explicit escalation, not the default.

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

## Results (n=5, preliminary — hardening to n=30)

| Config | Score | Truncations on q3 (Monty Hall) |
|---|---:|---|
| qwen3.6 think=on @ num_predict 10000 | **5.60 / 6** (28/30) | 2 (across all puzzles) |
| gpt-oss think=high @ 6000 (router's reason budget) | 5.40 / 6 (27/30) | 3 / 5 |
| gpt-oss think=high @ 10000 (equal budget) | 5.20 / 6 (26/30) | 3 / 5 |

Two findings:

1. **No quality advantage (weak signal).** qwen is top in both comparisons; at equal budget qwen
   5.60 > gpt-oss 5.20. Extra budget did NOT help gpt-oss. Margins are 1-2 answers / 30 at n=5, so
   this is a *tie with qwen slightly ahead*, not a rout — but it refutes "gpt-oss reasons better".
2. **Runaway-thinking truncation (strong, reproducible signal).** gpt-oss-high failed to finish
   Monty Hall (q3) 3/5 times **even at 10000 tokens** — so on a genuinely counterintuitive problem
   it can emit no answer at all, regardless of budget. qwen does not do this.

## Conclusion

Routing `reason` to gpt-oss:
- buys **no** measurable reasoning quality over qwen3.6-think-on,
- adds a **truncation risk** on hard problems,
- adds a 13 GB model and heavy hidden thinking (slow visible output),
- while qwen3.6 is **already the base** (decision B).

**Decision:** the `reason` route dispatches to `qwen-fast` with thinking ON. `gpt-oss` stays
available as an explicit escalation (a `--reason-hard` flag), not the default reason route.

## Caveats

n=5 with small margins is a decision signal, not a published fact (temp 1.0 is stochastic; the
judge is an LLM and fallible; 6 puzzles is a narrow set). Hardening to n=30 at equal budget is in
progress; this note is updated with the harder numbers when it completes.
