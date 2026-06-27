#!/usr/bin/env python3
"""Tiny task router for three local models on Ollama - zero dependencies, stdlib only.

It sends your prompt to the model that fits the task, then tells you which one it picked:
  - code-ish prompt   -> qwen-fast    (dedicated coder, no thinking, fast in a loop)
  - reasoning prompt   -> gpt-oss-fast (thinking model, for algorithms / step-by-step logic)
  - short / simple     -> gemma-fast   (tiny 10 GB all-rounder, for quick questions)

Routing: the tiny model (gemma) classifies the task (code/reason/quick) at temperature 0, so the
choice is reliable AND deterministic - same prompt always routes the same way. If gemma is down or
unsure it falls back to a keyword heuristic. You can always override:

  ./ask.py "write an is_prime function in Python"          # auto (gemma classifies) -> qwen-fast
  ./ask.py --reason "prove that sqrt(2) is irrational"     # force reasoning
  ./ask.py --quick  "capital of Australia?"                 # force the tiny model
  ./ask.py --code   "refactor this loop ..."               # force the coder
  ./ask.py --keyword "..."                                  # skip gemma, use keyword router only

Build the models first with ./setup.sh. Override the endpoint with OLLAMA_HOST.
"""
import json
import os
import re
import sys
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST

# task -> (model, think, num_predict). gpt-oss thinks (cannot be disabled); qwen/gemma do not.
ROUTES = {
    "code":   ("qwen-fast",    None,   4000),  # coder, no thinking
    "reason": ("gpt-oss-fast", "high", 6000),  # thinking model, deep reasoning
    "quick":  ("gemma-fast",   False,  1500),  # tiny, thinking off = fast factual answer
}

CODE_HINTS = r"\b(napisz|funkcj|function|def |class |refaktor|refactor|debug|bug|błąd|blad|kod|code|python|javascript|typescript|rust|golang|sql|regex|skrypt|script|implement|napraw)\b"
REASON_HINTS = r"\b(dlaczego|udowodnij|prove|dowód|dowod|algorytm|algorithm|zagadk|krok po kroku|step by step|wyjaśnij dlaczego|wyjasnij dlaczego|optymaln|complexity|big.?o|oblicz|policz)\b"


def route_keyword(prompt: str) -> str:
    """Fallback router: crude keyword heuristic. Misses tasks that lack obvious keywords."""
    p = prompt.lower()
    if re.search(REASON_HINTS, p):
        return "reason"
    if re.search(CODE_HINTS, p):
        return "code"
    if len(prompt.split()) <= 8:
        return "quick"
    return "code"  # default: the daily-driver coder


def classify_with_gemma(prompt: str, model: str = "gemma-fast"):
    """LLM router: let the tiny model label the task. More reliable than keywords for subtle
    prompts (e.g. a logic puzzle with no 'prove'/'compute' keyword). Returns the label or None
    on any failure (caller then falls back to keywords). One short call - gemma is fast."""
    instr = (
        "Classify the task below into ONE category and reply with ONLY one word.\n"
        "- code   = writing/fixing/explaining code, functions, debugging, refactoring, regex, SQL\n"
        "- reason = logic, algorithms, math, puzzles, proofs, step-by-step reasoning\n"
        "- quick  = a simple factual question with a short answer\n\n"
        f"Task: {prompt}\n\nCategory (code, reason, or quick):")
    try:
        # think=False: we want the bare label, not gemma's reasoning (which would eat the budget)
        out = (ask(model, instr, False, 20, temperature=0).get("response") or "").lower()
    except Exception:
        return None
    found = re.findall(r"\b(code|reason|quick)\b", out)
    return found[-1] if found else None


def ask(model, prompt, think, num_predict, temperature=None):
    opts = {"num_predict": num_predict}
    if temperature is not None:
        opts["temperature"] = temperature  # 0 = deterministic (used for stable routing)
    payload = {"model": model, "prompt": prompt, "stream": False, "options": opts}
    if think is not None:
        payload["think"] = think
    req = urllib.request.Request(
        HOST + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    args = sys.argv[1:]
    forced, no_llm = None, False
    if "--keyword" in args:
        no_llm = True
        args = [a for a in args if a != "--keyword"]
    for flag in ("--code", "--reason", "--quick"):
        if flag in args:
            forced = flag[2:]
            args = [a for a in args if a != flag]
    if not args:
        print(__doc__)
        sys.exit(1)
    prompt = " ".join(args)
    # routing: explicit flag > gemma classifier (temp 0, stable) > keyword fallback
    if forced:
        task, how = forced, "flag"
    else:
        cls = None if no_llm else classify_with_gemma(prompt)
        task, how = (cls, "gemma") if cls else (route_keyword(prompt), "keyword")
    model, think, num_predict = ROUTES[task]
    print(f"[router] task={task} (via {how}) -> {model}"
          f"{' (thinking, this may take a moment)' if think else ''}\n", file=sys.stderr)
    try:
        resp = ask(model, prompt, think, num_predict)
    except urllib.error.URLError as e:
        print(f"[!] Can't reach Ollama ({HOST}): {e}\n    Is the server running and is model '{model}' built? (./setup.sh)", file=sys.stderr)
        sys.exit(1)
    print((resp.get("response") or "").strip())
    if resp.get("done_reason") == "length":
        print(f"\n[router] note: answer truncated at num_predict={num_predict} - raise the budget if it looks cut off.", file=sys.stderr)


if __name__ == "__main__":
    main()
