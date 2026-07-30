#!/usr/bin/env bash
# One-shot setup for the local three-model router on Ollama.
# Pulls the base models and builds the tuned -fast variants from the Modelfiles next to this script.
#
# Why --check exists: `ollama create` snapshots the base's weight blob at build time - it is NOT a
# live link to the base tag. A later `ollama pull <base>` moves the tag to new weights while the
# -fast variant stays frozen on the old blob (silently: it still runs and still shows in `ollama
# list`). We hit exactly this - a stale qwen-fast was built from an older base, lost its `thinking`
# capability, and the reason route started returning HTTP 400. `./setup.sh --check` compares weight
# digests so the drift is visible; `./setup.sh` rebuilds from the current bases and fixes it.
set -euo pipefail
cd "$(dirname "$0")"

command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com"; exit 1; }

# Each -fast variant and the base tag its Modelfile is built FROM.
FAST_VARIANTS=(
  "qwen-fast=qwen3.6:35b-a3b"
  "gpt-oss-fast=gpt-oss:20b"
  "gemma-fast=gemma4:e4b"
)

# The model-weights blob a model resolves to right now (empty if the model is absent).
weight_blob() {
  ollama show "$1" --modelfile 2>/dev/null | awk '/^FROM /{print $2; exit}' | sed 's#.*/##'
}

# Compare each -fast variant's build-time base blob with the base tag's current blob.
check_staleness() {
  local stale=0 pair fast base fb bb
  for pair in "${FAST_VARIANTS[@]}"; do
    fast="${pair%%=*}"; base="${pair#*=}"
    fb="$(weight_blob "$fast")"
    bb="$(weight_blob "$base")"
    if [[ -z "$fb" ]]; then
      echo "  --  $fast: not built yet"
    elif [[ -z "$bb" ]]; then
      echo "  --  $base: base not pulled"
    elif [[ "$fb" == "$bb" ]]; then
      echo "  ok  $fast in sync with $base"
    else
      echo "  STALE  $fast built from ${fb#sha256-} but $base is now ${bb#sha256-}"
      stale=1
    fi
  done
  return "$stale"
}

if [[ "${1:-}" == "--check" ]]; then
  echo "== Checking -fast variants against their bases =="
  if check_staleness; then
    echo "All -fast variants are in sync."
    exit 0
  fi
  echo "Some variants are stale - run ./setup.sh to rebuild them from the current bases."
  exit 1
fi

echo "== Pulling base models (~43 GB total) =="
ollama pull qwen3.6:35b-a3b    # ~20 GB - dual-mode base, daily driver (code think-off / reason think-on)
ollama pull gpt-oss:20b        # ~13 GB - thinking model, for reasoning
ollama pull gemma4:e4b         # ~10 GB - tiny all-rounder

echo "== Building tuned -fast variants =="
ollama create qwen-fast    -f Modelfile.qwen-fast
ollama create gpt-oss-fast -f Modelfile.gpt-oss-fast
ollama create gemma-fast   -f Modelfile.gemma-fast

echo "== Verifying variants are in sync with their bases =="
check_staleness || true

echo
echo "Done. Three models ready: qwen-fast, gpt-oss-fast, gemma-fast"
echo "Re-run ./setup.sh after an 'ollama pull' updates a base; './setup.sh --check' reports drift without rebuilding."
echo "Try the router:"
echo "  ./ask.py 'write an is_prime function in Python'         # -> qwen-fast (code)"
echo "  ./ask.py --reason 'prove that sqrt(2) is irrational'    # -> qwen-fast (think on)"
echo "  ./ask.py --reason-hard 'hardest logic puzzle ...'       # -> gpt-oss-fast"
echo "  ./ask.py --quick 'capital of Australia?'                # -> gemma-fast"
