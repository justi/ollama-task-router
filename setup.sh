#!/usr/bin/env bash
# One-shot setup for the local three-model router on Ollama.
# Pulls the base models and builds the tuned -fast variants from the Modelfiles next to this script.
set -euo pipefail
cd "$(dirname "$0")"

command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com"; exit 1; }

echo "== Pulling base models (~31 GB total) =="
ollama pull qwen3-coder:30b    # ~18 GB - coder, daily driver
ollama pull gpt-oss:20b        # ~13 GB - thinking model, for reasoning
ollama pull gemma4:e4b         # ~10 GB - tiny all-rounder

echo "== Building tuned -fast variants =="
ollama create qwen-fast    -f Modelfile.qwen-fast
ollama create gpt-oss-fast -f Modelfile.gpt-oss-fast
ollama create gemma-fast   -f Modelfile.gemma-fast

echo
echo "Done. Three models ready: qwen-fast, gpt-oss-fast, gemma-fast"
echo "Try the router:"
echo "  ./ask.py 'write an is_prime function in Python'         # -> qwen-fast (code)"
echo "  ./ask.py --reason 'prove that sqrt(2) is irrational'    # -> gpt-oss-fast"
echo "  ./ask.py --quick 'capital of Australia?'                # -> gemma-fast"
