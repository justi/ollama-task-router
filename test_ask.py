#!/usr/bin/env python3
"""Tests for the task router. Stdlib only - no pytest, nothing to install.

    python3 test_ask.py            # run everything
    python3 -m unittest -v test_ask

Two layers:
  * Offline (always runs) - routing logic, classifier failure handling, flag parsing.
    The network call is mocked, so no Ollama is needed.
  * Live (auto-skipped unless Ollama is up with the -fast models built) - verifies YOUR
    configuration end to end: that the models answer and that classification works across
    languages. Build them first with ./setup.sh.
"""
import contextlib
import io
import json
import sys
import unittest
import urllib.request

import ask

ROUTE_MODELS = {model for model, _, _ in ask.ROUTES.values()}


def _ollama_tags():
    """Model names Ollama currently has, or None if the server is unreachable."""
    try:
        with urllib.request.urlopen(ask.HOST + "/api/tags", timeout=4) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    names = {m["name"] for m in data.get("models", [])}
    return names | {n.split(":")[0] for n in names}

_TAGS = _ollama_tags()
LIVE = _TAGS is not None and ROUTE_MODELS <= _TAGS
LIVE_REASON = "live: Ollama must be up with the -fast models built (run ./setup.sh)"


@contextlib.contextmanager
def mock_ask(recorder=None, response=None, raises=None):
    """Replace ask.ask so routing can be tested without a server. Records every call and
    returns `response` (default: a non-empty answer). The same payload is returned to the
    classifier and to generation, so `response` also drives what the classifier parses."""
    original = ask.ask

    def fake(model, prompt, think, num_predict, temperature=None, timeout=900, fmt=None):
        if recorder is not None:
            recorder.append({"model": model, "think": think, "num_predict": num_predict, "fmt": fmt})
        if raises is not None:
            raise raises
        return response if response is not None else {"response": "ok", "done_reason": "stop"}

    ask.ask = fake
    try:
        yield
    finally:
        ask.ask = original


def run_main(argv):
    """Invoke ask.main() with the given args, capturing stdout/stderr."""
    saved = sys.argv
    sys.argv = ["ask.py"] + argv
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                ask.main()
            except SystemExit:
                pass
    finally:
        sys.argv = saved
    return out.getvalue(), err.getvalue()


def route(argv, response=None):
    """Run the router and return the generation call it made (model, think, ...) plus the
    full call list (length tells you whether the classifier ran)."""
    calls = []
    with mock_ask(recorder=calls, response=response):
        run_main(argv)
    return (calls[-1] if calls else None), calls


def enum_response(category):
    return {"response": json.dumps({"category": category}), "done_reason": "stop"}


class TestFallback(unittest.TestCase):
    """route_no_classifier: the language-independent fallback when the classifier is gone."""

    def test_returns_a_valid_route(self):
        self.assertIn(ask.route_no_classifier("anything"), ask.ROUTES)

    def test_language_independent(self):
        # No natural-language keyword list, so every language resolves the same safe way.
        for prompt in ["napisz funkcje", "warum ist der Himmel blau", "quoi de neuf",
                       "rozwiaz zagadke", "naptu az eget", ""]:
            self.assertEqual(ask.route_no_classifier(prompt), "code")


class TestClassifier(unittest.TestCase):
    """classify_with_gemma must return a valid label or None - never crash, never guess."""

    def test_parses_enum(self):
        with mock_ask(response=enum_response("reason")):
            self.assertEqual(ask.classify_with_gemma("x"), "reason")

    def test_uses_constrained_format(self):
        calls = []
        with mock_ask(recorder=calls, response=enum_response("code")):
            ask.classify_with_gemma("x")
        self.assertIsNotNone(calls[0]["fmt"], "classifier must request a constrained JSON schema")

    def test_garbage_is_rejected(self):
        with mock_ask(response={"response": "sure, this is code!"}):
            self.assertIsNone(ask.classify_with_gemma("x"))

    def test_out_of_enum_is_rejected(self):
        with mock_ask(response=enum_response("banana")):
            self.assertIsNone(ask.classify_with_gemma("x"))

    def test_server_error_is_swallowed(self):
        with mock_ask(raises=RuntimeError("server down")):
            self.assertIsNone(ask.classify_with_gemma("x"))


class TestRouteMapping(unittest.TestCase):
    """Each task must reach the right model with the right thinking setting."""

    def test_force_code(self):
        gen, _ = route(["--code", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("qwen-fast", None))

    def test_force_reason(self):
        gen, _ = route(["--reason", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("gpt-oss-fast", "high"))

    def test_force_quick(self):
        gen, _ = route(["--quick", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("gemma-fast", False))

    def test_competing_flags_use_argv_order(self):
        self.assertEqual(route(["--code", "--reason", "x"])[0]["model"], "qwen-fast")
        self.assertEqual(route(["--reason", "--code", "x"])[0]["model"], "gpt-oss-fast")

    def test_no_classify_skips_classifier(self):
        # --no-classify -> exactly one call (generation), routed to the coder, classifier never run.
        gen, calls = route(["--no-classify", "solve this puzzle step by step"])
        self.assertEqual(gen["model"], "qwen-fast")
        self.assertEqual(len(calls), 1)

    def test_classifier_label_drives_route(self):
        self.assertEqual(route(["x"], response=enum_response("reason"))[0]["model"], "gpt-oss-fast")
        self.assertEqual(route(["x"], response=enum_response("quick"))[0]["model"], "gemma-fast")

    def test_classifier_failure_falls_back_to_coder(self):
        gen, _ = route(["x"], response={"response": "not json"})
        self.assertEqual(gen["model"], "qwen-fast")


@unittest.skipUnless(LIVE, LIVE_REASON)
class TestLive(unittest.TestCase):
    """Runs only when your models are built - this is the "does my config work" check."""

    def test_classifier_works_in_every_language(self):
        # The point of the redesign: classification is by meaning, so any language resolves
        # to a valid label (not None).
        for prompt in ["write an is_prime function", "udowodnij ze sqrt(2) jest niewymierne",
                       "Beweise, dass die Wurzel aus 2 irrational ist",
                       "Quelle est la capitale de l'Australie?",
                       "Resuelve este acertijo logico paso a paso"]:
            with self.subTest(prompt=prompt):
                self.assertIn(ask.classify_with_gemma(prompt), ask.ROUTES)

    def test_classifier_is_accurate_on_clear_cases(self):
        self.assertEqual(ask.classify_with_gemma("write an is_prime function in Python"), "code")
        self.assertEqual(ask.classify_with_gemma("What is the capital of Australia?"), "quick")

    def test_quick_model_answers(self):
        resp = ask.ask("gemma-fast", "What is the capital of Australia?", False, 200)
        self.assertTrue(resp.get("response", "").strip(), "gemma-fast returned an empty answer")

    def test_code_model_answers(self):
        resp = ask.ask("qwen-fast", "Write a one-line Python is_prime function.", None, 512)
        self.assertTrue(resp.get("response", "").strip(), "qwen-fast returned an empty answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
