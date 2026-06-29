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
            recorder.append({"model": model, "prompt": prompt, "think": think,
                             "num_predict": num_predict, "temperature": temperature,
                             "timeout": timeout, "fmt": fmt})
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

    def test_returns_the_documented_coder_default(self):
        # The contract is "always the coder", not merely "some valid route" - an always-quick
        # fallback would also be a valid route but would break the documented behavior.
        self.assertEqual(ask.route_no_classifier("anything"), "code")

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

    def test_classify_call_is_constrained_and_fast(self):
        calls = []
        with mock_ask(recorder=calls, response=enum_response("code")):
            ask.classify_with_gemma("x")
        first = calls[0]
        self.assertEqual(first["fmt"], ask.CLASSIFY_SCHEMA, "must constrain output to the enum schema")
        self.assertEqual(first["temperature"], 0, "routing must be temperature 0")
        self.assertEqual(first["timeout"], ask.ROUTE_TIMEOUT, "routing must fail fast, not block")
        self.assertEqual(first["num_predict"], 24)

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
        gen, _ = route(["--code", "--reason", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("qwen-fast", None))
        gen, _ = route(["--reason", "--code", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("gpt-oss-fast", "high"))

    def test_losing_flag_is_stripped_from_prompt(self):
        # The non-winning task flag must not leak into the text sent to the model.
        gen, _ = route(["--code", "--reason", "real prompt"])
        self.assertEqual(gen["prompt"], "real prompt")

    def test_no_classify_skips_classifier(self):
        # Spy on the classifier itself: --no-classify must never invoke it (counting network
        # calls is not enough - a classifier that returned without calling ask would slip through).
        original = ask.classify_with_gemma
        invoked = []
        ask.classify_with_gemma = lambda *a, **k: invoked.append(1) or "reason"
        try:
            gen, _ = route(["--no-classify", "solve this puzzle step by step"])
        finally:
            ask.classify_with_gemma = original
        self.assertEqual(invoked, [], "--no-classify must not run the classifier")
        self.assertEqual(gen["model"], "qwen-fast")

    def test_classifier_label_drives_route(self):
        # The classifier label must select the model, AND generation must actually happen after
        # it. For the quick route both calls hit gemma-fast, so assert the ORDER: a constrained
        # classify call, then an unconstrained generation call to the mapped model.
        for label, model in [("reason", "gpt-oss-fast"), ("quick", "gemma-fast")]:
            with self.subTest(label=label):
                calls = []
                with mock_ask(recorder=calls, response=enum_response(label)):
                    run_main(["x"])
                self.assertGreaterEqual(len(calls), 2, "generation call is missing")
                self.assertIsNotNone(calls[0]["fmt"], "first call should be the constrained classify")
                self.assertIsNone(calls[-1]["fmt"], "last call should be generation, not the classifier")
                self.assertEqual(calls[-1]["model"], model)

    def test_classifier_failure_retries_then_falls_back(self):
        # Malformed JSON -> two constrained attempts -> generation to the coder.
        calls = []
        with mock_ask(recorder=calls, response={"response": "not json"}):
            run_main(["x"])
        self.assertEqual(len(calls), 3, "expected two classify retries then one generation")
        self.assertIsNotNone(calls[0]["fmt"])
        self.assertIsNotNone(calls[1]["fmt"])
        self.assertIsNone(calls[2]["fmt"])
        self.assertEqual(calls[2]["model"], "qwen-fast")


@unittest.skipUnless(LIVE, LIVE_REASON)
class TestLive(unittest.TestCase):
    """Runs only when your models are built - this is the "does my config work" check."""

    def test_classifier_round_trips_in_every_language(self):
        # Every language must yield a valid enum label (the constrained round-trip works), and
        # the set must NOT collapse to one label - that would mean the classifier is stuck (e.g.
        # always "code") rather than actually reading meaning. We do NOT pin which label each
        # prompt gets: the reason-vs-quick boundary is a soft model judgement, not router code.
        prompts = ["write an is_prime function in Python",
                   "udowodnij, ze sqrt(2) jest niewymierne",
                   "Beweise, dass die Wurzel aus 2 irrational ist",
                   "Quelle est la capitale de l'Australie?",
                   "Resuelve este acertijo logico paso a paso"]
        labels = []
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                label = ask.classify_with_gemma(prompt)
                self.assertIn(label, ask.ROUTES, "classifier returned no valid label")
                labels.append(label)
        self.assertGreaterEqual(len(set(labels)), 2,
                                f"classifier looks stuck - only produced {set(labels)}")

    def test_classifier_separates_code_from_non_code(self):
        # The one hard discrimination we can pin without flakiness: an explicit "write a function"
        # is code; a plain factual question is never code (quick vs reason stays soft).
        self.assertEqual(ask.classify_with_gemma("write an is_prime function in Python"), "code")
        self.assertIn(ask.classify_with_gemma("What is the capital of Australia?"), {"quick", "reason"})

    def test_quick_model_answers_correctly(self):
        resp = ask.ask("gemma-fast", "What is the capital of Australia?", False, 200)
        text = resp.get("response", "")
        self.assertTrue(text.strip(), "gemma-fast returned an empty answer")
        self.assertIn("canberra", text.lower(), "gemma-fast did not actually answer the question")

    def test_code_model_answers_without_truncating(self):
        resp = ask.ask("qwen-fast", "Write a one-line Python is_prime function.", None, 512)
        text = resp.get("response", "")
        self.assertTrue(text.strip(), "qwen-fast returned an empty answer")
        self.assertNotEqual(resp.get("done_reason"), "length",
                            "answer was truncated at num_predict - the budget is too small")
        self.assertIn("prime", text.lower(), "qwen-fast did not return an is_prime function")


if __name__ == "__main__":
    unittest.main(verbosity=2)
