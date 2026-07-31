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
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
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
    """Replace BOTH ask.ask (classifier / constrained calls) and ask.run (generation) so routing can
    be tested without a server. Records every call; `response` (a /api/generate-style dict) drives
    what the classifier parses AND what generation returns. calls[-1] is the generation call."""
    original_ask, original_run = ask.ask, ask.run

    def fake_ask(model, prompt, think, num_predict, temperature=None, timeout=900, fmt=None):
        if recorder is not None:
            recorder.append({"model": model, "prompt": prompt, "think": think,
                             "num_predict": num_predict, "temperature": temperature,
                             "timeout": timeout, "fmt": fmt})
        if raises is not None:
            raise raises
        return response if response is not None else {"response": "ok", "done_reason": "stop"}

    def fake_run(endpoint, payload, stream=False, timeout=900, on_chunk=None):
        if recorder is not None:
            msgs = payload.get("messages")
            recorder.append({"model": payload.get("model"), "think": payload.get("think"),
                             "num_predict": (payload.get("options") or {}).get("num_predict"),
                             "prompt": payload.get("prompt"),
                             # snapshot: main() appends the assistant turn to this list AFTER run()
                             "messages": list(msgs) if msgs is not None else None,
                             "endpoint": endpoint, "stream": stream, "fmt": None})
        if raises is not None:
            raise raises
        r = response if response is not None else {"response": "ok", "done_reason": "stop"}
        answer = r.get("response") or (r.get("message") or {}).get("content") or ""
        return answer, r.get("done_reason")

    ask.ask, ask.run = fake_ask, fake_run
    try:
        yield
    finally:
        ask.ask, ask.run = original_ask, original_run


def run_main(argv):
    """Invoke ask.main() with the given args; return (stdout, stderr, exit_code)."""
    saved = sys.argv
    sys.argv = ["ask.py"] + argv
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                ask.main()
            except SystemExit as e:
                code = 0 if e.code is None else (e.code if isinstance(e.code, int) else 1)
    finally:
        sys.argv = saved
    return out.getvalue(), err.getvalue(), code


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
        self.assertEqual((gen["model"], gen["think"]), ("qwen-fast", False))

    def test_force_reason(self):
        gen, _ = route(["--reason", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("qwen-fast", True))

    def test_force_reason_hard_escalates_to_gpt_oss(self):
        # --reason-hard is the explicit escalation: the ONLY route to gpt-oss now that the default
        # reason route goes to qwen-fast (think on). Guards the escalation invariant.
        gen, _ = route(["--reason-hard", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("gpt-oss-fast", "high"))

    def test_force_quick(self):
        gen, _ = route(["--quick", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("gemma-fast", False))

    def test_competing_flags_use_argv_order(self):
        gen, _ = route(["--code", "--reason", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("qwen-fast", False))
        gen, _ = route(["--reason", "--code", "x"])
        self.assertEqual((gen["model"], gen["think"]), ("qwen-fast", True))

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
        for label, model in [("reason", "qwen-fast"), ("quick", "gemma-fast")]:
            with self.subTest(label=label):
                calls = []
                with mock_ask(recorder=calls, response=enum_response(label)):
                    run_main(["x"])
                self.assertGreaterEqual(len(calls), 2, "generation call is missing")
                self.assertIsNotNone(calls[0]["fmt"], "first call should be the constrained classify")
                self.assertIsNone(calls[-1]["fmt"], "last call should be generation, not the classifier")
                self.assertEqual(calls[-1]["model"], model)

    def test_classifier_code_forces_think_off(self):
        # Decision B: a classifier-selected 'code' route must reach qwen-fast with thinking OFF,
        # not just the explicit --code flag. Guards the invariant against a ROUTES regression.
        calls = []
        with mock_ask(recorder=calls, response=enum_response("code")):
            run_main(["x"])
        self.assertEqual(calls[-1]["model"], "qwen-fast")
        self.assertIs(calls[-1]["think"], False, "code route must force thinking off")

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


class TestErrorReporting(unittest.TestCase):
    """A failed call must explain itself, not crash or hand back a silent empty result.
    All offline - the network is mocked - so these are deterministic and fast."""

    def test_thinking_overflow_is_reported_not_silent(self):
        # A thinking route (--reason -> qwen-fast think-on, or gpt-oss via --reason-hard) can burn the
        # whole budget on thinking and emit no answer (done_reason=length, empty response). The router
        # must say so and signal failure, not print a blank line.
        with mock_ask(response={"response": "", "done_reason": "length"}):
            out, err, code = run_main(["--reason", "a hard logic puzzle"])
        self.assertEqual(out.strip(), "")
        self.assertIn("overflow", err.lower())
        self.assertNotEqual(code, 0)

    def test_truncated_but_nonempty_answer_is_kept_with_note(self):
        with mock_ask(response={"response": "partial answer", "done_reason": "length"}):
            out, err, code = run_main(["--code", "x"])
        self.assertIn("partial answer", out)
        self.assertIn("truncated", err.lower())
        self.assertEqual(code, 0)

    def test_missing_model_is_distinct_from_server_down(self):
        err404 = urllib.error.HTTPError(ask.HOST, 404, "not found", {}, None)
        with mock_ask(raises=err404):
            out, err, code = run_main(["--code", "x"])
        low = err.lower()
        self.assertIn("model", low)
        self.assertNotIn("can't reach", low)  # 404 != server down - must not mislabel
        self.assertNotEqual(code, 0)

    def test_server_unreachable_is_reported(self):
        with mock_ask(raises=urllib.error.URLError("connection refused")):
            out, err, code = run_main(["--code", "x"])
        self.assertIn("reach", err.lower())
        self.assertNotEqual(code, 0)

    def test_malformed_body_is_handled_cleanly(self):
        with mock_ask(raises=json.JSONDecodeError("expecting value", "doc", 0)):
            out, err, code = run_main(["--code", "x"])
        self.assertNotEqual(code, 0)
        self.assertIn("bad response", err.lower())


class TestContextSessionBudget(unittest.TestCase):
    """The multi-turn/context features: --file context, --num-predict, and --session shared history."""

    def test_file_context_is_prepended_to_the_prompt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("def buggy():\n    return 1 / 0\n")
            path = f.name
        try:
            gen, _ = route(["--code", "--file", path, "why does this crash?"])
        finally:
            os.unlink(path)
        self.assertEqual(gen["endpoint"], "generate")
        self.assertIn("def buggy", gen["prompt"], "file content must reach the model")
        self.assertIn("why does this crash?", gen["prompt"], "the task must too")

    def test_num_predict_overrides_the_route_budget(self):
        gen, _ = route(["--code", "--num-predict", "9000", "x"])
        self.assertEqual(gen["num_predict"], 9000, "explicit budget must win over the ROUTES default")

    def test_num_predict_rejects_non_positive_int(self):
        _, err, code = run_main(["--code", "--num-predict", "-5", "x"])
        self.assertNotEqual(code, 0)
        self.assertIn("num-predict", err.lower())

    def _in_temp_sessions(self):
        d = tempfile.mkdtemp()
        self._saved_dir, self._saved_db = ask.SESSION_DIR, ask.SESSION_DB
        ask.SESSION_DIR = d
        ask.SESSION_DB = os.path.join(d, "sessions.db")
        self.addCleanup(lambda: setattr(ask, "SESSION_DIR", self._saved_dir))
        self.addCleanup(lambda: setattr(ask, "SESSION_DB", self._saved_db))
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        return d

    def test_session_uses_chat_and_persists_the_turn(self):
        self._in_temp_sessions()
        gen, _ = route(["--session", "t", "--code", "write is_prime"])
        self.assertEqual(gen["endpoint"], "chat", "a session must use /api/chat, not /api/generate")
        self.assertEqual(gen["model"], "qwen-fast")
        self.assertEqual(gen["messages"][-1], {"role": "user", "content": "write is_prime"})
        stored = ask.load_session("t")
        self.assertEqual(stored[0]["role"], "user")
        self.assertEqual(stored[-1]["role"], "assistant", "the answer must be persisted")

    def test_session_second_turn_sees_prior_history(self):
        self._in_temp_sessions()
        route(["--session", "t", "--code", "write is_prime"])
        gen2, _ = route(["--session", "t", "--code", "now add a docstring"])
        self.assertGreaterEqual(len(gen2["messages"]), 3, "turn 2 must include prior user+assistant")
        self.assertEqual(gen2["messages"][-1]["content"], "now add a docstring")

    def test_explicit_flag_overrides_sticky_per_turn(self):
        # Sessions are sticky by default, but an explicit flag is the per-turn escape hatch: turn 1
        # code (-> qwen), turn 2 forced --reason-hard (-> gpt-oss), over the same shared history.
        self._in_temp_sessions()
        g1, _ = route(["--session", "t", "--code", "write a sort"])
        g2, _ = route(["--session", "t", "--reason-hard", "prove it terminates"])
        self.assertEqual((g1["model"], g1["think"]), ("qwen-fast", False))
        self.assertEqual((g2["model"], g2["think"]), ("gpt-oss-fast", "high"))
        self.assertIn("write a sort", [m["content"] for m in g2["messages"]])  # shared history

    def test_session_is_sticky_after_first_turn(self):
        # Sticky routing: the session locks onto its first turn's model and stays there. A trivia
        # follow-up that the classifier WOULD send to gemma keeps running on qwen - no model bounce,
        # warm cache. And the classifier must not even run on the sticky turn.
        self._in_temp_sessions()
        g1, _ = route(["--session", "t", "write is_prime"], response=enum_response("code"))
        self.assertEqual(g1["model"], "qwen-fast")
        g2, calls2 = route(["--session", "t", "what is its time complexity?"],
                           response=enum_response("quick"))
        self.assertEqual(g2["model"], "qwen-fast", "sticky: turn 2 stays on the session's model")
        self.assertEqual(len(calls2), 1, "sticky turn must skip classification (generation call only)")

    def test_session_locks_route_on_first_turn(self):
        self._in_temp_sessions()
        route(["--session", "t", "solve this step by step"], response=enum_response("reason"))
        self.assertEqual(ask.load_route("t"), "reason", "first turn must lock the session's route")

    def test_oneoff_flag_does_not_move_the_sticky_lock(self):
        # An explicit flag routes THAT turn but must not re-lock the session: after a one-off
        # --reason-hard, a later no-flag turn returns to the originally locked model, not gpt-oss.
        self._in_temp_sessions()
        route(["--session", "t", "write code"], response=enum_response("code"))      # lock = code
        g2, _ = route(["--session", "t", "--reason-hard", "one hard question"])       # one-off gpt-oss
        self.assertEqual(g2["model"], "gpt-oss-fast")
        g3, _ = route(["--session", "t", "keep going"], response=enum_response("quick"))
        self.assertEqual(g3["model"], "qwen-fast", "one-off flag must not move the sticky lock")

    def test_session_windows_the_payload_but_keeps_full_history(self):
        self._in_temp_sessions()
        cid = ask.create_conversation(name="t")
        ask.append_turns(cid, [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
                               for i in range(ask.MAX_MESSAGES + 6)])
        gen, _ = route(["--session", "t", "--code", "next"])
        self.assertLessEqual(len(gen["messages"]), ask.MAX_MESSAGES, "payload windowed to fit num_ctx")
        self.assertEqual(gen["messages"][-1]["content"], "next")
        self.assertGreater(len(ask.load_session("t")), ask.MAX_MESSAGES,
                           "the full history stays in the DB even though the sent payload is windowed")

    def test_continue_resumes_the_latest_conversation(self):
        self._in_temp_sessions()
        route(["--session", "a", "--code", "first"])
        route(["--session", "b", "--code", "second"])          # b is now the most recent
        gen, _ = route(["--continue", "--code", "more"])
        contents = [m["content"] for m in gen["messages"]]
        self.assertIn("second", contents, "--continue must resume the latest thread (b)")
        self.assertNotIn("first", contents, "not the older thread (a)")
        self.assertEqual(gen["messages"][-1]["content"], "more")

    def test_cid_resumes_a_specific_conversation(self):
        self._in_temp_sessions()
        route(["--session", "a", "--code", "alpha"])
        cid = ask.resolve_conversation(name="a")
        route(["--session", "b", "--code", "beta"])            # a different thread is now latest
        gen, _ = route(["--cid", str(cid), "--code", "back to a"])
        self.assertEqual(gen["endpoint"], "chat")
        contents = [m["content"] for m in gen["messages"]]
        self.assertIn("alpha", contents, "--cid must target that exact conversation")
        self.assertNotIn("beta", contents)


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
        resp = ask.ask("qwen-fast", "Write a one-line Python is_prime function.", False, 512)
        text = resp.get("response", "")
        self.assertTrue(text.strip(), "qwen-fast returned an empty answer")
        self.assertNotEqual(resp.get("done_reason"), "length",
                            "answer was truncated at num_predict - the budget is too small")
        self.assertIn("prime", text.lower(), "qwen-fast did not return an is_prime function")


if __name__ == "__main__":
    unittest.main(verbosity=2)
