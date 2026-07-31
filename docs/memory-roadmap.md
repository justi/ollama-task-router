# Memory roadmap: from JSON sessions to a real local memory layer

Status: proposal / backlog. Scope here is **steps 1-2 only** - the cheap, high-certainty wins.
Retrieval and curated/consolidating memory are deliberately deferred (see "Later"). Everything in
1-2 stays stdlib-only and fully offline, in keeping with the repo's zero-dependency ethos.

## Why

Sessions today are JSON files under `~/.ask-sessions/<name>.json` (`load_session`, `save_session`,
`load_route` in `ask.py`), and long histories are bounded by `trim()` - keep the last
`MAX_MESSAGES`, drop the rest. Two documented problems with that bounding:

- **Lost in the Middle** (Liu et al., 2023, arXiv 2307.03172): models under-use the *middle* of a
  long context, so stuffing the window is not free.
- **StreamingLLM** (Xiao et al., 2023, arXiv 2309.17453): a pure recency window is unstable and
  genuinely forgets whatever falls out of it - it is streaming, not memory.

The consensus replacement is a hybrid: a **rolling summary** of old turns + a **verbatim recency
window** (+ retrieval later). This roadmap does the two lowest-effort pieces.

## Step 1 - SQLite conversation log (replace the JSON files)

Replace per-session JSON with one SQLite DB (`~/.ask-sessions/sessions.db`), stdlib `sqlite3`, WAL
mode so concurrent CLI invocations don't race. Proven pattern: Simon Willison's `llm` tool (which
already runs against Ollama).

```sql
CREATE TABLE conversations (
  id         INTEGER PRIMARY KEY,
  name       TEXT UNIQUE,       -- --session <name>; NULL for --continue-only threads
  route      TEXT,              -- sticky lock (today's load_route / save_session route)
  summary    TEXT DEFAULT '',   -- rolling summary (step 2)
  created_at TEXT, updated_at TEXT
);
CREATE TABLE messages (
  id              INTEGER PRIMARY KEY,
  conversation_id INTEGER REFERENCES conversations(id),
  role            TEXT,          -- user | assistant | system
  content         TEXT,
  created_at      TEXT
);
```

Keep function signatures close to today's to minimize churn:

- `load_session(name)` -> messages for the named conversation (or `[]`).
- `save_session(name, messages, route)` -> upsert conversation, append the new turn(s), set
  route/updated_at - in one transaction (drop the tmp+rename dance; SQLite is ACID).
- `load_route(name)` -> `conversations.route`.
- New: `--continue` = most-recent `updated_at`; `--cid <id>` = a specific thread; `--sessions` =
  list `(id, name, route, #turns, updated_at)`.

Tasks:

- **1.1** DB-on-first-use helper: create schema, `PRAGMA journal_mode=WAL`.
- **1.2** Port `load_session` / `save_session` / `load_route` to SQLite (one transaction per turn).
- **1.3** `--cid <id>` + `--sessions` list; wire `--continue` to the latest thread.
- **1.4** (optional, low priority) one-time import of existing `~/.ask-sessions/*.json`. Default is
  **no migration**: sessions are ephemeral working state, so the switch simply abandons old JSON
  threads (the files stay on disk, unused). Add the importer only if someone misses a live thread.

Tests (`test_ask.py`; patch a temp DB path the way `_in_temp_sessions` patches `SESSION_DIR` today):

- create + resume by name; a turn persists content + route; `--cid` targets a specific thread;
  `--continue` picks the latest; two concurrent writers don't corrupt (WAL); the sticky route
  survives a reload.

Risk: none material - SQLite is stdlib, ACID replaces the manual atomic write, and it is a strict
upgrade over flat JSON (queryable, multi-thread, no per-file races).

## Step 2 - Hybrid context bounding (retire pure `trim()`)

Replace "keep last N messages" with "keep last K turns verbatim + a rolling summary of everything
older", injected as a leading `system` message.

Per turn:

1. Load `conversation.summary` + all messages.
2. Keep the last **K** turns verbatim (recency window).
3. If there are turns older than the window not yet folded in, do it with **one** summarizer call:
   `summary' = summarize(summary + evicted_turns)` - rolling and incremental, never re-summarize the
   whole history. Persist `summary'`.
4. Build the `/api/chat` payload as
   `[system: "Earlier in this conversation: <summary>"] + last-K-verbatim + new user turn`.

Config (constants first, flags later):

- `WINDOW_TURNS` (verbatim recency, e.g. 8) and/or an approximate char budget (no tokenizer without a
  dep, so approximate by chars and note the imprecision).
- Summarizer model: **the session's own (sticky-locked) model**, not a fixed one. It is already
  resident, so the summary call adds no model load and shares the warm cache; and it is as faithful
  as the thread itself - a code session is summarized by qwen, a trivia session by gemma. Run it with
  thinking OFF. (This is exactly the swap-tax lesson applied: don't switch models just to summarize.)
- Summarize prompt: preserve decisions, definitions, named identifiers, and open questions; drop
  chit-chat (per recursive-summary / RecurrentGPT).

Tasks:

- **2.1** Replace `trim()` with window + evict selection - deterministic, testable without the LLM.
- **2.2** Rolling-summary update on eviction (one `run`/`ask` call); persist to `conversations.summary`.
- **2.3** Inject the summary as a leading `system` message + the window into the chat payload.
- **2.4** Constants -> optional flags (`--window`, summarizer model).

Tests:

- eviction math is deterministic (which turns get summarized at which sizes) - no LLM;
- the summarizer call is mocked; payload shape = `[system summary] + window + user`;
- a long thread still fits (summary + window) and never sends the full history;
- the route / sticky lock is unaffected.

Risks / decisions:

- A summary can lose information -> keep a generous window, a conservative prompt, and (step 3)
  retrieval as the safety net for anything the summary dropped.
- Extra latency: one summarizer call **only on eviction** (amortized), not every turn.
- Char-based budget approximates tokens - fine for bounding; note it.

## Later (explicitly out of scope here)

- **Step 3 - retrieval-as-memory:** embed each turn (`embeddinggemma` / `nomic-embed-text` via
  `POST /api/embed`) into a `sqlite-vec` `vec0` table in the *same* DB; inject top-k semantically
  relevant old turns alongside the window. This is what recalls something from 200 turns ago;
  `sqlite-vec` keeps it single-file and pip-only.
- **Step 4+ - scored / self-consolidating memory:** rank by recency x importance x relevance
  (Generative Agents), consolidate on write (ADD/UPDATE/DELETE, Mem0), decay-based forgetting; or
  adopt a local library (Mem0 / Cognee / A-MEM) at the cost of the zero-dependency ethos.
- **Step 6 (orthogonal) - KV-cache latency:** Ollama already reuses an identical in-VRAM prefix
  (keep `keep_alive` long, stable prefix first); disk KV persistence exists only in llama.cpp. A
  latency lever, not a memory one.

References: Lost in the Middle (arXiv 2307.03172); StreamingLLM (arXiv 2309.17453); Generative Agents
memory stream (arXiv 2304.03442); Mem0 (arXiv 2504.19413); sqlite-vec (github.com/asg017/sqlite-vec);
`llm` conversation log (llm.datasette.io). Operational-value framing: the blog-series post on this
router's memory layer.
