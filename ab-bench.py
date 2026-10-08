#!/usr/bin/env python3
"""ab-bench: prefill + decode sweep for one engine or a paired A/B of two configs.

Protocol 3 (2026-10-08). Base protocol from sparkDash DecodeBench.js / PrefillBench.js / llmPrompts.js:
  decode : lab prompts per type (structured = counting, prose = data-structure explanation,
           code = a Python task, json = a fake-metrics catalog), max_tokens fixed with
           min_tokens/ignore_eos, one wave per level; per-stream decode tok/s =
           (completion_tokens - 1) / (last - first token); aggregate = decode tokens /
           (last token - first token) across the wave.
  prefill: one request per context size, fresh salt, "Reply with the single word OK",
           max_tokens 8, thinking off; client rate = prompt_tokens / TTFT, server rate =
           prompt_tokens / the engine's own prefill_s (TensorFold), so HTTP, scheduling and
           first-decode-step overhead no longer read as slow prefill on short prompts.

Profiles (--profile):
  greedy  (default) temperature 0, top_p 1, thinking off, 400 tokens: the archive's protocol.
  sampled production-like: temperature 1.0, top_p 0.95, top_k 20, thinking on, 2000 tokens,
          fixed per-stream/per-round seeds so both arms of a pair draw the same samples.
Every stream gets its own prompt (distinct prefixes and outputs; stream 0 keeps the sparkDash
prompt). --sparkdash-prompts restores one shared prompt per type (protocol 2, runs <= 26).

Per cell: --warm discarded cold rounds, a --drain pause, then --reps timed rounds. Reported:
median (and mean, min, max, spread = (max - min) / median). A cell whose spread exceeds
--spread-max % is re-measured with another --reps rounds, up to --reruns times; the median
covers every timed round. Also per cell: MTP acceptance (accepted / drafted from the engine's
reply stats), streams that stopped short of max_tokens, and rounds that overlapped foreign
requests (TensorFold /health requests_total moved by more than this tool sent).

Guards: refuses to start, or to run a block, while the server reports running requests
(--force overrides). Engine detail stores the container's image, serve command, TENSORFOLD_*
environment and cpuset (read over ssh, read-only).

Paired A/B (--pair LABEL=COMMAND, given twice): COMMAND switches the server to that arm (any
shell command, run on this host; it must leave the new server starting on the same URL).
Blocks (prefill, then one block per decode type) alternate A B | B A | A B ..., so each arm sees
early and late slots equally and there is one switch per block. Each arm gets its own run row,
sharing a pair_id; the end prints per-cell medians and B-vs-A deltas.

Examples:
  python3 ab-bench.py                                        # full greedy sweep, live Spark
  python3 ab-bench.py --profile sampled --phases decode      # production-like decode
  python3 ab-bench.py --phases prefill --prefill-long        # adds 64K and 128K
  python3 ab-bench.py --pair "old=ssh spark 'cd OLD && ./stop.sh && ./start.sh'" \\
                      --pair "new=ssh spark 'cd NEW && ./stop.sh && ./start.sh'"
  python3 ab-bench.py --url http://host:8090/v1 --model m --streams 1 --levels 1-4 --plain
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

DEFAULT_URL = "http://10.10.2.196:8888/v1"
CTX_TO_STREAMS = {262144: 5, 174766: 7}  # spark-tf-profile.sh profiles
PROTOCOL = 3

# --- prompts: stream 0 is verbatim from sparkDash src/shared/llmPrompts.js ---------------
STRUCTURED = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
PROSE = ("Write a detailed step-by-step explanation of how a hash map works, "
         "including collision handling, resizing, and time complexity. Be thorough.")
JSON_PROMPT = ("Emit only a JSON array of fake GPU metrics rows. Each object needs host, gpuIndex, "
               "utilPct, tempC, powerW, memUsedMb. Invent many rows. No markdown. Keep expanding the array.")
PROSE_TOPICS = [  # (structure, aspects): stream i explains topic i
    ("a hash map", "collision handling, resizing"),
    ("a B-tree", "node splits, merges on delete"),
    ("a skip list", "level promotion, search paths"),
    ("a trie", "prefix lookup, memory layout"),
    ("a binary heap", "sift up and down, heapify"),
    ("a bloom filter", "hash functions, false positive rates"),
    ("a red-black tree", "rotations, recoloring"),
    ("a ring buffer", "wraparound, full and empty detection"),
]
JSON_DOMAINS = [  # (rows, fields): stream i emits catalog i
    ("GPU metrics", "host, gpuIndex, utilPct, tempC, powerW, memUsedMb"),
    ("disk metrics", "host, device, readMBps, writeMBps, iops, usedPct"),
    ("network interface metrics", "host, iface, rxMbps, txMbps, errors, drops"),
    ("container metrics", "host, container, cpuPct, memMb, restarts, uptimeS"),
    ("UPS metrics", "site, unit, loadPct, batteryPct, runtimeMin, inputV"),
    ("fan sensor metrics", "host, fanIndex, rpm, dutyPct, tempC, alarm"),
    ("database metrics", "host, db, qps, p99Ms, connections, cacheHitPct"),
    ("power meter metrics", "rack, pdu, outlet, watts, amps, volts"),
]
CODE_TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
             "Then add tests and the helpers this needs. Keep writing code.")
CODE_TASKS = [
    ("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
    ("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
    ("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
    ("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
    ("ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."),
    ("dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."),
    ("edit_distance", "def edit_distance(a, b) -> int: Levenshtein distance between two strings."),
    ("semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b."),
    ("url_parse", "def url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries."),
    ("json_pointer", "def json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing."),
    ("glob_match", "def glob_match(pattern, text) -> bool: * and ? wildcards, no character classes."),
    ("csv_parse", "def csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes."),
    ("rle", "def rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs."),
    ("top_k", "def top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap."),
    ("interval_merge", "def merge_intervals(spans) -> list: merge overlapping [start, end] pairs."),
    ("topo_sort", "def topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle."),
    ("bloom_filter", "class BloomFilter: add(item) and might_contain(item) with two hash functions over a bit array."),
    ("moving_average", "class MovingAverage: next(x) returns the mean of the last window values."),
    ("retry_backoff", "def backoff_delays(attempts, base_ms, cap_ms) -> list: exponential delays clipped at the cap."),
    ("base64_encode", "def b64_encode(data) -> str and b64_decode(text) -> bytes: standard base64, no libraries."),
    ("expr_eval", "def eval_expr(text) -> int: evaluate non-negative ints with + - * / and parentheses."),
    ("histogram_percentile", "def percentile(samples, p) -> float: nearest-rank percentile of a list of numbers."),
    ("redact_secrets", "def redact(text) -> str: replace AWS-looking keys and password= values with ***. Keep the rest."),
    ("chunk_text", "def chunk_text(text, size) -> list: split into chunks of at most size chars, breaking on spaces when possible."),
    ("route_match", "def route_match(pattern, path) -> dict or None: /users/:id style params."),
    ("crc32", "def crc32(data) -> int: IEEE CRC-32 of a bytes object."),
    ("fixed_window", "class FixedWindow: allow() is True up to limit events per window_s, else False."),
    ("diff_lines", "def diff_lines(a, b) -> list: line diff as equal/delete/insert ops using a simple LCS."),
    ("infix_postfix", "def infix_to_postfix(tokens) -> list: shunting-yard for + - * / and parentheses."),
    ("consistent_hash", "class ConsistentHash: add_node, remove_node, and get_node(key) on a ring of virtual nodes."),
    ("utf8_decode", "def utf8_decode(data) -> str: decode UTF-8 bytes, replacing invalid sequences with U+FFFD."),
    ("dependency_closure", "def closure(root, deps) -> list: packages reachable from root, each name once, in visit order."),
]
CODE_WARMUP = f"warmup_noop\ndef warmup_noop(x): return x unchanged.\n{CODE_TAIL}"

PROFILES = {
    "greedy": {"temperature": 0, "top_p": 1, "thinking": False, "gen": 400},
    "sampled": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "thinking": True, "gen": 2000},
}


def stream_prompt(ptype: str, i: int) -> str:
    """Stream i's prompt of this type; i = 0 is the sparkDash prompt."""
    if ptype == "structured":
        lo = 200 * i + 1
        return f"Count from {lo} to {lo + 199}. Output only the numbers, separated by spaces. No other text."
    if ptype == "prose":
        topic, aspects = PROSE_TOPICS[i % len(PROSE_TOPICS)]
        return (f"Write a detailed step-by-step explanation of how {topic} works, "
                f"including {aspects}, and time complexity. Be thorough.")
    if ptype == "json":
        rows, fields = JSON_DOMAINS[i % len(JSON_DOMAINS)]
        return (f"Emit only a JSON array of fake {rows} rows. Each object needs {fields}. "
                f"Invent many rows. No markdown. Keep expanding the array.")
    if ptype == "code":
        name, spec = CODE_TASKS[i % len(CODE_TASKS)]
        return f"{name}\n{spec}\n{CODE_TAIL}"
    raise SystemExit(f"unknown prompt type {ptype!r}")


def decode_prompts(ptype: str, n: int, shared: bool) -> list[str]:
    """One prompt per stream. shared=True: protocol 2 (one prompt for every stream, except code)."""
    if shared and ptype != "code":
        return [{"structured": STRUCTURED, "prose": PROSE, "json": JSON_PROMPT}[ptype]] * n
    return [stream_prompt(ptype, i) for i in range(n)]


# --- prefill filler (sparkDash's prefill bench pads with filler; sized here by real tokens)
SUBJ = ("the archivist", "a courier", "the old engineer", "a quiet clerk", "the ferryman",
        "the night guard", "a travelling tinker", "the harbour master", "the baker")
VERB = ("carried", "repaired", "catalogued", "misplaced", "painted", "measured", "traded")
OBJ = ("a crate of lanterns", "the brass compass", "several faded maps", "a basket of pears",
       "the broken clock", "a bundle of letters", "the copper kettle", "an unmarked ledger")
PLACE = ("near the north gate", "behind the granary", "along the canal", "inside the workshop",
         "beside the chapel steps", "under the market awning", "at the orchard edge")


def filler(rng: random.Random, chars: int) -> str:
    parts, n = [], 0
    while n < chars:
        s = f"{rng.choice(SUBJ).title()} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(PLACE)}. "
        parts.append(s)
        n += len(s)
    return "".join(parts)


def prefill_prompt(rng: random.Random, chars: int) -> str:
    salt = f"{rng.randrange(1 << 48):012x}"
    return (f"[prefill-bench {salt}]\nIgnore the filler below. Reply with the single word OK.\n"
            + filler(rng, chars))


class Cfg:
    def __init__(self, args):
        self.base_url = args.url.rstrip("/")
        parsed = urllib.parse.urlparse(self.base_url)
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path.rstrip("/") + "/chat/completions"  # base usually ends in /v1
        self.model = args.model
        self.extra = json.loads(args.extra)
        self.plain = args.plain
        self.timeout = args.timeout


def parse_sizes(spec: str) -> list[int]:
    """'1k,2k,4096' -> token counts (k = 1024)."""
    out = []
    for s in spec.split(","):
        s = s.strip().lower()
        out.append(int(float(s[:-1]) * 1024) if s.endswith("k") else int(s))
    return out


def parse_levels(spec: str) -> list[int]:
    out = []
    for item in spec.split(","):
        if "-" in item:
            a, b = item.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(item))
    return out


def chat(cfg: Cfg, prompt: str, max_tokens: int, fill: bool, profile: dict | None = None,
         seed: int | None = None) -> dict:
    """One streaming request. fill=True forces exactly max_tokens (min_tokens + ignore_eos).
    profile=None: greedy with thinking off (prefill, warmups). Times are seconds since the
    request started; server_* come from the engine's own reply stats when it sends them."""
    prof = profile or PROFILES["greedy"]
    body = {"model": cfg.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": prof["temperature"], "top_p": prof["top_p"],
            "stream": True, "stream_options": {"include_usage": True}}
    if "top_k" in prof:
        body["top_k"] = prof["top_k"]
    if seed is not None:
        body["seed"] = seed
    if not cfg.plain:
        on = prof["thinking"]
        body["chat_template_kwargs"] = {"enable_thinking": on, "thinking": on,
                                        "thinking_mode": "enabled" if on else "disabled"}  # sparkDash applyThinkingFlags
        if fill:
            body.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    body.update(cfg.extra)
    out = {"ok": False, "ptok": None, "ctok": None, "first": None, "last": None, "error": None,
           "server_prefill_s": None, "accepted": None, "drafted": None, "token_sha": None}
    t0 = time.perf_counter()
    first = last = None
    text = ""
    usage: dict = {}
    stats: dict = {}
    try:
        conn = http.client.HTTPConnection(cfg.host, cfg.port, timeout=cfg.timeout)
        conn.request("POST", cfg.path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            payload = resp.read()
            conn.close()
            out["error"] = f"HTTP {resp.status}: {payload[:200]!r}"
            return out
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            if isinstance(ev.get("tensorfold"), dict):
                stats = ev["tensorfold"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if piece:
                    now = time.perf_counter() - t0
                    if first is None:
                        first = now
                    last = now
                    text += piece
        conn.close()
        if first is None:
            out["error"] = "stream without content"
            return out
        out.update(ok=True, first=first, last=last,
                   ptok=usage.get("prompt_tokens") or max(1, len(prompt) // 4),
                   ctok=usage.get("completion_tokens") or max(1, len(text) // 4),
                   server_prefill_s=stats.get("prefill_s"), accepted=stats.get("accepted"),
                   drafted=stats.get("drafted"), token_sha=stats.get("token_sha"))
        return out
    except Exception as exc:  # noqa: BLE001 — one round must not kill the sweep
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def wave(cfg: Cfg, prompts: list[str], max_tokens: int, profile: dict, seeds: list[int | None]) -> list[dict]:
    with ThreadPoolExecutor(len(prompts)) as pool:
        return list(pool.map(lambda ps: chat(cfg, ps[0], max_tokens, True, profile, ps[1]), zip(prompts, seeds)))


def decode_wave(cfg: Cfg, prompts: list[str], gen: int, profile: dict, seeds: list[int | None]) -> dict:
    """Aggregate tok/s = decode tokens / (last token - first token) across the wave (sparkDash
    arithmetic); per-stream = (completion_tokens - 1) / (last - first); plus acceptance and
    streams that stopped short of gen."""
    results = wave(cfg, prompts, gen, profile, seeds)
    ok = [r for r in results if r["ok"] and r["last"] > r["first"]]
    if not ok:
        return {"error": f"no usable streams, first error: {results[0]['error']}"}
    window = max(r["last"] for r in ok) - min(r["first"] for r in ok)
    acc = [r["accepted"] for r in ok if r["accepted"] is not None]
    dra = [r["drafted"] for r in ok if r["drafted"] is not None]
    return {"error": None if len(ok) == len(results) else f"{len(results) - len(ok)} stream(s) failed",
            "agg": sum(r["ctok"] - 1 for r in ok) / window,
            "per": sum((r["ctok"] - 1) / (r["last"] - r["first"]) for r in ok) / len(ok),
            "gauge": sum(r["last"] - r["first"] for r in ok) / window,
            "ttft": sum(r["first"] for r in ok) / len(ok),
            "accepted": sum(acc) if acc else None, "drafted": sum(dra) if dra else None,
            "ctok_min": min(r["ctok"] for r in ok),
            "short": sum(1 for r in ok if r["ctok"] < gen) + (len(results) - len(ok)),
            "shas": [r["token_sha"] if r["ok"] else None for r in results]}  # stream order


def calibrate(cfg: Cfg, rng: random.Random) -> float:
    """Characters per token of the prefill filler on this server, from one real request."""
    p = prefill_prompt(rng, 8000)
    r = chat(cfg, p, 1, False)
    if not r["ok"]:
        raise SystemExit(f"calibration request failed: {r['error']}")
    return len(p) / r["ptok"]


def probe(cfg: Cfg, path: str) -> str:
    try:
        root = cfg.base_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(root + path, timeout=10) as r:
            return r.read().decode()
    except Exception as exc:  # noqa: BLE001
        return f"<unreachable {exc}>"


DEFAULT_DB = Path.home() / "projects" / "llm-bench" / "results" / "ab-bench.db"
KNOWN_SSH = {"10.10.2.196": "spark"}  # url host -> ssh alias, used only to read the engine container's details
TOOL = f"ab-bench {PROTOCOL}"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY,
    started_at     TEXT NOT NULL,           -- UTC ISO-8601
    finished_at    TEXT,
    tool           TEXT,
    base_url       TEXT NOT NULL,
    engine         TEXT,                    -- tensorfold | ollama | llama.cpp | vllm | ...
    engine_version TEXT,                    -- e.g. v0.6.5-concurrent-vision (image tag), ollama version, llama.cpp build
    engine_detail  TEXT,                    -- JSON: container, image, serve command, TENSORFOLD_* env, cpuset, /health extract
    model          TEXT NOT NULL,
    max_streams    INTEGER,
    context_length INTEGER,
    thinking       TEXT,                    -- off | on | n/a
    config         TEXT,                    -- JSON: reps, warm, drain, gen, levels, types, prefill sizes, profile, ...
    notes          TEXT
);
CREATE TABLE IF NOT EXISTS samples (        -- one row per timed or discarded round
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    phase          TEXT NOT NULL,           -- decode | prefill
    ptype          TEXT,                    -- decode prompt type (NULL for prefill)
    concurrency    INTEGER NOT NULL,        -- streams in the wave (prefill: 1)
    context_tokens INTEGER,                 -- prefill context size target (NULL for decode)
    round          INTEGER NOT NULL,        -- 0 = cold warm-up round, 1..N timed
    discarded      INTEGER NOT NULL,        -- 1 for cold rounds
    tps            REAL,                    -- decode: aggregate tok/s; prefill: prompt tokens / TTFT
    per_stream_tps REAL,                    -- decode: mean per-stream tok/s
    streams_mean   REAL,                    -- decode dilution gauge (mean streams decoding in window)
    ttft_s         REAL,                    -- prefill: TTFT of the request; decode: mean first-token time of the wave's streams
    prompt_tokens  INTEGER,                 -- prefill: real prompt token count
    error          TEXT
);
CREATE INDEX IF NOT EXISTS samples_run ON samples(run_id, phase);
CREATE TABLE IF NOT EXISTS cell_stats (     -- protocol 3: one row per measured cell
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    phase          TEXT NOT NULL,
    ptype          TEXT,
    concurrency    INTEGER NOT NULL,
    context_tokens INTEGER,
    rounds         INTEGER,                 -- timed rounds behind the numbers (reruns included)
    median_tps     REAL,                    -- decode: aggregate; prefill: client rate (prompt / TTFT)
    mean_tps       REAL,
    min_tps        REAL,
    max_tps        REAL,
    spread_pct     REAL,                    -- (max - min) / median * 100
    flagged        INTEGER,                 -- spread still above --spread-max after the reruns
    reruns         INTEGER,                 -- extra --reps batches this cell needed
    median_ttft_s  REAL,
    median_server_tps REAL,                 -- prefill: prompt tokens / engine prefill_s
    accept_rate    REAL,                    -- decode: accepted / drafted over the timed rounds
    short_streams  INTEGER,                 -- decode: streams below max_tokens over the timed rounds
    foreign_rounds INTEGER,                 -- timed rounds that overlapped requests from other clients
    output_variants INTEGER                 -- decode: distinct per-stream output hash lists over the timed rounds
);
CREATE TABLE IF NOT EXISTS host_state (     -- engine host, read over ssh before and after each block
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    block          TEXT NOT NULL,           -- prefill | decode/<type>
    moment         TEXT NOT NULL,           -- before | after
    at             TEXT NOT NULL,           -- UTC ISO-8601
    gpu_temp_c     REAL,
    gpu_power_w    REAL,
    gpu_sm_mhz     REAL,
    throttle       TEXT,                    -- nvidia-smi clocks_throttle_reasons.active
    mem_avail_gib  REAL,
    load1          REAL
);
DROP VIEW IF EXISTS decode_cells;
CREATE VIEW decode_cells AS
    SELECT r.id AS run_id, r.started_at, r.engine, r.engine_version, r.model, s.ptype, s.concurrency,
           ROUND(AVG(s.tps), 2) AS mean_agg_tps, ROUND(AVG(s.per_stream_tps), 2) AS mean_per_stream_tps,
           ROUND(AVG(s.streams_mean), 2) AS mean_streams, ROUND(AVG(s.ttft_s), 3) AS mean_ttft_s, COUNT(*) AS rounds
    FROM samples s JOIN runs r ON r.id = s.run_id
    WHERE s.phase = 'decode' AND s.discarded = 0 AND s.error IS NULL
    GROUP BY s.run_id, s.ptype, s.concurrency;
CREATE VIEW IF NOT EXISTS prefill_cells AS
    SELECT r.id AS run_id, r.started_at, r.engine, r.engine_version, r.model, s.context_tokens,
           ROUND(AVG(s.prompt_tokens)) AS mean_prompt_tokens, ROUND(AVG(s.tps), 1) AS mean_prefill_tps,
           ROUND(AVG(s.ttft_s), 3) AS mean_ttft_s, COUNT(*) AS rounds
    FROM samples s JOIN runs r ON r.id = s.run_id
    WHERE s.phase = 'prefill' AND s.discarded = 0 AND s.error IS NULL
    GROUP BY s.run_id, s.context_tokens;
"""
# columns added by protocol 3 to tables created by earlier versions
MIGRATIONS = {
    "runs": [("pair_id", "TEXT"), ("arm", "TEXT"), ("profile", "TEXT"), ("protocol", "INTEGER")],
    "samples": [("server_prefill_s", "REAL"), ("accepted", "INTEGER"), ("drafted", "INTEGER"),
                ("completion_min", "INTEGER"), ("short_streams", "INTEGER"), ("foreign_requests", "INTEGER"),
                ("token_shas", "TEXT")],
    "cell_stats": [("output_variants", "INTEGER")],
}


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, typ in cols:
                if col not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        self.conn.commit()

    def start(self, started_at, base_url, engine, model, max_streams, context_length, thinking, config, notes,
              pair_id=None, arm=None, profile=None, protocol=PROTOCOL) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (started_at, tool, base_url, engine, engine_version, engine_detail, model, "
            "max_streams, context_length, thinking, config, notes, pair_id, arm, profile, protocol) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (started_at, TOOL, base_url, engine["name"], engine["version"], json.dumps(engine["detail"]),
             model, max_streams, context_length, thinking, json.dumps(config), notes, pair_id, arm, profile,
             protocol))
        self.conn.commit()
        return cur.lastrowid

    def sample(self, run_id, phase, ptype, concurrency, context_tokens, rnd, discarded, tps=None, per=None,
               streams=None, ttft=None, ptok=None, error=None, server_prefill_s=None, accepted=None,
               drafted=None, completion_min=None, short=None, foreign=None, shas=None):
        self.conn.execute(
            "INSERT INTO samples (run_id, phase, ptype, concurrency, context_tokens, round, discarded, tps, "
            "per_stream_tps, streams_mean, ttft_s, prompt_tokens, error, server_prefill_s, accepted, drafted, "
            "completion_min, short_streams, foreign_requests, token_shas) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, phase, ptype, concurrency, context_tokens, rnd, int(discarded), tps, per, streams,
             ttft, ptok, error, server_prefill_s, accepted, drafted, completion_min, short, foreign,
             json.dumps(shas) if shas is not None else None))
        self.conn.commit()  # commit per round: an interrupted sweep keeps what it measured

    def host(self, run_id, block, moment, snap: dict):
        self.conn.execute(
            "INSERT INTO host_state (run_id, block, moment, at, gpu_temp_c, gpu_power_w, gpu_sm_mhz, throttle, "
            "mem_avail_gib, load1) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, block, moment, now_utc(), snap.get("temp"), snap.get("power"), snap.get("sm"),
             snap.get("throttle"), snap.get("mem"), snap.get("load")))
        self.conn.commit()

    def cell(self, run_id, phase, ptype, concurrency, context_tokens, st: dict):
        self.conn.execute(
            "INSERT INTO cell_stats (run_id, phase, ptype, concurrency, context_tokens, rounds, median_tps, "
            "mean_tps, min_tps, max_tps, spread_pct, flagged, reruns, median_ttft_s, median_server_tps, "
            "accept_rate, short_streams, foreign_rounds, output_variants) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, phase, ptype, concurrency, context_tokens, st["n"], st["median"], st["mean"], st["min"],
             st["max"], st["spread"], int(st["flagged"]), st["reruns"], st.get("ttft"), st.get("server_tps"),
             st.get("accept"), st.get("short"), st.get("foreign"),
             len(st["shas"]) if st.get("shas") else None))
        self.conn.commit()

    def finish(self, run_id, finished_at):
        self.conn.execute("UPDATE runs SET finished_at=? WHERE id=?", (finished_at, run_id))
        self.conn.commit()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jget(cfg: Cfg, path: str):
    try:
        return json.loads(probe(cfg, path))
    except (ValueError, TypeError):
        return None


def ssh_run(ssh: str, command: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", ssh, command],
                          capture_output=True, text=True, timeout=30).stdout


def detect_engine(cfg: Cfg, args) -> dict:
    """Which inference engine answers on this URL, and which version. HTTP probes first
    (TensorFold /health 'backend', Ollama /api/version, llama.cpp /props, vLLM /version);
    TensorFold exposes no version over HTTP, so its container is read over ssh (read-only
    `docker ps` / `docker inspect`): image tag, serve command, TENSORFOLD_* env, cpuset."""
    name = version = None
    detail: dict = {}
    health = jget(cfg, "/health")
    if isinstance(health, dict):
        if health.get("backend"):
            name = str(health["backend"])
        detail["health"] = {k: health[k] for k in ("streams", "context_length") if k in health}
    ver = jget(cfg, "/api/version")
    props = jget(cfg, "/props")
    vv = jget(cfg, "/version")
    if isinstance(ver, dict) and ver.get("version"):
        name, version = "ollama", str(ver["version"])
    elif isinstance(props, dict) and props.get("build_info"):
        name, version = "llama.cpp", str(props["build_info"])
    elif isinstance(vv, dict) and vv.get("version"):
        name, version = name or "vllm", str(vv["version"])
    ssh = args.ssh or KNOWN_SSH.get(cfg.host)
    if ssh and version is None:
        try:
            rows = [line.split("|", 1) for line in ssh_run(ssh, "docker ps --format '{{.Names}}|{{.Image}}'")
                    .splitlines() if "|" in line]
            pick = next((r for r in rows if r[0] == args.container), None) if args.container else None
            if pick is None and name:
                pick = next((r for r in rows if name.lower() in r[1].lower() or name.lower() in r[0].lower()), None)
            if pick:
                raw = ssh_run(ssh, f"docker inspect {pick[0]} --format "
                                   "'{{json .Config.Image}}|{{json .State.StartedAt}}|{{json .Image}}|"
                                   "{{json .Config.Cmd}}|{{json .Config.Env}}|{{json .HostConfig.CpusetCpus}}'")
                parts = [json.loads(p) for p in raw.strip().split("|")] if raw.strip() else []
                image = parts[0] if parts and parts[0] else pick[1]
                tag = image.split(":", 1)[1] if ":" in image else image
                m = re.match(r"v?\d+(?:\.\d+)+", tag)  # v0.6.5-concurrent-vision -> v0.6.5 (full tag stays in engine_detail)
                version = m.group(0) if m else tag
                env = parts[4] if len(parts) > 4 and parts[4] else []
                detail["container"] = {"name": pick[0], "image": image,
                                       "started_at": parts[1] if len(parts) > 1 else None,
                                       "image_id": parts[2] if len(parts) > 2 else None, "ssh": ssh,
                                       "cmd": parts[3] if len(parts) > 3 else None,
                                       "env": sorted(e for e in env if e.startswith("TENSORFOLD_")),
                                       "cpuset": (parts[5] or None) if len(parts) > 5 else None}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            detail["container_probe_error"] = str(exc)
    if args.engine:
        name = args.engine
    pretty = {"tensorfold": "TensorFold", "ollama": "Ollama", "vllm": "vLLM", "llama.cpp": "llama.cpp"}
    name = name or "unknown"
    return {"name": pretty.get(name.lower(), name), "version": version or "unknown", "detail": detail}


def server_state(cfg: Cfg) -> tuple[int | None, int | None]:
    """(requests running, requests_total) from /health, None where the engine does not say."""
    h = jget(cfg, "/health")
    if not isinstance(h, dict):
        return None, None
    running = h.get("requests_running")
    if running is None and h.get("busy") is not None:
        running = 1 if h["busy"] else 0
    return running, h.get("requests_total")


def host_snapshot(ssh: str | None) -> dict:
    """GPU temperature / power / SM clock / throttle reasons, MemAvailable and load average of the
    engine host, read-only over ssh; {} when there is no ssh alias or the read fails."""
    if not ssh:
        return {}
    try:
        out = ssh_run(ssh, "nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,"
                           "clocks_throttle_reasons.active --format=csv,noheader,nounits; "
                           "grep MemAvailable /proc/meminfo; cat /proc/loadavg").splitlines()
    except (OSError, subprocess.SubprocessError):
        return {}
    snap: dict = {}

    def num(text):
        try:
            return float(text)
        except ValueError:
            return None
    if out and "," in out[0]:
        f = [x.strip() for x in out[0].split(",")]
        snap.update(temp=num(f[0]), power=num(f[1]), sm=num(f[2]), throttle=f[3] if len(f) > 3 else None)
    for line in out[1:]:
        if line.startswith("MemAvailable"):
            snap["mem"] = round(int(line.split()[1]) / 1048576, 2)
        elif line and line[0].isdigit():
            snap["load"] = num(line.split()[0])
    return snap


def require_idle(cfg: Cfg, force: bool, where: str) -> None:
    running, _ = server_state(cfg)
    if running:
        msg = f"server reports {running} running request(s) {where}"
        if not force:
            raise SystemExit(msg + "; other traffic skews every number. Wait, or pass --force.")
        print(f"WARNING: {msg} (--force)")


def cell_stats(values: list[float], spread_max: float) -> dict:
    med = statistics.median(values)
    spread = (max(values) - min(values)) / med * 100 if med else 0.0
    return {"n": len(values), "median": med, "mean": statistics.fmean(values), "min": min(values),
            "max": max(values), "spread": spread, "flagged": len(values) > 1 and spread > spread_max}


FOREIGN = "foreign"  # measure() result for a round that overlapped other clients' requests


def run_cell(args, label: str, measure) -> dict | None:
    """--warm discarded rounds, then --reps timed rounds; a cell whose spread exceeds
    --spread-max gets up to --reruns more batches. measure(rnd, discarded, slot) records the
    round itself and returns (value, extras), None on failure, or FOREIGN when other clients'
    requests overlapped it: such a round is left out and replaced (up to --reps extra attempts
    a batch). slot numbers the clean timed rounds, so sampled seeds line up across arms."""
    values, extras, rnd, first, foreign = [], [], 0, True, 0
    for _ in range(args.warm):
        if not first:
            time.sleep(args.drain)
        first = False
        measure(0, True, 0)
    reruns = 0
    while True:
        target, attempts = len(values) + args.reps, 0
        while len(values) < target and attempts < 2 * args.reps:
            if not first:
                time.sleep(args.drain)
            first = False
            rnd += 1
            attempts += 1
            got = measure(rnd, False, len(values) + 1)
            if got == FOREIGN:
                foreign += 1
            elif got is not None:
                values.append(got[0])
                extras.append(got[1])
        if not values:
            print(f"  {label}: no usable rounds")
            return None
        st = cell_stats(values, args.spread_max)
        if not st["flagged"] or reruns >= args.reruns:
            break
        reruns += 1
        print(f"  {label}: spread {st['spread']:.1f}% > {args.spread_max}%, re-measuring ({reruns}/{args.reruns})")
    st["reruns"] = reruns
    st["extras"] = extras
    st["foreign"] = foreign
    return st


class Arm:
    def __init__(self, label: str, command: str | None):
        self.label, self.command = label, command
        self.run_id = None
        self.engine = None
        self.cells: dict = {}


def switch_to(arm: Arm, cfg: Cfg, args) -> None:
    print(f"\n### switching to arm {arm.label}: {arm.command}")
    rc = subprocess.run(arm.command, shell=True).returncode
    if rc != 0:
        raise SystemExit(f"switch command for {arm.label} exited {rc}")
    deadline = time.time() + args.switch_timeout
    while time.time() < deadline:
        h = jget(cfg, "/health")
        models = jget(cfg, "/v1/models")
        if isinstance(h, dict) and h.get("ok", True) and isinstance(models, dict) and models.get("data"):
            break
        time.sleep(5)
    else:
        raise SystemExit(f"arm {arm.label}: server not ready after {args.switch_timeout:.0f}s")
    time.sleep(args.drain)
    print(f"### arm {arm.label} ready")


def main() -> None:
    ap = argparse.ArgumentParser(prog="ab-bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=None, help="model id; default: first id from /v1/models")
    ap.add_argument("--phases", default="prefill,decode")
    ap.add_argument("--types", default="structured,prose,code,json", help="decode prompt types")
    ap.add_argument("--levels", default=None, help="decode concurrency, e.g. 1-5; default 1..max streams")
    ap.add_argument("--streams", type=int, default=None, help="max concurrent streams (else read from /health)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="greedy",
                    help="decode sampling: greedy (archive protocol) or sampled (production-like)")
    ap.add_argument("--sparkdash-prompts", action="store_true",
                    help="protocol 2 prompts: one shared prompt per type on every stream (runs <= 26)")
    ap.add_argument("--reps", type=int, default=3, help="timed rounds per cell (per batch)")
    ap.add_argument("--warm", type=int, default=1, help="discarded cold rounds per cell")
    ap.add_argument("--drain", type=float, default=5.0, help="seconds of queue drain between rounds")
    ap.add_argument("--spread-max", type=float, default=5.0,
                    help="re-measure a cell whose (max - min) / median exceeds this percentage")
    ap.add_argument("--reruns", type=int, default=1, help="extra --reps batches allowed per noisy cell")
    ap.add_argument("--gen", type=int, default=None, help="decode max tokens per stream (profile default)")
    ap.add_argument("--prefill-ctx", default="1k,2k,4k,8k,16k,32k",
                    help="prefill context sizes in real prompt tokens (k = 1024)")
    ap.add_argument("--prefill-long", action="store_true", help="also prefill 64K and 128K")
    ap.add_argument("--prefill-gen", type=int, default=8, help="tokens generated in prefill requests")
    ap.add_argument("--extra", default="{}", help="extra JSON merged into every request body")
    ap.add_argument("--plain", action="store_true",
                    help="omit vLLM/Qwen-only params (thinking flags, min_tokens, ignore_eos, stop)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="run even while the server reports other requests")
    ap.add_argument("--pair", action="append", default=[], metavar="LABEL=COMMAND",
                    help="paired A/B: give twice; COMMAND switches the server to that arm")
    ap.add_argument("--switch-timeout", type=float, default=900.0, help="seconds to wait for a switched server")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite results database (created if missing)")
    ap.add_argument("--no-db", action="store_true", help="print only, record nothing")
    ap.add_argument("--notes", default=None, help="free text stored with the run (patches applied, etc.)")
    ap.add_argument("--engine", default=None, help="override detected engine name")
    ap.add_argument("--engine-version", action="append", default=[], metavar="[LABEL=]VERSION",
                    help="override detected engine version; with --pair give LABEL=VERSION per arm")
    ap.add_argument("--ssh", default=None, help="ssh alias of the engine host (to read its container details)")
    ap.add_argument("--container", default=None, help="engine container name on that host (default: auto)")
    ap.add_argument("--ingest-log", default=None, metavar="FILE",
                    help="load a finished protocol-2 console log into the database and exit")
    ap.add_argument("--started-at", default=None, help="with --ingest-log: run start time, UTC ISO-8601")
    args = ap.parse_args()

    cfg = Cfg(args)
    if args.ingest_log:
        ingest_log(args.ingest_log, cfg, args, Store(args.db))
        return
    if args.pair and len(args.pair) != 2:
        raise SystemExit("--pair takes exactly two LABEL=COMMAND arms")
    arms = []
    for spec in args.pair:
        label, sep, command = spec.partition("=")
        if not sep or not label or not command:
            raise SystemExit(f"--pair {spec!r}: expected LABEL=COMMAND")
        arms.append(Arm(label.strip(), command))
    if arms and arms[0].label == arms[1].label:
        raise SystemExit("--pair labels must differ")
    if not arms:
        arms = [Arm("", None)]
    versions = {}
    for spec in args.engine_version:
        label, sep, ver = spec.partition("=")
        versions[label if sep else ""] = ver if sep else spec
    profile = dict(PROFILES[args.profile])
    gen = args.gen or profile["gen"]
    prefill_sizes = parse_sizes(args.prefill_ctx) + ([64 * 1024, 128 * 1024] if args.prefill_long else [])
    phases = [p.strip() for p in args.phases.split(",")]
    types = [t.strip() for t in args.types.split(",")]
    blocks = ([("prefill", None)] if "prefill" in phases else []) + \
             ([("decode", t) for t in types] if "decode" in phases else [])
    nonce = uuid.uuid4().hex[:12]
    pair_id = nonce if args.pair else None
    store = None if args.no_db else Store(args.db)
    config = {"reps": args.reps, "warm": args.warm, "drain_s": args.drain, "gen": gen, "types": args.types,
              "prefill_ctx": ",".join(f"{s // 1024}k" for s in prefill_sizes), "prefill_gen": args.prefill_gen,
              "phases": args.phases, "seed": args.seed, "plain": args.plain, "profile": args.profile,
              "distinct_prompts": not args.sparkdash_prompts, "spread_max": args.spread_max,
              "reruns": args.reruns, "protocol": PROTOCOL}
    if pair_id:
        config["pair"] = {a.label: a.command for a in arms}
    thinking = "n/a" if args.plain else ("on" if profile["thinking"] else "off")
    if not arms[0].command:
        require_idle(cfg, args.force, "before the sweep")

    def open_arm(arm: Arm) -> None:
        """First time an arm is live: resolve model, engine and levels; open its run row."""
        if not cfg.model:
            models = jget(cfg, "/v1/models")
            try:
                cfg.model = models["data"][0]["id"]
            except Exception:  # noqa: BLE001
                raise SystemExit(f"cannot auto-detect model from {cfg.base_url}/v1/models; pass --model")
        arm.engine = detect_engine(cfg, args)
        ver = versions.get(arm.label) or versions.get("")
        if ver:
            arm.engine["version"] = ver
        health = arm.engine["detail"].get("health", {})
        streams = args.streams or (health.get("streams") or {}).get("max") \
            or CTX_TO_STREAMS.get(health.get("context_length"))
        if "decode" in phases and not args.levels and not streams:
            raise SystemExit("cannot tell the engine's max concurrent streams (no /health 'streams.max'); "
                             "pass --streams N or --levels")
        arm.levels = parse_levels(args.levels) if args.levels else list(range(1, (streams or 1) + 1))
        arm.streams = streams
        tag = f"arm {arm.label}: " if arm.label else ""
        print(f"{tag}target {cfg.base_url} model {cfg.model}; streams={streams} decode levels={arm.levels} "
              f"reps={args.reps} warm={args.warm} drain={args.drain}s gen={gen} profile={args.profile} "
              f"thinking={thinking} distinct_prompts={not args.sparkdash_prompts}")
        c = arm.engine["detail"].get("container") or {}
        print(f"{tag}engine {arm.engine['name']} {arm.engine['version']} image={c.get('image')} "
              f"cpuset={c.get('cpuset')} env={c.get('env')}")
        if store:
            arm.run_id = store.start(now_utc(), cfg.base_url, arm.engine, cfg.model, streams,
                                     health.get("context_length"), thinking, {**config, "levels": arm.levels},
                                     args.notes, pair_id, arm.label or None, args.profile)
            print(f"{tag}recording to {args.db} (run {arm.run_id})")

    def rec(arm: Arm, *a, **k):
        if store:
            store.sample(arm.run_id, *a, **k)

    host_ssh = args.ssh or KNOWN_SSH.get(cfg.host)

    def host_log(arm: Arm, block: str, moment: str) -> None:
        snap = host_snapshot(host_ssh)
        if not snap:
            return
        if store and arm.run_id is not None:
            store.host(arm.run_id, block, moment, snap)
        throttle = snap.get("throttle")
        print(f"  host {moment} {block}: gpu {snap.get('temp')} C, {snap.get('power')} W, sm {snap.get('sm')} MHz"
              + (f", throttle {throttle}" if throttle and throttle.strip("0x") else "")
              + f", mem avail {snap.get('mem')} GiB, load {snap.get('load')}")

    def run_block(arm: Arm, phase: str, ptype: str | None, rng: random.Random) -> None:
        if phase == "prefill":
            cpt = calibrate(cfg, rng)
            print(f"\n== prefill (concurrency 1, fresh salt per request, {cpt:.2f} chars/token) ==")
            for size in prefill_sizes:
                chars = int((size - 25) * cpt)  # ~25 tokens of header

                def measure(rnd, disc, slot, size=size, chars=chars):
                    _, before = server_state(cfg)
                    r = chat(cfg, prefill_prompt(rng, chars), args.prefill_gen, False)
                    _, after = server_state(cfg)
                    foreign = (after - before - 1) if before is not None and after is not None else None
                    tag = f"{size // 1024}K r{rnd}{'w' if disc else ''}"
                    if not r["ok"]:
                        print(f"  {tag}: FAILED, {r['error']}")
                        rec(arm, "prefill", None, 1, size, rnd, disc, error=r["error"], foreign=foreign)
                        return None
                    rate = r["ptok"] / r["first"]
                    srv = r["ptok"] / r["server_prefill_s"] if r["server_prefill_s"] else None
                    contaminated = bool(foreign) and not disc
                    rec(arm, "prefill", None, 1, size, rnd, disc, tps=rate, ttft=r["first"], ptok=r["ptok"],
                        server_prefill_s=r["server_prefill_s"], foreign=foreign,
                        error=f"foreign traffic (+{foreign})" if contaminated else None)
                    print(f"  {tag}: {rate:.1f} tok/s (ttft {r['first']:.2f}s"
                          + (f", server {srv:.1f} tok/s" if srv else "") + f") prompt {r['ptok']} tok"
                          + ((f" FOREIGN+{foreign}" + (", replaced" if contaminated else "")) if foreign else "")
                          + (" [discarded]" if disc else ""))
                    if contaminated:
                        return FOREIGN
                    return rate, {"ttft": r["first"], "server": srv}

                st = run_cell(args, f"{size // 1024}K", measure)
                if st is None:
                    continue
                ex = st.pop("extras")
                st["ttft"] = statistics.median(e["ttft"] for e in ex)
                srv = [e["server"] for e in ex if e["server"]]
                st["server_tps"] = statistics.median(srv) if srv else None
                arm.cells[("prefill", size)] = st
                if store:
                    store.cell(arm.run_id, "prefill", None, 1, size, st)
                print(f"  {size // 1024}K => median {st['median']:.1f} tok/s (spread {st['spread']:.1f}%"
                      + (", FLAGGED" if st["flagged"] else "") + f"), ttft {st['ttft']:.2f}s"
                      + (f", server {st['server_tps']:.1f} tok/s" if st["server_tps"] else ""))
            return
        print(f"\n== decode / {ptype} (max_tokens {gen}, {args.profile}) ==")
        warm_prompt = CODE_WARMUP if ptype == "code" else decode_prompts(ptype, 1, args.sparkdash_prompts)[0]
        chat(cfg, warm_prompt, 32, True, profile)  # sparkDash-style 32-token JIT warmup, not counted
        for lvl in arm.levels:
            prompts = decode_prompts(ptype, lvl, args.sparkdash_prompts)

            def measure(rnd, disc, slot, lvl=lvl, prompts=prompts):
                seeds = [args.seed + 7919 * i + 104729 * slot for i in range(lvl)] \
                    if args.profile == "sampled" else [None] * lvl
                _, before = server_state(cfg)
                w = decode_wave(cfg, prompts, gen, profile, seeds)
                _, after = server_state(cfg)
                foreign = (after - before - lvl) if before is not None and after is not None else None
                tag = f"x{lvl} r{rnd}{'w' if disc else ''}"
                if "agg" not in w:
                    print(f"  {tag}: FAILED, {w['error']}")
                    rec(arm, "decode", ptype, lvl, None, rnd, disc, error=w["error"], foreign=foreign)
                    return None
                contaminated = bool(foreign) and not disc
                err = w["error"] or (f"foreign traffic (+{foreign})" if contaminated else None)
                rec(arm, "decode", ptype, lvl, None, rnd, disc, tps=w["agg"], per=w["per"], streams=w["gauge"],
                    ttft=w["ttft"], error=err, accepted=w["accepted"], drafted=w["drafted"],
                    completion_min=w["ctok_min"], short=w["short"], foreign=foreign, shas=w["shas"])
                acc = f", accept {w['accepted'] / w['drafted']:.0%}" if w["drafted"] else ""
                note = f", streams {w['gauge']:.2f}/{lvl}" if lvl > 1 else ""
                print(f"  {tag}: agg {w['agg']:.1f} tok/s, mean/str {w['per']:.1f}{note}, "
                      f"ttft {w['ttft'] * 1000:.0f} ms{acc}"
                      + (f" SHORT {w['short']} (min {w['ctok_min']})" if w["short"] else "")
                      + ((f" FOREIGN+{foreign}" + (", replaced" if contaminated else "")) if foreign else "")
                      + (f" ({w['error']})" if w["error"] else "") + (" [discarded]" if disc else ""))
                if w["error"]:  # a partial wave is recorded but not averaged in
                    return None
                if contaminated:
                    return FOREIGN
                return w["agg"], {"ttft": w["ttft"], "acc": w["accepted"], "dra": w["drafted"],
                                  "short": w["short"], "shas": tuple(w["shas"])}

            st = run_cell(args, f"x{lvl}", measure)
            if st is None:
                continue
            ex = st.pop("extras")
            st["ttft"] = statistics.median(e["ttft"] for e in ex)
            dra = sum(e["dra"] or 0 for e in ex)
            st["accept"] = sum(e["acc"] or 0 for e in ex) / dra if dra else None
            st["short"] = sum(e["short"] for e in ex)
            st["shas"] = {e["shas"] for e in ex}  # greedy: one entry when every round wrote the same text
            arm.cells[("decode", ptype, lvl)] = st
            if store:
                store.cell(arm.run_id, "decode", ptype, lvl, None, st)
            print(f"  x{lvl} => median agg {st['median']:.1f} tok/s (mean {st['mean']:.1f}, spread "
                  f"{st['spread']:.1f}%" + (", FLAGGED" if st["flagged"] else "") + ")"
                  + (f", accept {st['accept']:.0%}" if st["accept"] is not None else "")
                  + (f", {st['short']} short stream(s)" if st["short"] else "")
                  + (f", {st['foreign']} round(s) replaced for foreign traffic" if st["foreign"] else "")
                  + (f", outputs varied across rounds ({len(st['shas'])} variants)"
                     if args.profile == "greedy" and len(st["shas"]) > 1 else ""))

    live = None
    try:
        for i, (phase, ptype) in enumerate(blocks):
            order = arms if len(arms) == 1 or i % 2 == 0 else arms[::-1]  # A B | B A | A B ...
            for arm in order:
                if arm.command and live is not arm:
                    switch_to(arm, cfg, args)
                    live = arm
                require_idle(cfg, args.force, f"before {phase}{'/' + ptype if ptype else ''}")
                if arm.run_id is None and arm.engine is None:
                    open_arm(arm)
                block = phase if phase == "prefill" else f"decode/{ptype}"
                host_log(arm, block, "before")
                # same prompt salts for both arms of a block; the per-invocation nonce keeps a later
                # run on a server that was not restarted from hitting this run's cached prompts
                run_block(arm, phase, ptype, random.Random(f"{args.seed}-{nonce}-{phase}-{ptype}"))
                host_log(arm, block, "after")
    finally:
        if store:
            for arm in arms:
                if arm.run_id is not None:
                    store.finish(arm.run_id, now_utc())
    if len(arms) == 2:
        report_pair(arms, args.spread_max)


def report_pair(arms: list[Arm], spread_max: float) -> None:
    a, b = arms
    keys = [k for k in a.cells if k in b.cells]
    print(f"\n== paired result: {b.label} vs {a.label} (medians; delta = {b.label} / {a.label} - 1) ==")
    deltas, same, differ = [], 0, 0
    for k in keys:
        ca, cb = a.cells[k], b.cells[k]
        d = (cb["median"] / ca["median"] - 1) * 100
        noisy = max(ca["spread"], cb["spread"])
        deltas.append(d)
        name = f"prefill {k[1] // 1024}K" if k[0] == "prefill" else f"{k[1]} x{k[2]}"
        extra = ""
        if k[0] == "prefill" and ca.get("server_tps") and cb.get("server_tps"):
            extra = f"  server {ca['server_tps']:.0f} -> {cb['server_tps']:.0f} " \
                    f"({(cb['server_tps'] / ca['server_tps'] - 1) * 100:+.1f}%)"
        if k[0] == "decode" and ca.get("accept") is not None and cb.get("accept") is not None:
            extra = f"  accept {ca['accept']:.0%} -> {cb['accept']:.0%}"
        if k[0] == "decode" and ca.get("shas") and cb.get("shas") and None not in {h for t in ca["shas"] | cb["shas"] for h in t}:
            if ca["shas"] == cb["shas"]:
                same += 1
                extra += "  same text"
            else:
                differ += 1
                extra += "  TEXT DIFFERS"
        within = " ~noise" if abs(d) <= noisy else ""
        print(f"  {name:16s} {ca['median']:8.1f} -> {cb['median']:8.1f}  {d:+6.1f}%  "
              f"(spread {ca['spread']:.1f}/{cb['spread']:.1f}%){within}{extra}")
    if deltas:
        better = sum(1 for d in deltas if d > 0)
        print(f"  {b.label} faster in {better}/{len(deltas)} cells; median delta {statistics.median(deltas):+.1f}%; "
              f"'~noise' = |delta| within the larger cell spread (threshold for reruns: {spread_max}%)")
    if same or differ:
        print(f"  outputs (engine token hashes, same prompts and seeds): identical in {same} decode cells, "
              f"different in {differ}")


def ingest_log(path: str, cfg: Cfg, args, store: Store) -> None:
    """Load a finished protocol-2 console log of this tool into the database (for sweeps that
    ran before database logging existed). Engine/version are probed now, not at run time."""
    lines = Path(path).read_text().splitlines()
    head = re.search(r"target (\S+) model (.+?); streams=(\d+) decode levels=\[(.*?)\] reps=(\d+) "
                     r"warm=(\d+) drain=([\d.]+)s gen=(\d+)", next((l for l in lines if l.startswith("target ")), ""))
    if not head:
        raise SystemExit(f"{path}: no 'target ...' header line; not an ab-bench log")
    cfg.base_url = head.group(1)
    cfg.model = head.group(2)
    engine = detect_engine(cfg, args)
    if args.engine_version:
        engine["version"] = args.engine_version[0]
    health = engine["detail"].get("health", {})
    config = {"reps": int(head.group(5)), "warm": int(head.group(6)), "drain_s": float(head.group(7)),
              "gen": int(head.group(8)), "levels": [int(x) for x in head.group(4).split(",") if x.strip()],
              "ingested_from": Path(path).name}
    run_id = store.start(args.started_at or now_utc(), cfg.base_url, engine, cfg.model, int(head.group(3)),
                         health.get("context_length"), "off", config,
                         (args.notes + "; " if args.notes else "")
                         + "ingested from console log; engine/version probed at ingest time",
                         profile="greedy", protocol=2)
    phase = ptype = None
    n = 0
    for line in lines:
        if line.startswith("== prefill"):
            phase, ptype = "prefill", None
        elif m := re.match(r"== decode / (\w+)", line):
            phase, ptype = "decode", m.group(1)
        elif phase == "prefill" and (m := re.match(
                r"\s+(\d+)K r(\d+)(w?): ([\d.]+) tok/s \(ttft ([\d.]+)s\) prompt (\d+) tok", line)):
            store.sample(run_id, "prefill", None, 1, int(m.group(1)) * 1024, int(m.group(2)), m.group(3) == "w",
                         tps=float(m.group(4)), ttft=float(m.group(5)), ptok=int(m.group(6)))
            n += 1
        elif phase == "decode" and (m := re.match(
                r"\s+x(\d+) r(\d+)(w?): agg ([\d.]+) tok/s, mean/str ([\d.]+)(?:, streams ([\d.]+)/\d+)?"
                r"(?:, ttft ([\d.]+) ms)?", line)):
            store.sample(run_id, "decode", ptype, int(m.group(1)), None, int(m.group(2)), m.group(3) == "w",
                         tps=float(m.group(4)), per=float(m.group(5)),
                         streams=float(m.group(6)) if m.group(6) else None,
                         ttft=float(m.group(7)) / 1000 if m.group(7) else None)
            n += 1
        elif m := re.match(r"\s+(?:(\d+)K|x(\d+)) r(\d+)(w?): FAILED, (.*)", line):
            store.sample(run_id, phase, ptype, int(m.group(2) or 1), int(m.group(1)) * 1024 if m.group(1) else None,
                         int(m.group(3)), m.group(4) == "w", error=m.group(5))
            n += 1
    store.finish(run_id, now_utc())
    print(f"ingested {n} rounds into run {run_id} ({engine['name']} {engine['version']}, {cfg.model})")


if __name__ == "__main__":
    main()
