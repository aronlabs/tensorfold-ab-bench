# tensorfold-ab-bench

Prefill and decode benchmark for one OpenAI-compatible LLM server, or a paired A/B of two server configs. Built for [TensorFold](https://github.com/MiaAI-Lab) on a DGX Spark; the base protocol (prompts and rate formulas) comes from sparkDash's DecodeBench/PrefillBench.

One file, Python 3 standard library only. Results go to SQLite.

## What it measures

- **Decode:** tok/s per stream and in aggregate, at 1 to N concurrent streams, for four prompt types: structured, prose, code, and JSON. Each stream gets its own prompt.
- **Prefill:** prompt tok/s from 1K to 32K tokens, and to 128K with `--prefill-long`. It reports the client rate (prompt tokens / TTFT) and, on TensorFold, the engine's own rate from `prefill_s`.
- **Per cell:** median, mean, min, max and spread. A cell whose spread is above `--spread-max` is re-measured.
- **Also per cell:** MTP draft acceptance, streams that stopped short, and output hashes, which show whether two configs produced the same text.
- **Host state** before and after each block: GPU temperature, clock, power, throttle reasons, free RAM, and load.

## Guards

- Refuses to start while the server is busy. `--force` overrides this.
- Rounds that overlap outside traffic are excluded and re-run.

## Paired A/B

Pass `--pair LABEL=COMMAND` twice. Each command switches the server to that arm; it can be any shell command that leaves the new server starting on the same URL. Blocks alternate A B, then B A, and so on, so both arms get early and late slots. At the end the tool prints per-cell medians, the B-vs-A delta, and whether the output text matched.

## Usage

```sh
python3 ab-bench.py --url http://HOST:8888/v1                   # full greedy sweep
python3 ab-bench.py --profile sampled --phases decode            # production-like decode
python3 ab-bench.py --phases prefill --prefill-long              # adds 64K and 128K
python3 ab-bench.py --pair "old=ssh box 'cd OLD && ./stop.sh && ./start.sh'" \
                    --pair "new=ssh box 'cd NEW && ./stop.sh && ./start.sh'"
python3 ab-bench.py --help
```

Defaults in the script (server URL, the SSH alias used to read container details, and the database path `~/projects/llm-bench/results/ab-bench.db`) point at the author's lab. Override them with `--url` and `--db`.
