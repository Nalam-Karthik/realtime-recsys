"""Latency / throughput benchmark.

  python -m scripts.benchmark                 # in-process, per-stage latency
  python -m scripts.benchmark --url http://localhost:8000 --concurrency 32   # over HTTP
"""
import argparse
import asyncio
import json
import random
import time

import numpy as np

import recsys  # noqa: F401


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 3)


def in_process(n: int) -> dict:
    from recsys.service import Recommender

    rec = Recommender("artifacts")
    hists = json.loads((rec.dir / "histories.json").read_text())
    users = list(hists)
    for u in users[:20]:  # warm-up
        rec.recommend([tuple(x) for x in hists[u]], int(u))
    totals, stages = [], {}
    for _ in range(n):
        u = random.choice(users)
        t = time.perf_counter()
        out = rec.recommend([tuple(x) for x in hists[u][-200:]], int(u), k=10)
        totals.append((time.perf_counter() - t) * 1000)
        for s, ms in out["timings_ms"].items():
            stages.setdefault(s, []).append(ms)
    report = {"requests": n, "p50_ms": pct(totals, 50), "p95_ms": pct(totals, 95), "p99_ms": pct(totals, 99)}
    report["stages_p50_ms"] = {s: pct(v, 50) for s, v in stages.items()}
    return report


async def over_http(url: str, n: int, concurrency: int) -> dict:
    import httpx

    lat, sem = [], asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=10) as client:
        async def one():
            async with sem:
                t = time.perf_counter()
                r = await client.get(f"/recommendations/{random.randint(1, 610)}", params={"k": 10})
                r.raise_for_status()
                lat.append((time.perf_counter() - t) * 1000)

        await asyncio.gather(*[one() for _ in range(50)])  # warm-up
        lat.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(n)])
        wall = time.perf_counter() - t0
    return {"requests": n, "concurrency": concurrency, "rps": round(n / wall, 1),
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "p99_ms": pct(lat, 99)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()
    random.seed(0)
    report = asyncio.run(over_http(args.url, args.n, args.concurrency)) if args.url else in_process(args.n)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
