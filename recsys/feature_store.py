"""Online feature store: per-user interaction history + real-time trending counters.

RedisStore is the production backend: the API appends events to a Redis Stream and the
worker (worker/stream_consumer.py) applies them. MemoryStore applies events inline, which
keeps local dev and tests dependency-free.
"""
import json
import time
from collections import Counter, defaultdict, deque

HISTORY_CAP = 200  # matches PipelineConfig.max_history_feature
STREAM = "events"
GROUP = "feature-updater"


def apply_event(store, event: dict) -> None:
    """Single source of truth for how an event mutates online features."""
    store.append(int(event["user_id"]), int(event["movie_id"]), int(event["timestamp"]))
    store.incr_trending(int(event["movie_id"]))


class MemoryStore:
    kind = "memory"

    def __init__(self):
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_CAP))
        self._trending: Counter = Counter()

    def get_history(self, user_id: int) -> list[tuple[int, int]]:
        return list(self._hist.get(user_id, ()))

    def append(self, user_id: int, movie_id: int, ts: int) -> None:
        self._hist[user_id].append((movie_id, ts))

    def incr_trending(self, movie_id: int) -> None:
        self._trending[movie_id] += 1

    def trending(self, k: int) -> list[tuple[int, float]]:
        return [(m, float(c)) for m, c in self._trending.most_common(k)]

    def publish(self, event: dict) -> str:
        apply_event(self, event)
        return f"{int(time.time() * 1000)}-0"

    def is_empty(self) -> bool:
        return not self._hist

    def seed(self, histories: dict[str, list]) -> None:
        for uid, events in histories.items():
            for mid, ts in events[-HISTORY_CAP:]:
                self.append(int(uid), int(mid), int(ts))

    def ping(self) -> bool:
        return True


class RedisStore:
    kind = "redis"

    def __init__(self, url: str, socket_timeout: float | None = None):
        import redis

        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=socket_timeout)

    def get_history(self, user_id: int) -> list[tuple[int, int]]:
        return [tuple(map(int, x.split(":"))) for x in self.r.lrange(f"hist:{user_id}", 0, -1)]

    def append(self, user_id: int, movie_id: int, ts: int) -> None:
        key = f"hist:{user_id}"
        pipe = self.r.pipeline()
        pipe.rpush(key, f"{movie_id}:{ts}")
        pipe.ltrim(key, -HISTORY_CAP, -1)
        pipe.execute()

    def incr_trending(self, movie_id: int) -> None:
        self.r.zincrby("trending", 1, movie_id)

    def trending(self, k: int) -> list[tuple[int, float]]:
        return [(int(m), s) for m, s in self.r.zrevrange("trending", 0, k - 1, withscores=True)]

    def publish(self, event: dict) -> str:
        return self.r.xadd(STREAM, {"data": json.dumps(event)}, maxlen=100_000, approximate=True)

    def is_empty(self) -> bool:
        return not self.r.exists("seeded")

    def seed(self, histories: dict[str, list]) -> None:
        # SET NX: with several API workers booting at once, exactly one seeds.
        if not self.r.set("seeded", 1, nx=True):
            return
        pipe = self.r.pipeline(transaction=False)
        for uid, events in histories.items():
            key = f"hist:{uid}"
            pipe.delete(key)
            pipe.rpush(key, *[f"{m}:{t}" for m, t in events[-HISTORY_CAP:]])
        pipe.execute()

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


def make_store(redis_url: str | None):
    return RedisStore(redis_url) if redis_url else MemoryStore()
