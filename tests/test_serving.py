import io
import json

import fakeredis
import pytest
from fastapi.testclient import TestClient

from recsys.feature_store import HISTORY_CAP, MemoryStore, RedisStore, apply_event
from worker.stream_consumer import ensure_group, process_batch


@pytest.fixture()
def client(artifact_root, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "ARTIFACT_DIR", str(artifact_root))
    monkeypatch.setattr(main, "REDIS_URL", None)
    with TestClient(main.app) as c:
        yield c


def test_health_and_model(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["store"] == "memory"
    assert "two_tower" in client.get("/model").json()["metrics"]


def test_recommendations_exclude_watched_items(client):
    history = {m for m, _ in client.app.state.store.get_history(1)}
    r = client.get("/recommendations/1", params={"k": 10, "explain": True}).json()
    assert r["strategy"] == "two_tower+lambdarank"
    assert len(r["items"]) == 10
    assert not history & {i["movie_id"] for i in r["items"]}
    assert all(i["because_you_watched"] for i in r["items"])
    assert set(r["timings_ms"]) == {"encode_user", "ann_search", "rerank", "diversify"}


def test_unknown_user_gets_popular_then_personalized_after_event(client):
    r = client.get("/recommendations/424242").json()
    assert r["strategy"] == "popular"
    ack = client.post("/events", json={"user_id": 424242, "movie_id": 10})
    assert ack.status_code == 202
    r = client.get("/recommendations/424242").json()
    assert r["strategy"].startswith("two_tower")
    assert 10 not in {i["movie_id"] for i in r["items"]}
    assert client.get("/trending").json()[0]["movie_id"] == 10


def test_session_recommendations_and_validation(client):
    r = client.post("/recommendations", json={"history": [10, 20, 30], "k": 5, "diversity": 0.5})
    assert r.status_code == 200 and r.json()["strategy"].endswith("+mmr")
    assert client.post("/recommendations", json={"history": [7]}).status_code == 422
    assert client.post("/events", json={"user_id": 1, "movie_id": 7}).status_code == 422
    assert client.get("/recommendations/1", params={"k": 0}).status_code == 422


def test_similar_items(client):
    items = client.get("/items/10/similar", params={"k": 5}).json()
    assert len(items) == 5 and 10 not in {i["movie_id"] for i in items}
    assert client.get("/items/7/similar").status_code == 404


def test_hot_reload_and_metrics(client):
    current = client.get("/health").json()["model_version"]
    assert client.post("/admin/reload").json()["current"] == current
    client.get("/recommendations/2")
    body = client.get("/metrics").text
    assert "recsys_stage_latency_seconds_bucket" in body and "recsys_recommendations_total" in body


def test_memory_store_caps_history():
    s = MemoryStore()
    for k in range(HISTORY_CAP + 50):
        apply_event(s, {"user_id": 1, "movie_id": k, "timestamp": k})
    h = s.get_history(1)
    assert len(h) == HISTORY_CAP and h[-1] == (HISTORY_CAP + 49, HISTORY_CAP + 49)


def test_worker_applies_stream_events_and_acks():
    store = RedisStore.__new__(RedisStore)
    store.r = fakeredis.FakeRedis(decode_responses=True)
    ensure_group(store.r)
    ensure_group(store.r)  # idempotent
    for m in (10, 20, 10):
        store.publish({"user_id": 5, "movie_id": m, "timestamp": 100 + m})
    assert store.get_history(5) == []  # API only enqueues; the worker owns the write
    sink = io.StringIO()
    assert process_batch(store, "w1", sink, block_ms=10) == 3
    assert store.get_history(5) == [(10, 110), (20, 120), (10, 110)]
    assert store.trending(1) == [(10, 2.0)]
    assert [json.loads(line)["movie_id"] for line in sink.getvalue().splitlines()] == [10, 20, 10]
    assert store.r.xpending("events", "feature-updater")["pending"] == 0
    assert process_batch(store, "w1", sink, block_ms=10) == 0
