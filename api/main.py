"""FastAPI serving layer for the two-stage recommender."""
import os
import time
from contextlib import asynccontextmanager

import recsys  # noqa: F401  (sets OpenMP env vars before torch/faiss load)
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest, multiprocess

from recsys.feature_store import make_store
from recsys.service import Recommender

from .schemas import Event, EventAck, Item, RecommendResponse, SessionRequest

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
REDIS_URL = os.getenv("REDIS_URL")

REQUEST_LATENCY = Histogram("recsys_request_latency_seconds", "End-to-end request latency", ["endpoint"],
                            buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1))
STAGE_LATENCY = Histogram("recsys_stage_latency_seconds", "Per-stage inference latency", ["stage"],
                          buckets=(.0001, .00025, .0005, .001, .0025, .005, .01, .025, .05))
RECS_SERVED = Counter("recsys_recommendations_total", "Recommendation responses", ["strategy"])
EVENTS = Counter("recsys_events_total", "Ingested interaction events", ["event_type"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender = Recommender(ARTIFACT_DIR)
    app.state.store = make_store(REDIS_URL)
    if app.state.store.is_empty():
        app.state.store.seed(app.state.recommender.seed_histories())
    yield


app = FastAPI(title="Real-Time Recommendation Engine", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def track_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    REQUEST_LATENCY.labels(getattr(route, "path", "unmatched")).observe(time.perf_counter() - start)
    return response


def _respond(result: dict, user_id: int | None) -> RecommendResponse:
    for stage, ms in result["timings_ms"].items():
        STAGE_LATENCY.labels(stage).observe(ms / 1000)
    RECS_SERVED.labels(result["strategy"]).inc()
    return RecommendResponse(user_id=user_id, model_version=app.state.recommender.version, **result)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.recommender.version,
            "store": app.state.store.kind, "store_ok": app.state.store.ping()}


@app.get("/recommendations/{user_id}", response_model=RecommendResponse)
async def recommend_user(user_id: int, k: int = Query(10, ge=1, le=100),
                         diversity: float = Query(0.0, ge=0.0, le=1.0), explain: bool = False):
    history = app.state.store.get_history(user_id)
    result = await run_in_threadpool(app.state.recommender.recommend, history, user_id, k, diversity, explain)
    return _respond(result, user_id)


@app.post("/recommendations", response_model=RecommendResponse)
async def recommend_session(req: SessionRequest):
    rec = app.state.recommender
    unknown = [m for m in req.history if m not in rec.item2idx]
    if unknown:
        raise HTTPException(422, f"unknown movie_ids: {unknown[:10]}")
    history = [(m, rec.snapshot_ts) for m in req.history]
    result = await run_in_threadpool(rec.recommend, history, None, req.k, req.diversity, req.explain)
    return _respond(result, None)


@app.get("/items/{movie_id}/similar", response_model=list[Item])
def similar_items(movie_id: int, k: int = Query(10, ge=1, le=100)):
    if movie_id not in app.state.recommender.item2idx:
        raise HTTPException(404, "unknown movie_id")
    return app.state.recommender.similar(movie_id, k)


@app.get("/trending")
def trending(k: int = Query(10, ge=1, le=100)):
    rec = app.state.recommender
    return [{"movie_id": m, "events": c, **(rec.describe(m) or {})} for m, c in app.state.store.trending(k)]


@app.post("/events", response_model=EventAck, status_code=202)
def ingest_event(event: Event):
    if event.movie_id not in app.state.recommender.item2idx:
        raise HTTPException(422, "unknown movie_id")
    payload = event.model_dump()
    payload["timestamp"] = payload["timestamp"] or int(time.time())
    stream_id = app.state.store.publish(payload)
    EVENTS.labels(event.event_type).inc()
    return EventAck(queued=True, stream_id=stream_id)


@app.get("/model")
def model_info():
    m = app.state.recommender.manifest
    return {k: m[k] for k in ("version", "created_at", "dataset", "metrics", "ranker_feature_importance")}


@app.post("/admin/reload")
async def reload_model():
    """Hot-swap to the artifact version in LATEST without dropping traffic."""
    new = await run_in_threadpool(Recommender, ARTIFACT_DIR)
    old_version, app.state.recommender = app.state.recommender.version, new
    return {"previous": old_version, "current": new.version}


@app.get("/metrics")
def metrics():
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):  # aggregate across uvicorn worker processes
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
