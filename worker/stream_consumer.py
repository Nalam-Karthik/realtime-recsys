"""Redis Streams consumer: applies interaction events to the online feature store.

Runs as a consumer group, so several workers can share the stream; events are acked only
after they are applied, and unacked events are re-delivered after a crash. Every event is
also appended to an offline log that the next training run can consume.
"""
import json
import logging
import os
import socket
from pathlib import Path

from recsys.feature_store import GROUP, STREAM, RedisStore, apply_event

log = logging.getLogger("worker")


def ensure_group(r) -> None:
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception as e:  # BUSYGROUP: already exists
        if "BUSYGROUP" not in str(e):
            raise


BLOCK_MS = 2000


def process_batch(store: RedisStore, consumer: str, sink, block_ms: int = BLOCK_MS) -> int:
    r = store.r
    # First drain anything delivered to this consumer but never acked (crash recovery), then new events.
    for start in ("0", ">"):
        resp = r.xreadgroup(GROUP, consumer, {STREAM: start}, count=256, block=None if start == "0" else block_ms)
        if resp and resp[0][1]:
            break
    else:
        return 0
    n = 0
    for msg_id, fields in resp[0][1]:
        event = json.loads(fields["data"])
        apply_event(store, event)
        sink.write(json.dumps(event) + "\n")
        r.xack(STREAM, GROUP, msg_id)
        n += 1
    sink.flush()
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # The socket timeout must outlast the blocking XREADGROUP, or an idle stream raises TimeoutError.
    store = RedisStore(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), socket_timeout=BLOCK_MS / 1000 + 5)
    ensure_group(store.r)
    consumer = os.environ.get("CONSUMER_NAME", socket.gethostname())
    log_path = Path(os.environ.get("EVENT_LOG", "data/events.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("consuming stream=%s group=%s consumer=%s", STREAM, GROUP, consumer)
    with log_path.open("a") as sink:
        while True:
            n = process_batch(store, consumer, sink)
            if n:
                log.info("applied %d events", n)


if __name__ == "__main__":
    main()
