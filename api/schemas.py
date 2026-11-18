from typing import Literal

from pydantic import BaseModel, Field


class Item(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float
    because_you_watched: str | None = None


class RecommendResponse(BaseModel):
    user_id: int | None = None
    model_version: str
    strategy: str
    items: list[Item]
    timings_ms: dict[str, float]


class SessionRequest(BaseModel):
    """Anonymous / cold-start session: recommend from an ad-hoc list of watched movies."""
    history: list[int] = Field(default_factory=list, max_length=200)
    k: int = Field(10, ge=1, le=100)
    diversity: float = Field(0.0, ge=0.0, le=1.0)
    explain: bool = False


class Event(BaseModel):
    user_id: int
    movie_id: int
    event_type: Literal["watch", "click", "rate"] = "watch"
    rating: float | None = Field(None, ge=0.5, le=5.0)
    timestamp: int | None = None


class EventAck(BaseModel):
    queued: bool
    stream_id: str
