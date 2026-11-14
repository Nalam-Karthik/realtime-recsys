from dataclasses import dataclass, field, asdict


@dataclass
class TowerConfig:
    dim: int = 64
    history_len: int = 30
    temperature: float = 0.05
    user_id_dropout: float = 0.15  # trains the cold-start (unknown-user) path
    batch_size: int = 1024
    epochs: int = 15
    lr: float = 2e-3
    weight_decay: float = 1e-6
    seed: int = 42


@dataclass
class RankerConfig:
    n_candidates: int = 200
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 50


@dataclass
class IndexConfig:
    kind: str = "hnsw"  # "hnsw" (approximate) or "flat" (exact)
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 128


@dataclass
class PipelineConfig:
    data_dir: str = "data/ml-latest-small"
    artifact_dir: str = "artifacts"
    val_frac: float = 0.1
    test_frac: float = 0.1
    max_history_feature: int = 200  # same cap the online feature store applies
    tower: TowerConfig = field(default_factory=TowerConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    def to_dict(self) -> dict:
        return asdict(self)
