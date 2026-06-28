"""Typed, YAML-backed experiment configuration.

A configuration fully specifies a reproducible benchmark run: which dataset, which
engineered features, which model/backend and fusion strategy, and the evaluation
protocol. Everything the runner needs lives here so that a single YAML file is a
complete, shareable description of an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    name: str = "irplag"
    root: str = "data"
    # Optional explicit split files (relative to ``root``); if omitted the loader
    # falls back to a deterministic seeded split of ``data2.json``.
    train: Optional[str] = None
    validation: Optional[str] = None
    test: Optional[str] = None
    single_file: Optional[str] = "data2.json"
    split_fractions: List[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])


@dataclass
class ModelConfig:
    # backend: "classical" (sklearn over engineered features — runs anywhere) or
    # "neural" (HuggingFace encoder + fusion head — reproduces the paper, needs torch).
    backend: str = "classical"
    encoder: str = "microsoft/graphcodebert-base"
    classifier: str = "gradient_boosting"   # used by the classical backend
    fusion: str = "concat"                   # used by the neural backend
    max_length: int = 512
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5


@dataclass
class EvalConfig:
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    primary_metric: str = "f1"
    positive_label: int = 1


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    output_dir: str = "runs"
    features: List[str] = field(default_factory=lambda: ["exec_output_similarity"])
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- (de)serialisation -------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = dict(d or {})
        data = DataConfig(**(d.pop("data", {}) or {}))
        model = ModelConfig(**(d.pop("model", {}) or {}))
        ev = EvalConfig(**(d.pop("eval", {}) or {}))
        return cls(data=data, model=model, eval=ev, **d)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
