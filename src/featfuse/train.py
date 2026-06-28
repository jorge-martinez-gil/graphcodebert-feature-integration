"""Neural training loop for the fusion classifier (optional, requires torch).

This reproduces the paper's setup in a generalized form: any HuggingFace code
encoder + any registered fusion strategy + an arbitrary engineered-feature vector.
It is imported lazily by :meth:`featfuse.experiment.Experiment._run_neural` so the
classical benchmark never pays the torch import cost.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from .config import ExperimentConfig
from .features.base import FeatureSet
from .metrics import classification_metrics
from .registry import DATASETS, MODELS
from .reproducibility import make_manifest, set_seed
from .types import CodePair


def _encode_batch(tokenizer, pairs: List[CodePair], max_length: int):
    import torch

    enc = tokenizer(
        [p.code1 for p in pairs], [p.code2 for p in pairs],
        truncation=True, padding="max_length", max_length=max_length, return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def run_neural(cfg: ExperimentConfig, write: bool = True, run_dir: str = "runs/neural") -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .models.head import build_fusion_classifier

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DATASETS.create(
        cfg.data.name, root=cfg.data.root, train=cfg.data.train, validation=cfg.data.validation,
        test=cfg.data.test, single_file=cfg.data.single_file,
        split_fractions=tuple(cfg.data.split_fractions), seed=cfg.seed,
    )
    fs = FeatureSet.from_names(cfg.features)
    encoder = MODELS.create("hf", name=cfg.model.encoder)
    tokenizer = encoder.tokenizer
    model = build_fusion_classifier(encoder, fs.dim, fusion=cfg.model.fusion, num_labels=2,
                                    dropout=0.1).to(device)

    def make_loader(split, shuffle):
        pairs = ds[split]
        ids, mask = _encode_batch(tokenizer, pairs, cfg.model.max_length)
        feats = torch.tensor(fs.transform(pairs), dtype=torch.float32)
        labels = torch.tensor([p.label for p in pairs], dtype=torch.long)
        return DataLoader(TensorDataset(ids, mask, feats, labels),
                          batch_size=cfg.model.batch_size, shuffle=shuffle)

    train_loader = make_loader("train", True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.model.learning_rate)

    model.train()
    for _ in range(cfg.model.epochs):
        for ids, mask, feats, labels in train_loader:
            opt.zero_grad()
            out = model(ids.to(device), mask.to(device), feats.to(device), labels.to(device))
            out.loss.backward()
            opt.step()

    rows = []
    preds_store = {}
    model.eval()
    for split in ("validation", "test"):
        if split not in ds.available():
            continue
        loader = make_loader(split, False)
        ys, ps, probs = [], [], []
        with torch.no_grad():
            for ids, mask, feats, labels in loader:
                logits = model(ids.to(device), mask.to(device), feats.to(device)).logits
                prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                ys.extend(labels.numpy().tolist())
                probs.extend(prob.tolist())
                ps.extend((prob >= 0.5).astype(int).tolist())
        m = classification_metrics(ys, ps, probs, positive_label=cfg.eval.positive_label)
        m.update({"method": f"{cfg.model.encoder} + {cfg.model.fusion} + features", "split": split})
        rows.append(m)
        preds_store[split] = {"y": ys, "pred": ps, "proba": probs}

    results = {"rows": rows, "predictions": preds_store, "feature_columns": fs.columns}
    if write:
        os.makedirs(run_dir, exist_ok=True)
        manifest = make_manifest(cfg.to_dict(), cfg.seed)
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in results.items() if k != "predictions"}, fh, indent=2)
    return results
