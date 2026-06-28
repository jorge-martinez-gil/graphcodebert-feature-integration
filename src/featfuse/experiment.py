"""The config-driven, reproducible experiment runner.

``Experiment.run()`` executes one benchmark configuration end to end and writes a
fully traceable run directory:

    runs/<name>/
      manifest.json        reproducibility manifest (seed, versions, git, config)
      results.json         metrics (+ bootstrap CIs) for every split
      results_table.md     auto-generated Markdown table
      results_table.tex    auto-generated LaTeX (booktabs) table
      REPORT.md            human-readable report
      figures/*.png        publication-quality figures

Two backends are supported. The **classical** backend (engineered features → a
scikit-learn classifier) runs anywhere — no GPU, no network — and is what powers
CI and the quickstart. The **neural** backend (HuggingFace encoder + fusion head)
reproduces the paper and requires the ``featfuse[neural]`` extra.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig
from .features.base import FeatureSet
from .metrics import classification_metrics
from .registry import DATASETS
from .reproducibility import make_manifest, set_seed
from .stats import bootstrap_metric_ci, mcnemar_test
from .types import CodePair


def get_classifier(name: str, seed: int = 42):
    """Map a classifier name to a scikit-learn estimator."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    name = name.lower()
    table = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced"),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=seed),
        "svm": lambda: SVC(probability=True, class_weight="balanced", random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed),
    }
    if name not in table:
        raise KeyError(f"Unknown classifier '{name}'. Available: {sorted(table)}")
    return table[name]()


def _build_pipeline(name: str, seed: int):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()), ("clf", get_classifier(name, seed))])


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.run_dir = os.path.join(config.output_dir, config.name)

    # ---- data / features ---------------------------------------------------
    def load_dataset(self):
        d = self.cfg.data
        return DATASETS.create(
            d.name, root=d.root, train=d.train, validation=d.validation, test=d.test,
            single_file=d.single_file, split_fractions=tuple(d.split_fractions), seed=self.cfg.seed,
        )

    def featurize(self, pairs: List[CodePair], feature_set: FeatureSet,
                  encoder_name: Optional[str], encoder_dim: int) -> np.ndarray:
        X = feature_set.transform(pairs)
        if encoder_name and encoder_name != "none":
            from .registry import MODELS

            enc = MODELS.create(encoder_name, embed_dim=encoder_dim) if encoder_name == "toy" \
                else MODELS.create(encoder_name)
            emb = enc.encode_pairs([p.code1 for p in pairs], [p.code2 for p in pairs])
            X = np.hstack([X, emb]) if X.size else emb
        return X

    # ---- core run ----------------------------------------------------------
    def run(self, write: bool = True) -> Dict[str, Any]:
        cfg = self.cfg
        if cfg.model.backend == "neural":
            return self._run_neural(write=write)

        set_seed(cfg.seed)
        ds = self.load_dataset()
        fs = FeatureSet.from_names(cfg.features)
        enc_name = getattr(cfg.model, "encoder_kind", None)  # optional embedding block
        enc_dim = 64

        Xtr = self.featurize(ds["train"], fs, enc_name, enc_dim)
        ytr = np.array([p.label for p in ds["train"]])
        pipe = _build_pipeline(cfg.model.classifier, cfg.seed)
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        train_time = time.perf_counter() - t0

        rows: List[Dict[str, Any]] = []
        per_split_preds: Dict[str, Any] = {}
        for split in ("validation", "test"):
            if split not in ds.available():
                continue
            pairs = ds[split]
            X = self.featurize(pairs, fs, enc_name, enc_dim)
            y = np.array([p.label for p in pairs])
            t1 = time.perf_counter()
            proba = pipe.predict_proba(X)[:, 1]
            infer_time = time.perf_counter() - t1
            pred = (proba >= 0.5).astype(int)
            m = classification_metrics(y, pred, proba, positive_label=cfg.eval.positive_label)
            # bootstrap CI on the primary metric
            ci = bootstrap_metric_ci(
                lambda t, p: classification_metrics(t, p)[cfg.eval.primary_metric],
                y, pred, n_boot=cfg.eval.bootstrap_samples, confidence=cfg.eval.confidence, seed=cfg.seed,
            )
            m.update({
                "method": f"{cfg.model.classifier} ({fs.dim} feat) · {split}",
                "split": split,
                f"{cfg.eval.primary_metric}_ci_low": ci["low"],
                f"{cfg.eval.primary_metric}_ci_high": ci["high"],
                "infer_latency_ms": 1000.0 * infer_time / max(1, len(y)),
            })
            rows.append(m)
            per_split_preds[split] = {"y": y.tolist(), "pred": pred.tolist(), "proba": proba.tolist()}

        results = {
            "rows": rows,
            "feature_columns": fs.columns,
            "train_time_s": train_time,
            "n_features": fs.dim if not enc_name or enc_name == "none" else None,
            "predictions": per_split_preds,
        }
        if write:
            self._write_artifacts(results, fs, ds)
        return results

    def _run_neural(self, write: bool = True) -> Dict[str, Any]:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "The neural backend requires torch + transformers. "
                "Install:  pip install 'featfuse[neural]'  (or use backend: classical)."
            ) from exc
        # Full neural training loop lives in featfuse.train (kept out of the import
        # path so the classical benchmark has no heavy dependencies).
        from .train import run_neural

        return run_neural(self.cfg, write=write, run_dir=self.run_dir)

    # ---- ablation & importance --------------------------------------------
    def ablate(self, write: bool = True) -> Dict[str, Any]:
        """Leave-one-feature-out and single-feature ablation on the primary metric."""
        cfg = self.cfg
        set_seed(cfg.seed)
        ds = self.load_dataset()
        metric = cfg.eval.primary_metric
        all_feats = list(cfg.features)

        def eval_subset(names: List[str]) -> Dict[str, float]:
            if not names:
                return {metric: float("nan")}
            fs = FeatureSet.from_names(names)
            pipe = _build_pipeline(cfg.model.classifier, cfg.seed)
            pipe.fit(fs.transform(ds["train"]), np.array([p.label for p in ds["train"]]))
            te = ds["test"]
            proba = pipe.predict_proba(fs.transform(te))[:, 1]
            pred = (proba >= 0.5).astype(int)
            return classification_metrics(np.array([p.label for p in te]), pred, proba)

        full = eval_subset(all_feats)
        rows = [{"setting": "all_features", "features": ",".join(all_feats), metric: full[metric]}]
        for f in all_feats:
            sub = [x for x in all_feats if x != f]
            r = eval_subset(sub)
            rows.append({"setting": f"-{f}", "features": ",".join(sub),
                         metric: r[metric], f"delta_{metric}": r[metric] - full[metric]})
        for f in all_feats:
            r = eval_subset([f])
            rows.append({"setting": f"only:{f}", "features": f, metric: r[metric]})

        results = {"ablation": rows, "primary_metric": metric}
        if write:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "ablation.json"), "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
        return results

    def feature_importance(self) -> Dict[str, Any]:
        """Permutation importance (+ native importances) for the engineered features."""
        from sklearn.inspection import permutation_importance

        cfg = self.cfg
        set_seed(cfg.seed)
        ds = self.load_dataset()
        fs = FeatureSet.from_names(cfg.features)
        pipe = _build_pipeline(cfg.model.classifier, cfg.seed)
        Xtr = fs.transform(ds["train"]); ytr = np.array([p.label for p in ds["train"]])
        pipe.fit(Xtr, ytr)
        Xte = fs.transform(ds["test"]); yte = np.array([p.label for p in ds["test"]])
        perm = permutation_importance(pipe, Xte, yte, n_repeats=20, random_state=cfg.seed, scoring="f1")
        out = {"columns": fs.columns,
               "permutation_importance": perm.importances_mean.tolist(),
               "permutation_std": perm.importances_std.tolist()}
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            out["native_importance"] = clf.feature_importances_.tolist()
        return out

    # ---- artifacts ---------------------------------------------------------
    def _write_artifacts(self, results: Dict[str, Any], fs: FeatureSet, ds) -> None:
        from . import report as _report
        from . import viz as _viz

        os.makedirs(self.run_dir, exist_ok=True)
        fig_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        manifest = make_manifest(self.cfg.to_dict(), self.cfg.seed,
                                 extra={"dataset_summary": ds.summary()})
        with open(os.path.join(self.run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        with open(os.path.join(self.run_dir, "results.json"), "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in results.items() if k != "predictions"}, fh, indent=2)

        _report.write_tables(results["rows"], self.run_dir,
                             caption=f"{self.cfg.data.name} results ({self.cfg.name}).",
                             label=f"tab:{self.cfg.name}")
        figures = []
        try:
            test_pred = results.get("predictions", {}).get("test")
            if test_pred:
                figures.append(_viz.plot_calibration(
                    test_pred["y"], test_pred["proba"], os.path.join(fig_dir, "calibration.png")))
            mm = {r["split"]: r for r in results["rows"]}
            figures.append(_viz.plot_metric_bars(
                {k: v for k, v in mm.items()},
                ["accuracy", "f1", "mcc", "roc_auc"], os.path.join(fig_dir, "metrics.png")))
            Xtr = fs.transform(ds["train"])
            if Xtr.shape[1] > 1:
                figures.append(_viz.plot_feature_correlation(
                    Xtr, fs.columns, os.path.join(fig_dir, "feature_correlation.png")))
        except Exception:  # pragma: no cover - viz must never break a run
            pass
        _report.write_report(self.run_dir, results, manifest, figures)


def run_config(path: str, do_ablate: bool = False) -> Dict[str, Any]:
    cfg = ExperimentConfig.from_yaml(path)
    exp = Experiment(cfg)
    out = exp.run()
    if do_ablate:
        out["ablation"] = exp.ablate()["ablation"]
    return out
