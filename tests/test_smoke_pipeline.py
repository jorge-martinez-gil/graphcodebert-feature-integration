"""End-to-end CPU smoke test: the whole benchmark on tiny synthetic data.

No network, no GPU, no JDK, no real dataset files required — this guarantees the
pipeline (data -> features -> classifier -> metrics -> tables/report/figures) keeps
working everywhere, which is what makes the benchmark reproducible.
"""

import json
import os

from featfuse.config import ExperimentConfig
from featfuse.experiment import Experiment


def _write_synthetic_dataset(root):
    os.makedirs(root, exist_ok=True)
    recs = []
    for i in range(120):
        if i % 2 == 0:  # positive: near-identical
            c = f"public int f(int x){{ return x + {i % 5}; }}"
            recs.append({"index": i, "code1": c, "code2": c, "score": 1, "output": 1.0})
        else:  # negative: different
            recs.append({
                "index": i,
                "code1": f"int a = {i};",
                "code2": f"while(j < {i}){{ System.out.println(j++); }}",
                "score": 0, "output": 0.0,
            })
    with open(os.path.join(root, "data2.json"), "w") as fh:
        json.dump(recs, fh)


def test_classical_pipeline_end_to_end(tmp_path):
    data_root = tmp_path / "data"
    _write_synthetic_dataset(str(data_root))
    cfg = ExperimentConfig.from_dict({
        "name": "smoke_test",
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "features": ["exec_output_similarity", "token_jaccard", "edit_ratio", "cyclomatic_ratio"],
        "data": {"name": "irplag", "root": str(data_root), "single_file": "data2.json"},
        "model": {"backend": "classical", "classifier": "random_forest"},
        "eval": {"bootstrap_samples": 50, "primary_metric": "f1"},
    })
    exp = Experiment(cfg)
    results = exp.run()

    # results structure
    assert results["rows"], "no metric rows produced"
    test_row = [r for r in results["rows"] if r["split"] == "test"][0]
    assert test_row["f1"] >= 0.9  # separable synthetic data

    # artifacts written and traceable
    run_dir = exp.run_dir
    for fname in ["manifest.json", "results.json", "results_table.md",
                  "results_table.tex", "REPORT.md"]:
        assert os.path.exists(os.path.join(run_dir, fname)), f"missing {fname}"
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    assert manifest["seed"] == 0
    assert "config_hash" in manifest


def test_ablation_and_importance(tmp_path):
    data_root = tmp_path / "data"
    _write_synthetic_dataset(str(data_root))
    cfg = ExperimentConfig.from_dict({
        "name": "smoke_ablate",
        "output_dir": str(tmp_path / "runs"),
        "features": ["token_jaccard", "edit_ratio", "cyclomatic_ratio"],
        "data": {"name": "irplag", "root": str(data_root), "single_file": "data2.json"},
        "model": {"backend": "classical", "classifier": "random_forest"},
    })
    exp = Experiment(cfg)
    ab = exp.ablate(write=False)
    settings = {r["setting"] for r in ab["ablation"]}
    assert "all_features" in settings
    assert any(s.startswith("only:") for s in settings)

    imp = exp.feature_importance()
    assert len(imp["columns"]) == 3
    assert len(imp["permutation_importance"]) == 3
