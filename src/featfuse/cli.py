"""FeatFuse command-line interface.

    featfuse list [features|fusion|models|datasets]
    featfuse run     -c configs/smoke.yaml [--ablate]
    featfuse ablate  -c configs/smoke.yaml
    featfuse importance -c configs/smoke.yaml
    featfuse info

Every benchmark runs with a single command, and every run writes a fully traceable
directory (see :mod:`featfuse.experiment`).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .registry import DATASETS, FEATURES, FUSIONS, MODELS

# Ensure all plugins are registered.
from . import features as _f  # noqa: F401
from . import fusion as _fu  # noqa: F401
from . import models as _m  # noqa: F401
from . import data as _d  # noqa: F401

_REG = {"features": FEATURES, "fusion": FUSIONS, "models": MODELS, "datasets": DATASETS}


def _cmd_list(args) -> int:
    kinds = [args.kind] if args.kind else list(_REG)
    for kind in kinds:
        reg = _REG[kind]
        print(f"\n{kind} ({len(reg)}):")
        for name in reg.names():
            meta = reg.meta(name)
            extra = " ".join(f"{k}={v}" for k, v in meta.items() if k in {"family", "description", "language"})
            print(f"  - {name:26s} {extra}")
    return 0


def _cmd_run(args) -> int:
    from .experiment import ExperimentConfig, Experiment

    cfg = ExperimentConfig.from_yaml(args.config)
    exp = Experiment(cfg)
    results = exp.run()
    if args.ablate:
        exp.ablate()
    print(f"\nRun '{cfg.name}' complete → {exp.run_dir}/")
    for r in results.get("rows", []):
        print(f"  [{r.get('split'):10s}] "
              f"F1={r.get('f1', float('nan')):.4f}  MCC={r.get('mcc', float('nan')):.4f}  "
              f"AUC={r.get('roc_auc', float('nan')):.4f}")
    print(f"  artifacts: manifest.json, results.json, results_table.{{md,tex}}, REPORT.md, figures/")
    return 0


def _cmd_ablate(args) -> int:
    from .experiment import ExperimentConfig, Experiment

    cfg = ExperimentConfig.from_yaml(args.config)
    out = Experiment(cfg).ablate()
    print(json.dumps(out, indent=2))
    return 0


def _cmd_importance(args) -> int:
    from .experiment import ExperimentConfig, Experiment

    cfg = ExperimentConfig.from_yaml(args.config)
    out = Experiment(cfg).feature_importance()
    for col, imp in sorted(zip(out["columns"], out["permutation_importance"]), key=lambda t: -t[1]):
        print(f"  {col:28s} {imp:+.4f}")
    return 0


def _cmd_info(args) -> int:
    print(f"FeatFuse {__version__}")
    print(f"  features:  {len(FEATURES)}  ({', '.join(FEATURES.names())})")
    print(f"  fusion:    {len(FUSIONS)}  ({', '.join(FUSIONS.names())})")
    print(f"  models:    {len(MODELS)}  ({', '.join(MODELS.names())})")
    print(f"  datasets:  {len(DATASETS)}  ({', '.join(DATASETS.names())})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="featfuse", description="Feature integration benchmark for code LMs.")
    p.add_argument("--version", action="version", version=f"featfuse {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="list registered plugins")
    pl.add_argument("kind", nargs="?", choices=list(_REG), help="which registry to list")
    pl.set_defaults(func=_cmd_list)

    pr = sub.add_parser("run", help="run a benchmark configuration")
    pr.add_argument("-c", "--config", required=True)
    pr.add_argument("--ablate", action="store_true", help="also run feature ablation")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("ablate", help="run feature ablation only")
    pa.add_argument("-c", "--config", required=True)
    pa.set_defaults(func=_cmd_ablate)

    pi = sub.add_parser("importance", help="compute feature importance")
    pi.add_argument("-c", "--config", required=True)
    pi.set_defaults(func=_cmd_importance)

    pf = sub.add_parser("info", help="show registered components")
    pf.set_defaults(func=_cmd_info)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
