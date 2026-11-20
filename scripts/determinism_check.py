"""Retrain CTGAN and TVAE with the pipeline's settings and record hashes and scores.

usage: python scripts/determinism_check.py <label> <out_dir> [default|det]
  default = the pipeline as-is (seeded random/NumPy/torch before fit)
  det     = additionally torch.use_deterministic_algorithms(True) and torch.set_num_threads(1)

Writes <out_dir>/<label>.json with, per generator: md5 of the synthetic CSV, md5 of the network
weights, fit time, quality_score, utility_ratio, selection_score, mia_auc, exact_match_rate.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch  # noqa: I001
import xgboost  # noqa: F401  (load before torch: see stages/evaluate.py)

from synthguard.config import load_params
from synthguard.evaluation.fidelity import evaluate_fidelity
from synthguard.evaluation.privacy import evaluate_privacy
from synthguard.evaluation.score import score_candidate
from synthguard.evaluation.utility import evaluate_trtr, evaluate_tstr
from synthguard.generators.runner import load_processed, n_synthetic, training_frame
from synthguard.generators.sdv_models import CTGANGenerator, TVAEGenerator


def weights_md5(gen) -> str:
    model = gen.model._model
    nets = [getattr(model, a) for a in ("_generator", "decoder") if getattr(model, a, None)]
    h = hashlib.md5()
    for net in nets:
        for tensor in net.state_dict().values():
            h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    label, out_dir = sys.argv[1], Path(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "default"
    if mode == "det":
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = load_params(smoke=False)
    train, holdout, metadata = load_processed()
    fit_df, n = training_frame(train, p), n_synthetic(holdout, p)
    u, seed = p["evaluation"]["utility"], p["seed"]
    trtr = evaluate_trtr(fit_df, holdout, u, seed)

    result = {
        "label": label, "mode": mode, "platform": platform.platform(),
        "machine": platform.machine(), "cpus": os.cpu_count(),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
    }  # fmt: skip
    for cls, key in [(CTGANGenerator, "ctgan"), (TVAEGenerator, "tvae")]:
        start = time.time()
        gen = cls(metadata, seed=seed, **p["generators"][key]).fit(fit_df)
        synth = gen.sample(n)
        csv = synth.to_csv(index=False).encode()
        f, _ = evaluate_fidelity(fit_df, synth, metadata)
        t = evaluate_tstr(synth, holdout, trtr, u, seed)
        pr = evaluate_privacy(
            synth, fit_df, holdout, p["evaluation"]["privacy"]["mia_sample_size"], seed
        )
        scored = score_candidate({**f, **t, **pr}, p["selection"])
        result[key] = {
            "csv_md5": hashlib.md5(csv).hexdigest(),
            "weights_md5": weights_md5(gen),
            "fit_seconds": round(time.time() - start, 1),
            **{k: scored[k] for k in ("quality_score", "utility_ratio", "selection_score",
                                      "mia_auc", "exact_match_rate", "privacy_gate")},
        }  # fmt: skip
        print(label, key, result[key], flush=True)
    (out_dir / f"{label}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
