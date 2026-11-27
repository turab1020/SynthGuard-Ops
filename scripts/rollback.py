"""Roll the served model back to the one before the last promotion.

Git mode (default; this is what Render serves): restore deploy/model/ and metrics/classifier.json
from just before the most recent commit that changed them on the current branch (or from --to REF),
and write reports/rollback.md. Commit that on a rollback/* branch and merge its PR: deploy.yml
re-runs tests + the eval gate against the restored benchmark and redeploys.

Registry mode (--registry, local MLflow registry): move synthguard-classifier@champion (and the
generator) back to @previous, tag the replaced version `rolled_back`; then re-export with
scripts/export_models.py.

usage: python scripts/rollback.py [--to REF] [--registry]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = ("deploy/model", "metrics/classifier.json")
CLASSIFIER, GENERATOR = "synthguard-classifier", "synthguard-generator"
CHAMPION, PREVIOUS = "champion", "previous"


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def last_model_change(cwd: Path = ROOT) -> str:
    """Most recent commit (first-parent history) that changed the served model or its benchmark."""
    try:
        sha = git("log", "--first-parent", "-n1", "--format=%H", "--", *MODEL_PATHS, cwd=cwd)
    except subprocess.CalledProcessError:  # no commits yet
        sha = ""
    if not sha:
        raise SystemExit("no commit has changed deploy/model yet: nothing to roll back to")
    return sha


def served(cwd: Path = ROOT) -> dict:
    meta = json.loads((cwd / "deploy" / "model" / "meta.json").read_text())
    bench = json.loads((cwd / "metrics" / "classifier.json").read_text())
    clf = meta.get("classifier", {})
    return {"version": str(clf.get("version", "?")), "trained_on": clf.get("trained_on"),
            "roc_auc": bench.get("roc_auc")}  # fmt: skip


def git_rollback(to: str | None = None, cwd: Path = ROOT) -> dict:
    before = served(cwd)
    changed = None
    if to is None:
        changed = last_model_change(cwd)
        to = f"{changed}^1"
    target = git("rev-parse", "--verify", f"{to}^{{commit}}", cwd=cwd)
    git("checkout", target, "--", *MODEL_PATHS, cwd=cwd)
    after = served(cwd)
    return {"from": before, "to": after, "restored_from": target, "undone_commit": changed}


def report_markdown(result: dict) -> str:
    f, t = result["from"], result["to"]
    src = result["restored_from"][:7]
    undone = result["undone_commit"]
    why = f" (the state before `{undone[:7]}`)" if undone else ""
    lines = [
        "## Rollback",
        "",
        "| | Served now | After rollback |",
        "|---|---|---|",
        f"| classifier version | v{f['version']} | v{t['version']} |",
        f"| trained on | {f['trained_on']} | {t['trained_on']} |",
        f"| hold-out ROC-AUC (benchmark) | {f['roc_auc']:.4f} | {t['roc_auc']:.4f} |",
        "",
        f"Restored `deploy/model/` and `metrics/classifier.json` from `{src}`{why}.",
        "Merging this deploys it through deploy.yml (tests + eval gate, then Render).",
    ]
    return "\n".join(lines) + "\n"


def registry_rollback() -> dict:
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from synthguard.tracking.mlflow_utils import setup_mlflow

    def version_for(client, name, alias):
        try:
            return client.get_model_version_by_alias(name, alias)
        except MlflowException:
            return None

    setup_mlflow()
    client = MlflowClient()
    moved = {}
    for name in (CLASSIFIER, GENERATOR):
        cur, prev = version_for(client, name, CHAMPION), version_for(client, name, PREVIOUS)
        if prev is None:
            if name == CLASSIFIER:
                raise SystemExit(f"no {name}@{PREVIOUS}: nothing to roll back to")
            continue
        client.set_registered_model_alias(name, CHAMPION, prev.version)
        client.delete_registered_model_alias(name, PREVIOUS)
        if cur is not None:
            client.set_model_version_tag(name, cur.version, "promotion", "rolled_back")
        moved[name] = {"from": None if cur is None else str(cur.version), "to": str(prev.version)}
    return moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", help="git ref to restore (default: before the last model change)")
    ap.add_argument("--registry", action="store_true", help="move MLflow aliases instead of git")
    ap.add_argument("--report", default=str(ROOT / "reports" / "rollback.md"))
    args = ap.parse_args(argv)
    if args.registry:
        for name, m in registry_rollback().items():
            print(f"{name}@champion: v{m['from']} -> v{m['to']}")
        print("now re-export: python scripts/export_models.py")
        return 0
    result = git_rollback(args.to)
    text = report_markdown(result)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
