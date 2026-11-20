import json

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from synthguard.tracking import export
from synthguard.tracking.mlflow_utils import start_candidate_run


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """A temp MLflow store with two runs and one registered model version (+ alias, tags)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    run_ids = []
    for name, score in [("marginal", 0.5), ("ctgan", 0.9)]:
        with start_candidate_run(name, role="baseline", stage="test") as run:
            mlflow.log_param("seed", 42)
            mlflow.log_metric("quality_score", score)
            mlflow.set_tag("notes", f"{export.PROJECT_ROOT}/data/synthetic/{name}.csv")
            run_ids.append(run.info.run_id)
    client = MlflowClient()
    client.create_registered_model("synthguard-generator")
    mv = client.create_model_version("synthguard-generator", "file:///tmp/x", run_id=run_ids[1])
    client.set_model_version_tag("synthguard-generator", mv.version, "generator", "ctgan")
    client.set_registered_model_alias("synthguard-generator", "champion", mv.version)
    return tmp_path, run_ids


def test_export_writes_runs_and_models(populated):
    out, run_ids = populated
    export.main(out_dir=out / "submission")

    runs = pd.read_csv(out / "submission" / "mlflow_runs.csv")
    assert set(runs["run_id"]) == set(run_ids)
    assert {"params.seed", "metrics.quality_score", "tags.generator", "tags.git_commit"} <= set(
        runs.columns
    )
    assert runs.iloc[0]["tags.mlflow.runName"] == "ctgan"  # newest first

    models = json.loads((out / "submission" / "registered_models.json").read_text())
    assert models[0]["name"] == "synthguard-generator"
    v = models[0]["versions"][0]
    assert v["run_id"] == run_ids[1]
    assert v["aliases"] == ["champion"]
    assert v["tags"] == {"generator": "ctgan"}


def test_export_contains_no_local_details(populated):
    out, _ = populated
    export.main(out_dir=out / "submission")
    text = (out / "submission" / "mlflow_runs.csv").read_text()
    text += (out / "submission" / "registered_models.json").read_text()
    assert str(export.PROJECT_ROOT) not in text
    assert "tags.mlflow.user" not in text and "artifact_uri" not in text
    assert "./data/synthetic/ctgan.csv" in text  # project paths become relative
    assert "file:///tmp/x" not in text  # model source location dropped
