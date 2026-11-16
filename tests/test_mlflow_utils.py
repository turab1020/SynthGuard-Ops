import mlflow

from synthguard.tracking import mlflow_utils as mu


def test_candidate_run_logs_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    with mu.start_candidate_run("dummy", role="baseline", stage="test") as run:
        mlflow.log_param("seed", 42)
        run_id = run.info.run_id
    # resume the same run to add a metric, as the evaluate stage will
    with mu.start_candidate_run("dummy", role="baseline", stage="test", run_id=run_id):
        mlflow.log_metric("quality_score", 0.5)
    got = mlflow.get_run(run_id)
    assert got.data.tags["generator"] == "dummy"
    assert got.data.tags["role"] == "baseline"
    assert len(got.data.tags["git_commit"]) == 40
    assert "data_train_md5" in got.data.tags and "dvc_lock_md5" in got.data.tags
    assert got.data.metrics["quality_score"] == 0.5


def test_file_md5(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("abc")
    assert mu.file_md5(f) == "900150983cd24fb0d6963f7d28e17f72"
