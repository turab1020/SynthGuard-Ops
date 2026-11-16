# T03 — DVC + MLflow init

**What was built**
- `dvc init` → `.dvc/` + `.dvcignore`. Default remote `localremote` → `../synthguard-dvc-remote`
  (a sibling folder of the repo). `core.autostage` on, so DVC `git add`s its own pointer files.
- `src/synthguard/tracking/mlflow_utils.py`:
  - `get_tracking_uri()` → `MLFLOW_TRACKING_URI` or `sqlite:///<repo>/mlflow.db`
  - `start_candidate_run(generator, role, stage, run_id=None)` → one run per candidate; passing
    `run_id` *resumes* it, which is how `evaluate` will add metrics to the training run.
  - `common_tags()` → `git_commit`, `data_train_md5`, `data_holdout_md5`, `dvc_lock_md5`.
- `tests/test_mlflow_utils.py` (uses a temp SQLite DB, so tests never touch the real one).

**Why:** SQLite is a real backend (needed for the Model Registry in T16), not just files.
The md5 tags tie every run to the exact data and pipeline version DVC produced.

**Verify**
```bash
make lint test          # 6 tests pass
make ui                 # http://localhost:5001 → experiment synthguard-d1, run dummy_t03
.venv/bin/dvc status    # "no data or pipelines tracked yet" until T04
```
Port 5001, not 5000: macOS AirPlay already uses 5000 (see docs/decisions.md).

**Teacher questions**
- *Why tag runs with file md5s?* Proves which data version trained a model; DVC can check that version out again.
- *Why can a run be resumed?* Generators train in one stage and are scored later; resuming keeps
  params, timings and metrics for a candidate in a single run.
- *Why is the remote outside the repo?* A DVC remote is storage for cached data, never part of git history.
