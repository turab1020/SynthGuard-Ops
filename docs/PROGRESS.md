# Progress log

Source of truth for task status from Claude Code sessions. Turab syncs the team trackers from this.
Status: To do / In progress / Blocked / Done.

| ID | Owner | Task | Status | Result / notes | Date |
|---|---|---|---|---|---|
| T01 | Turab | GitHub repo + collaborators (manual) | Done | Initial commit pushed to main; repo-level identity set to Turab's noreply; attribution check OK | 2026-11-17 |
| T02 | Turab | Repo scaffold | Done | Package, Makefile, ruff/pre-commit, params.yaml, config.py; lint + 4 tests green; requirements.lock (262 pins) | 2026-11-17 |
| T03 | Turab | DVC + MLflow init | Done | dvc init + local remote; mlflow_utils with tags; dummy run visible in UI (port 5001); 6 tests green | 2026-11-17 |
| T04 | Nufil | Download raw data + EDA | In progress | Built + tested; handoff bundle ready (handoff/nufil/T04-download.zip); waiting for Nufil's push on nufil/data | 2026-11-17 |
| T05 | Nufil | Validation stage (pandera) | In progress | Built + tested (corruption demo: 2 errors caught); bundle handoff/nufil/T05-validate.zip; waiting for Nufil | 2026-11-17 |
| T06 | Nufil | Prepare stage + leakage-free split | In progress | Built + tested: 24 dups removed, 22,775/9,762 split, >50K 0.2409 both; bundle handoff/nufil/T06-prepare.zip; waiting for Nufil | 2026-11-17 |
| T07 | Faizan | Metrics spec | In progress | docs/metrics.md written; bundle handoff/faizan/T07-metrics-spec.zip; waiting for Faizan's push | 2026-11-17 |
| T08 | Nufil | Generator wrapper + baselines | To do | | |
| T09 | Nufil | CTGAN + TVAE | To do | | |
| T10 | Faizan | Fidelity module | To do | | |
| T11 | Faizan | Utility module | To do | | |
| T12 | Turab | Dockerfile | To do | | |
| T13 | Faizan | Privacy module | To do | | |
| T14 | Faizan | Evaluate stage + score | To do | | |
| T15 | Nufil | Params-driven generate + seeds | To do | | |
| T16 | Turab | Register top generator | To do | | |
| T17 | All | MILESTONE: end-to-end pipeline | To do | | |
| T18 | Turab | One-command reproduction | To do | | |
| T19 | Turab | GitHub Actions CI | To do | | |
| T20 | Nufil | Data/generator tests | To do | | |
| T21 | Faizan | Metric tests | To do | | |
| T22 | Nufil | Fresh-clone reproducibility check | To do | | |
| T23 | Faizan | Results table | To do | | |
| T24 | Turab | Submission evidence + README | To do | | |
| T25 | All | Buffer: fix + merge PRs | To do | | |
| T26 | Turab | Submit on Classroom (manual) | To do | | |
