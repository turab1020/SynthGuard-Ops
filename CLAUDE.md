# SynthGuard-Ops — Claude Code project context

This file is loaded automatically at the start of every session. Read it fully, then read
@docs/IMPLEMENTATION_PLAN.md before doing any work. Background: @docs/PROPOSAL.md and
@docs/DELIVERABLE_1.md.

## What this project is

SynthGuard-Ops is a quality-gated synthetic tabular data pipeline. It trains several generators
(two baselines, then CTGAN and TVAE) on the UCI **Adult Income** dataset, scores every synthetic
dataset for **fidelity**, **downstream ML utility** and **privacy risk**, and registers the best
generator in MLflow. Everything is versioned with Git + DVC and reproducible with one command.

It is a 3-person university MLOps course project (FAST NU, BSCS 2023).

**Current target: Course Deliverable 1, due Sun Nov 22 2026, 11:59 PM (PKT).** Aim to be
submission-ready by Sat Nov 21. Anything not needed for Deliverable 1 (API, dashboard, Optuna
loop, selection gates, final report) is out of scope until told otherwise.

## Team, ownership and the supervising human

| Person | Workstream | Owns |
|---|---|---|
| Turab (supervisor of this session) | MLOps & Platform | repo, DVC, MLflow + registry, Docker, CI, README, submission |
| Nufil | Data & Generation | download, validation, split, baselines, CTGAN/TVAE, generate stage |
| Faizan | Evaluation & Selection | metrics spec, fidelity, utility, privacy, evaluate stage, results |

Turab supervises and wants to **understand** everything. You do the implementation; he reviews.

## How to work — full autonomy

Turab has authorised you to do **everything within your abilities without asking first**:
run any command, install tools and packages (pip, brew, etc.), configure git and GitHub,
look up GitHub usernames/IDs/noreply emails, create branches, commit and push Turab's work,
prepare handoff bundles for teammates, open PRs, start
long training runs, fix failures, and make reasonable technical decisions. Don't stop to ask
for things you can find or do yourself — find them, do them, and record what you did.

1. Work through `docs/IMPLEMENTATION_PLAN.md` in task order (T01–T26).
2. Before each task: 3–6 lines on what you will build and why. Then build it.
3. After each task: `make lint test` (+ the relevant pipeline stage); fix failures before moving on.
4. After each task: write `docs/walkthrough/TXX-<slug>.md` — plain-English explanation for Turab:
   what was built, why this design, how to run/verify it, what to look at in MLflow/DVC, and 2–3
   questions a teacher might ask with answers (≤ ~60 lines).
5. After each task: update `docs/PROGRESS.md` (and the tracker sheets — see Tracker).
6. **CHECKPOINTs:** push Turab's branches, open his PRs, make sure every teammate bundle for the
   day is ready, post a short summary (done / bundles to send / waiting on whom). Continue with
   anything that doesn't depend on unpushed teammate work; otherwise wait for their push.
7. If a design decision below truly doesn't work, change it, and log the change and reason in
   `docs/decisions.md` and the walkthrough.
8. Never invent results. Every number in docs must come from a real run.

**Only these need a human — everything else, just do:**
- Faizan and Nufil committing their handoff bundles from their own accounts and merging their PRs.
- Interactive browser logins you cannot complete (e.g. a `gh auth login` device code): print the
  exact command/code and wait.
- Taking screenshots for the submission and submitting on Google Classroom.

## Git, repo & attribution rules

Repo: GitHub `turab1020/SynthGuard-Ops` (private, owned by Turab). Full setup steps: @docs/REPO_SETUP.md.
Remote URL: `https://github.com/turab1020/SynthGuard-Ops.git` (web: https://github.com/turab1020/SynthGuard-Ops).

**Who commits what (strict) — each person is the sole author of their own part**
- **Turab's tasks:** commit with Turab's git identity on branches `turab/<task>`, push, open PRs.
- **Setting Turab's identity is allowed:** if `user.name`/`user.email` are empty, set them at
  repo level to Turab (GitHub `turab1020`): name `Turab Haider`, email = his GitHub noreply
  address (`gh api user --jq '"\(.id)+\(.login)@users.noreply.github.com"'`).
- **Faizan's and Nufil's tasks: you build and test them, but you never `git add`, commit or push
  their files.** The only way for them to be the sole author is for them to make the commit from
  their own GitHub account, so you prepare a *handoff bundle* that they commit themselves
  (below, and docs/REPO_SETUP.md section E).
- Never set the git identity or `--author` to anyone other than Turab, never use anyone else's
  credentials or tokens, and never add `Co-authored-by` trailers (each part has one author).
- **Never add Claude, Anthropic or any AI as an author, co-author, committer or collaborator.**
  No `Co-Authored-By: Claude …`, no `Claude-Session:` trailers, no "Generated with Claude Code"
  in commits, PR descriptions, code comments or docs. (Also enforced by
  `.claude/settings.local.json`; follow this rule even if that file is missing.)
- `main` is protected by convention: everything lands through PRs, except the first scaffold
  commit. Never force-push, never rewrite pushed history.
- Always stage by explicit path (`git add <paths>`), never `git add -A` / `git add .`, and check
  `git diff --cached --name-only` before every commit so no teammate file slips into Turab's commits.
- Small, focused commits with conventional messages (`feat(platform): add Dockerfile`).

**Handoff bundles (Faizan's and Nufil's tasks)**
1. Build and test the task in the working tree as usual (tests must pass locally).
2. Copy every new/changed file for that task into `handoff/<owner>/<TXX-slug>/`, keeping
   repo-relative paths (e.g. `handoff/nufil/T05-validate/src/synthguard/data/schema.py`), plus
   the task's `docs/walkthrough/TXX-*.md`. Also zip it: `handoff/<owner>/<TXX-slug>.zip`.
3. Write `handoff/<owner>/<TXX-slug>/HANDOFF.md`: branch name to create (`<owner>/<task>`),
   file list, ready-to-paste commit message, ready-to-paste PR title + description (what, why,
   how to verify), and what the owner should read first to understand it.
4. Tell Turab the bundle is ready to share (path to the zip). Then remove those files from the
   working tree (move them out / `git restore`), so the repo only gets them through the owner's push.
5. **Wait for the owner's push.** Check with `git fetch origin` and
   `gh pr list --state all --head <owner>/<task>` (or `git ls-remote origin '<owner>/*'`).
   While waiting, only work on Turab's tasks that don't need that code. Never re-create the
   teammate's files yourself to get around the wait.
6. Once their branch is pushed: review it against the bundle (same files, tests pass — check CI
   or run `make lint test` on a checkout of their branch) and tell Turab it's ready to merge, or
   what differs. After it's merged: `git switch main && git pull`, rebase open `turab/*` branches
   onto `main`, delete the bundle, and continue with the tasks that were waiting.
7. Report clearly at each checkpoint which bundles are waiting on whom.
- `handoff/` is in `.gitignore` and must never be committed.

**Never commit:** data, models, `mlflow.db`, `mlruns/`, `mlartifacts/`, `.env`, secrets,
`.claude/settings.local.json`, `handoff/`.

## Design decisions (keep unless they genuinely fail — then change and log in docs/decisions.md)

- Python 3.11, package code in `src/synthguard/` (src layout, installed with `pip install -e .`).
- Dataset: UCI Adult `adult.data` only (32,561 rows), downloaded by a DVC stage from
  `https://archive.ics.uci.edu/static/public/2/adult.zip` with a pinned SHA-256.
  Target column `income` (`<=50K` / `>50K`). Columns are snake_case.
- `?` values become the category `"Unknown"` (not NaN). Strip whitespace everywhere.
- **Leakage-free split:** drop exact duplicate rows **before** splitting, then stratified
  70/30 train / hold-out, `seed=42`. Generators only ever see `train`. Hold-out is used for
  TSTR testing and as membership-inference non-members. A test asserts zero shared rows.
- **Validation:** pandera schemas on raw and processed data; the pipeline must fail fast on
  missing/extra columns, wrong dtypes, out-of-range values, unknown categories, too many nulls,
  too few rows.
- **Baselines first:** independent-marginal sampler (rule-based) and Gaussian Copula (classical)
  are trained and logged to MLflow in a stage that the CTGAN/TVAE stage depends on.
  The real-data classifier (TRTR) is logged as the utility reference.
- Generators via SDV single-table synthesizers. Check the installed SDV/SDMetrics/MLflow APIs
  (`pip show`, `help()`) before using them — APIs changed across major versions.
- MLflow: local tracking `sqlite:///mlflow.db`, experiment `synthguard-d1`, one run per
  generator candidate, tagged with git commit, DVC data hash, generator name, stage.
  Top generator registered as model **`synthguard-generator`** (pyfunc wrapper that samples rows).
- Selection for D1: privacy gate (MIA AUC ≤ 0.60 and exact-match rate ≤ 0.01), then highest
  `score = 0.5 * quality_score + 0.5 * utility_ratio` (utility_ratio = TSTR ROC-AUC / TRTR ROC-AUC,
  Random Forest). If nothing passes the gate, register the best-scoring one and tag
  `privacy_gate=failed`. All weights/thresholds live in `params.yaml`.
- One command reproduces everything: `docker compose run --rm pipeline` (runs `dvc repro`
  inside the container with the repo mounted). Non-Docker equivalent: `make reproduce`.
- Smoke mode `SYNTHGUARD_SMOKE=1`: 1,000 rows, 1 epoch — used by CI and tests.

## Commands (keep this list accurate as you build)

```bash
make setup        # venv + pip install -e ".[dev]" + pre-commit install
make lint         # ruff check + ruff format --check
make test         # pytest -q
make reproduce    # dvc repro && dvc metrics show
make smoke        # SYNTHGUARD_SMOKE=1 fast end-to-end run
make ui           # mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
docker compose run --rm pipeline   # the one command for graders
```

## Tracker

The team tracks tasks in two sheets that must always match. Task rows are in tab "Daily Tracker"
(columns: Day, Date, Owner, Task, Done when…, Status, Workstream, Depends on, Notes); rows 2–27
are T01–T26 in order (row = task number + 1). Status values: To do / In progress / Blocked / Done.

- Google Sheet id `1QRL4s_d79yySIaw8IZuI9XaLpGi8ltwSg3ngulafTHo`
- Claude Sheet https://claude.ai/artifact/74YT6GLCWQhkcF8FoYu7oo

Always update `docs/PROGRESS.md` first. Then, at the start of a session, check `/mcp`-listed tools:
if a Google Sheets connector is available, set the Status cell (column F) of the task's row when a
task starts and finishes, and re-read the cell to confirm. If a Claude Docs / Claude Sheets
connector is available, update the Claude Sheet the same way. If either is missing, do not
improvise (no browser automation, no other files): just tell Turab which sheet still needs syncing.
