# Repository setup (T01)

Repo: **`turab1020/SynthGuard-Ops`** (created; Faizan and Nufil have accepted their invites) · Owner: Turab · Visibility: Private

## A. Turab, on github.com
1. **Create the repo:** + → New repository → name `synthguard-ops`, description
   "Quality-gated synthetic tabular data pipeline (MLOps course project)", **Private**.
   Leave *Add README*, *.gitignore* and *license* **unticked** (the repo must be empty).
2. **Copy the HTTPS URL:** `https://github.com/turab1020/SynthGuard-Ops.git`.
3. **Add collaborators:** Settings → Collaborators → Add people → Faizan's and Nufil's
   GitHub usernames, role *Write*. They accept from email or github.com/notifications.
4. **Protect `main`:** Settings → Branches → Add rule → pattern `main` →
   tick *Require a pull request before merging*. (Private repos on free accounts can't enforce
   this; the GitHub Student Developer Pack gives Pro for free. If unavailable, skip — the
   branch + PR workflow is still followed by convention.)
5. **Teammates' commits:** Faizan and Nufil commit their own parts from their own accounts using
   the handoff bundles Claude Code prepares (section E), so they are the sole authors of their work.
6. **Before submitting (Sat):** add the instructor as a collaborator (read access is enough), or
   confirm with them how they want access to a private repo.

## B. Turab's Mac (VS Code terminal), once — Claude Code may do all of this itself
```bash
git config --global user.name   # must show Turab's name
git config --global user.email  # must be an email on Turab's GitHub account
```
Set them if empty. Sign in to GitHub in VS Code (Accounts menu) so `git push` can authenticate.
Optional: `brew install gh && gh auth login` lets Claude Code open PRs with `gh pr create`.

## C. Claude Code, after Turab gives the URL
```bash
git init -b main
# write .gitignore before anything else (venv, data/, models/, mlflow.db, mlruns/, mlartifacts/,
# .env, .claude/settings.local.json, handoff/, __pycache__/, .DS_Store)
git add . && git commit -m "chore: initial project scaffold and docs"
git remote add origin https://github.com/turab1020/SynthGuard-Ops.git
git push -u origin main
```
Turab's tasks: `git switch -c turab/<task>` → commits → push → PR → merge → `git switch main && git pull`.
Faizan's / Nufil's tasks: Claude Code prepares `handoff/<owner>/<TXX>/` (+ .zip, never committed) →
Turab sends it → the owner commits it from their own account (section E) → PR → merge →
`git switch main && git pull`.

## D. Attribution check before every push
```bash
git log origin/main..HEAD --format='%an <%ae>%n%b' | grep -iE 'claude|anthropic|generated with' \
  && echo "STOP: AI attribution found" || echo "OK"
```

## E. Teammates committing their handoff bundle (any browser, no laptop needed)
Turab sends the owner the bundle zip (WhatsApp/Drive). Unzip it (phones can too).
1. Open https://github.com/turab1020/SynthGuard-Ops and press **`.`** → github.dev opens.
   Make sure you're signed in with **your own** GitHub account.
2. Bottom-left branch name → **Create new branch** → use the name in `HANDOFF.md`.
3. Drag the bundle's folders into the Explorer so the paths match (skip `HANDOFF.md`).
   Read each file — you'll be asked about your part.
4. Source Control panel → paste the commit message from `HANDOFF.md` → **Commit & Push**.
5. github.com → **Compare & pull request** → paste title/description from `HANDOFF.md` → Create
   → wait for CI → **Merge** → tell Turab.
No github.dev? On github.com: switch to your branch → **Add file → Upload files** → drag the
folders (paths are kept) → commit. Any computer with git also works.
