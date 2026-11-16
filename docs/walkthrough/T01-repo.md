# T01 — GitHub repo wiring

**What was built:** local git repo (`main`), `.gitignore` written before anything else, first
commit `chore: initial project scaffold and docs`, remote `origin` → turab1020/SynthGuard-Ops, pushed.

**Why this way:** the `.gitignore` goes in first so data, models, MLflow files, secrets,
`.claude/settings.local.json` and `handoff/` can never be committed by accident.
Git identity is set at *repo* level (not global) to Turab Haider with the GitHub noreply email
`153242951+turab1020@users.noreply.github.com`, so commits link to the profile without exposing a private address.

**Known issue:** the very first commit has a malformed author email (a scripting bug inserted a space).
It is already pushed, and history is never rewritten, so it stays; all later commits are correct.

**Verify:**
```bash
git config user.name; git config user.email
git log --format='%an <%ae>' | head
git log origin/main --format='%an <%ae>%n%b' | grep -iE 'claude|anthropic|generated with' || echo OK
```

**Teacher questions**
- *Why ignore `data/` in git?* Large/binary files belong in DVC; git only stores the small `.dvc`/`dvc.lock` pointers.
- *Why a noreply email?* GitHub attributes the commit to the account without publishing a personal email.
- *Why not fix the first commit?* Rewriting pushed history breaks everyone's clones; a cosmetic author typo isn't worth it.
