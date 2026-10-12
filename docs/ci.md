# CI and the model check (module 8)

Every push and pull request runs `.github/workflows/ci.yml`: three jobs, about 4 minutes, all required to pass.

| Job | Checks | Fails when |
|---|---|---|
| **Lint, types, tests, audit** | `uv lock --check`, ruff (lint + format), mypy (strict), pytest (132 tests), pip-audit | the lockfile is stale, code is unformatted or mistyped, a test fails, or a dependency has a known vulnerability |
| **Model check** | downloads and validates the data, then cross-validates the model against `ci/model_baseline.json` | the model got worse (below) |
| **API image** | builds `infra/api/Dockerfile`, scans it with Trivy, smoke-tests it | a fixable HIGH/CRITICAL vulnerability, a failed import, running as root, or training libraries in the image |

## Two gates, two jobs

| | CI model check | Promotion gate (module 6) |
|---|---|---|
| Runs | every push / pull request | once per new model version |
| Data | 5-fold cross-validation on the **training set** | the **test set**, once |
| Question | "did this code change make the model worse?" | "should this version replace the champion?" |

**CI never uses the test set.** CI runs on every change, so test-set checks would mean hundreds of looks at the same
1,409 customers, slowly tuning the code to them. That's the module 5 rule ("look once") broken without anyone
noticing.

**How the check works.** The training code is deterministic (fixed seeds), so unchanged code reproduces the baseline
exactly (PR-AUC 0.6617). The check fails if PR-AUC drops more than **0.005**, profit@10% drops more than **3%**, the
model stops beating the B2 rule on every fold, or the training data differs from the baseline's (the numbers
wouldn't be comparable). An improvement beyond the tolerance passes and asks for the baseline to be updated
**in the same pull request** (`--update-baseline`), so the bar only moves up, through review.

**Tested by breaking it on purpose.** Dropping the model's strongest feature (`tenure`) lowers PR-AUC to 0.6514. The
check fails (floor 0.6567), and the change can't slip in quietly.

## Supply-chain safety

- **Actions pinned to commit SHAs**, not tags. A tag can be moved to point at different code, and hijacked action
  tags are a real attack route. The tag stays in a comment for humans.
- **Read-only token** (`permissions: contents: read`): no job can write to the repository.
- **Dependency cooldown:** `exclude-newer = "14 days"` in `pyproject.toml` (rolling: the lockfile records the span,
  and versions only change when someone upgrades). **Dependabot** (`.github/dependabot.yml`) opens weekly update pull
  requests for Python packages, GitHub Actions and Docker base images, with the same 14-day cooldown, each one
  run through this CI.
- **pip-audit** checks every locked package (with hashes) against known vulnerabilities. Its first run found
  `oauthlib 3.3.1` (PYSEC-2026-4114, a timing attack in OAuth PKCE checks), reached through prefect → apprise and not
  in the API image. Upgraded to 4.0.0, the fixed release, once it was 14 days old.
- **Trivy** scans the API image for fixable HIGH/CRITICAL vulnerabilities. First run: 0.

## Not (yet) enforced by GitHub

CI reports a red or green status, but GitHub doesn't *block* anything until **branch protection** is enabled for
`main` (Settings → Branches). Required checks and pull requests become mandatory, and direct pushes to `main` are
refused. That's a repository setting, not code. It is worth enabling once work moves to pull requests.
