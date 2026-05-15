# Agent rules for forge/ (inside AITER)

`forge/` is AITER's first-party AI-agent toolkit. Two missions, one set of rules.

## Missions

### Mission A — Kernel tuning harness

Provide AITER engineers with a repeatable, evidence-based AI-agent workflow for kernel optimization on AMD MI-series. Inputs: a `targets/<name>/target.yaml`. Outputs: tuned configs, dispatcher patches, FlyDSL/Triton/Opus kernel candidates, and one `learnings/` entry per completed experiment.

Standard 5-agent roster (see `agents/`):
- `bench-runner` — runs the perf benchmark against a target, collects rocprof + tok/s
- `kernel-writer` — generates kernel-source variants under a hypothesis
- `perf-analyzer` — reads rocprof traces, extracts the bottleneck claim
- `ir-inspector` — folds source ↔ HW-resource bundle for verification
- `learning-extractor` — emits the 4-section `learnings/*.md` entry

Two specialized roster members:
- `dashboard-refresh` — Mission A supporting ritual; re-measures `ROCm/AI-Frameworks-Dashboard` op-perf rows
- `flydsl-cleanup` — applies the FlyDSL-internal-types-cleanup skill to existing kernels

### Mission B — Guardian gate for incoming PRs

On every incoming AITER PR carrying a performance claim, guardian:

1. Extracts the claim(s) from PR body / commits
2. Binds each claim to one or more executable tests (rocprof kernel-name check, `benchmark_serving` tok/s diff, allocator-trace, dispatcher-lookup hit, etc.)
3. Runs the bound tests on the self-hosted MI-series runners
4. Emits a verdict label on the PR: `gate-passed`, `needs-evidence`, or `gate-blocked`
5. Surfaces a churn signal when the same author + file region was opened/closed multiple times in a short window

Maintainers can override any verdict — guardian raises the floor on evidence, not the ceiling on judgment.

See `proposals/02-mission-b-guardian/` for the per-feature design.

## Hard rules (both missions)

### Branch protection

**NEVER push directly to `main` or `master`.** Every change goes through a PR.

```bash
git checkout -b <type>/<short-description>
git add -p && git commit -m "..."
git push -u origin HEAD
gh pr create --fill
```

Enforcement layers active in `forge/`:
1. Local pre-push hook (`forge/.githooks/pre-push`) — activate once with `git config core.hooksPath forge/.githooks`.
2. Repository-level branch protection on `ROCm/aiter` `main` (server-side, separate from forge).

Force pushes to `main`/`master` are always blocked.

### Evidence-based workflow

- Experiments are `tasks/*.yaml` with a required non-empty `hypothesis` field. No hypothesis, no run. See `src/aiter_forge/predict_verify.py`.
- Every completed experiment appends one entry to `learnings/` with the four required sections (Hypothesis / Result / Root cause / Reusable rule) plus metadata. Missing sections fail CI.
- Complexity is a hard gate (`src/aiter_forge/complexity.py`): per-file LOC, env-flag count, branches-per-function. PRs exceeding limits fail CI.

### Boundary with AITER source

forge can edit AITER source — it is part of AITER now. Two etiquette rules nonetheless:

- **Tuning configs and dispatcher patches** are routine and welcome.
- **Kernel-source rewrites** still go through the regular AITER review pipeline. forge is a harness for engineers, not a substitute for kernel review.

The previous standalone-repo rule "never patches AITER source" no longer applies. forge sits inside AITER.

## Agent execution policy

- When told to "finish", "continue", or "go to finishline": execute ALL remaining tasks to completion.
- If a planning agent produces an implementation plan, immediately start implementing it (use parallel sub-agents where possible). Do not wait for user confirmation.
- Only pause for genuine blockers requiring user input (ambiguous design choice, missing credentials, etc.), not risk aversion.
- Always launch independent tasks as parallel sub-agents.

## Tooling (enforced in CI)

- `aiter_forge.complexity` — quantified "simple enough." Operationalizes the simplicity hard-gate.
- `aiter_forge.predict_verify` — experiments are `tasks/*.yaml` with required `hypothesis`. Tasks without a hypothesis cannot be parsed or run.
- `aiter_forge.learnings` — every entry under `learnings/` is validated for the four sections plus metadata.

See `docs/architecture.md` for full restructure rationale and agent lifecycle.

## PR etiquette (for changes to forge/ itself)

- Use `gh pr create --fill` plus a Test Plan section.
- Title ≤ 70 chars. Body explains WHY, not WHAT (the diff shows WHAT).
- Self-review with a checklist before requesting review.
- For changes to guardian rules or claim-extraction prompts: include at least one negative-test PR snapshot in the Test Plan.
