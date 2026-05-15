# [forge][Mission B] PR webhook + dispatcher — trigger guardian on PR open / push

Refs RFC #TBD-RFC. Acceptance test: #TBD-11 (must BLOCK #3204/#3205/#3206).

## Goal

When a PR is opened or pushed against `ROCm/aiter`, automatically launch the forge guardian pipeline with the PR's metadata (number, head SHA, body, file diff, labels, author).

## Approach

Two layers:

1. **GitHub Actions workflow** at `.github/workflows/forge-guardian.yml`, triggered on `pull_request: [opened, synchronize, ready_for_review]`. The workflow checks out the PR head and invokes `forge guardian dispatch --pr <num>`.

2. **`forge.guardian.dispatcher` module** under `forge/src/aiter_forge/guardian/dispatcher.py`. Responsible for:
   - Pulling PR metadata via `gh api`.
   - Determining whether the PR carries a perf claim (heuristic: title/body/commits contain `perf|kernel|tuning|MFU|MBU|tok/s|speedup|GBps|TFLOPS` + the PR touches a perf-relevant file glob).
   - If yes: hand off to claim-extraction (issue #TBD-07).
   - If no: emit `gate-passed` immediately with reason "no perf claim detected."

## Scope

- Workflow YAML + dispatcher module + minimal log output.
- Reuse existing self-hosted `linux-aiter-forge-mi355-{1,8}` runners.
- Run advisory-only for the first two weeks (label only, no maintainer-side gating).

## Out of scope

- Claim extraction itself (issue #TBD-07).
- Verdict labeling (issue #TBD-09) — dispatcher only sets a draft label, verdict engine finalizes.
- Cross-repo guardian (e.g. ATOM, vLLM-rocm) — AITER-only for MVP.

## Definition of done

- Workflow file exists and runs on every PR open/push.
- `forge guardian dispatch --pr <num>` works locally and emits a structured JSON status.
- Five real PRs (open at the time of MVP) get processed end-to-end.
- The acceptance test issue (#TBD-11) confirms guardian runs against #3204/#3205/#3206.

## Owner

forge maintainers + one CI engineer.

## Estimated effort

M — one workflow + ~200 LOC dispatcher.
