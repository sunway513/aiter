# forge/ project rules

`forge/` is AITER's first-party AI-agent toolkit. Two missions: kernel tuning harness for AITER engineers (Mission A) and guardian gate for incoming PRs (Mission B). See `AGENTS.md` for the full agent-execution policy.

## Branch protection (hard rule)

- **NEVER push directly to `ROCm/aiter:main`.** All changes — including version bumps, doc edits, CI config — go through a PR.
- Workflow: `git checkout -b <type>/<desc>` → commit → `git push -u origin HEAD` → `gh pr create --fill`.
- Force pushing `main` is ALWAYS forbidden.
- Local enforcement: `git config core.hooksPath forge/.githooks` after clone.

## Evidence-based workflow (hard rule)

- Experiments are `tasks/*.yaml` with required non-empty `hypothesis`. No hypothesis, no run. See `src/aiter_forge/predict_verify.py`.
- Every completed experiment appends one entry to `learnings/` with the 4 required sections plus metadata. See `src/aiter_forge/learnings.py`.
- Complexity is a hard CI gate via `src/aiter_forge/complexity.py`.
- Full rationale in `docs/architecture.md`.

## Execution policy

- When told to "finish", "continue", or "go to finishline": execute ALL remaining tasks to completion. Never treat tasks as "follow-up" or "next session" work.
- Launch independent tasks as parallel sub-agents immediately.
- Only pause for genuine blockers requiring user input (ambiguous design choice, missing credentials, etc.), not risk aversion.

## Kernel development environment

- Containers run on AMD MI355X (gfx950) and MI300X (gfx942) self-hosted runners (`linux-aiter-forge-mi355-{1,8}`).
- AITER source: at the repo root, two levels up from this file (`../`).
- Workspace: `forge/` is the working directory for harness operations.
- Models: shared mount at `/data/models/` on tuning hosts.

## Mission B (guardian) specifics

- Guardian verdicts (`gate-passed`, `needs-evidence`, `gate-blocked`) attach as PR labels.
- Guardian never hard-rejects. Maintainers can override any verdict.
- Negative-test corpus lives in `proposals/03-acceptance-test/` and CI test fixtures under `tests/guardian/`.
- New rules ship with at least one PR snapshot proving the rule fires correctly.
