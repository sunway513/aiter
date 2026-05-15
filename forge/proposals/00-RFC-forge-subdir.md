# [RFC] Adopt forge/ subdir as AITER's first-party AI-agent harness (kernel tuning + PR guardian)

## Summary

Fold the existing `sunway513/aiter-forge` toolkit into `ROCm/aiter` as a first-party subdirectory at `aiter/forge/`. Re-scope its mission from a standalone perf-tracking project into a dual-purpose AITER-internal toolkit:

- **Mission A — Kernel tuning harness for AITER engineers**: standardized AI-agent workflow (dispatcher, agent roster, predict-verify, learnings, complexity gate) for kernel optimization on AMD MI-series.
- **Mission B — Guardian gate for incoming AITER PRs**: claim extraction, claim-to-test binding, verdict engine, and churn detector. Raises the evidence floor on incoming perf-claim PRs without raising the maintainer ceiling.

## Motivation

Two pressures converged:

1. **Internal**: AITER kernel work needs a repeatable AI-assisted optimization workflow. Today engineers duplicate tuning scaffolding per project. `aiter-forge` already implements the full machinery (dispatcher, 5-agent roster, predict-verify, learnings, complexity gate) but lives in a personal namespace with no AITER-side adoption path.

2. **External**: AI lowered the cost of opening a PR. Recent perf-claim PR series (e.g. #3197, #3199 closed and re-opened as #3204/#3205/#3206 against RFC #3202) show the pattern: long well-formatted RFC body, multiple cosmetic re-splits over two weeks, missing CI labels, perf claims without rocprof or `benchmark_serving` artifacts. Maintainer review time is the bottleneck. We need to shift the burden of evidence back to the submitter automatically.

Both pressures are solved by the same toolkit, applied in two modes.

## Proposal

### Where it lives

`ROCm/aiter/forge/` — folder, not submodule, not separate repo. Single PR drops the filtered payload from `sunway513/aiter-forge` (dropping `results/` 331MB benchmark archive and one-shot tuning round scripts).

Final size: ~1MB / ~108 files.

### Re-scoping

The previous "Priority 1 competitive perf health / Priority 2 CK removal" framing is replaced by the dual-mission framing above. The previous hard rule "never patches AITER source" no longer applies (forge IS part of AITER now), but kernel-source rewrites still go through normal AITER review etiquette.

### Acceptance criteria for the merge-in PR

1. `forge/` payload imports cleanly with `pip install -e ./forge[dev]`.
2. `forge/AGENTS.md`, `forge/CLAUDE.md`, `forge/README.md` reflect the dual mission.
3. Existing `forge/learnings/` complexity / hypothesis / 4-section CI gates still pass.
4. `forge/proposals/` contains the umbrella RFC plus 11 child issue drafts (this folder is removed in a follow-up after issues are filed against ROCm/aiter).
5. No changes to AITER source files outside `forge/`.

### Phase plan (target completion: end of weekend 2026-05-17)

- **Phase 0 (this PR)**: drop `forge/` payload, file this RFC + 11 child issues.
- **Phase 1 (week of 2026-05-18)**: complete Mission A — wire `forge/dispatcher` as standard tuning entry; port the 11 sub-project-2 Triton optimization targets from `sunway513/aiter-forge` issue queue.
- **Phase 2 (weeks of 2026-05-25 and 2026-06-01)**: build Mission B MVP — PR webhook, claim extraction, claim-to-test binding, verdict engine. Acceptance test: must produce BLOCK on PR #3204/#3205/#3206 (see `proposals/03-acceptance-test/11-must-block-3202.md`).
- **Phase 3 (week of 2026-06-08)**: enable guardian on `ROCm/aiter` PRs, advisory-only labels for two weeks, then `gate-blocked` becomes a maintainer-respected signal.

## Child issues

Mission A — Kernel tuning harness (5):
- Adopt forge/ as standard tuning entry for new AITER kernel work
- Port 11 sub-project-2 Triton optimization targets from sunway513/aiter-forge
- Engineer onboarding doc — how to launch forge against a target
- Wire forge sweeps to ROCm/AI-Frameworks-Dashboard staleness detection
- Standard 5-agent roster — port and document

Mission B — Guardian gate (5):
- PR webhook + dispatcher — trigger guardian on PR open / push
- Claim extraction — parse PR body / commits for perf assertions
- Claim-to-test binding library (rocprof / benchmark_serving / alloc-trace)
- Verdict engine — emit gate-passed / needs-evidence / gate-blocked labels
- Churn detector — author × file × time-window similarity

Acceptance test (1):
- Guardian MVP must BLOCK #3202 series — canonical negative test forensic

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Guardian rules become a public attack surface (spammers reverse-engineer them) | Public stable API, rule weights and churn thresholds unpublished. Rules live in `forge/.guardian/rules/` with private tuning files. |
| Internal authors object that "we are gated too" | Guardian only adds labels; never auto-rejects. Maintainers retain full override. RFC frames it as "evidence floor, not ceiling on judgment." |
| `forge/` folder bloats AITER clone size | Filtered payload is ~1MB. `results/` archive (331MB) stays in the historical `sunway513/aiter-forge` repo. |
| Guardian misclassifies legitimate work | Verdict labels include rationale links. `needs-evidence` is the soft default; `gate-blocked` is reserved for clear policy violations (e.g. churn detector positive + missing CI). |
| Self-hosted runner naming `linux-aiter-forge-mi355-{1,8}` doesn't match new location | Runner names stay; CI workflow path references update. Coordinated with @okakarpa. |

## Notes

- Audience: AITER team only. No exec review needed.
- Origin discussion: internal Teams chat 2026-05-15, on the back of `ROCm/aiter` PR series #3204/#3205/#3206 and RFC #3202.
- Source repo to be archived after merge: `sunway513/aiter-forge`. Open issues there (sub-project-2 targets, RFC #11 monorepo integration, #28 agent workflow) will be transferred to `ROCm/aiter`.

## Acceptance

This RFC is accepted when (a) the forge-subdir PR is merged into `ROCm/aiter:main` and (b) all 11 child issues are filed and labeled `area: forge`.
