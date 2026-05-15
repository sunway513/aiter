# [forge][Mission A] Wire forge sweeps to ROCm/AI-Frameworks-Dashboard staleness detection

Refs RFC #TBD-RFC.

## Goal

When forge runs a tuning sweep on a kernel that also appears in the `ROCm/AI-Frameworks-Dashboard` op-perf tab, automatically diff the new measurement against the dashboard's published number. If the gap exceeds a threshold (or the dashboard row is older than a freshness window), open a data-patch PR against the dashboard.

## Background

`sunway513/aiter-forge` `agents/dashboard-refresh.md` plus `scripts/refresh_dashboard.py` and `scripts/triage_dashboard_refresh.py` already implement most of this. The MiniMax-M2.5 refresh found rows up to **30× stale**. This issue is about productizing that workflow as a routine ritual.

## Scope

- Promote the existing dashboard-refresh agent into a forge first-class subcommand: `forge dashboard-refresh --model <name> [--threshold ratio]`.
- Add a CI workflow that runs `forge dashboard-refresh` weekly across all dashboard models.
- Wire it to file a data-patch PR against `ROCm/AI-Frameworks-Dashboard` automatically when a row is N× stale.
- Surface "staleness" as a chart in the dashboard so reviewers can see freshness at a glance.

## Out of scope

- Re-architecting the dashboard schema (separate work).
- Cross-vendor (NVIDIA) dashboard refresh — this is AMD-side staleness only.

## Definition of done

- `forge dashboard-refresh` subcommand exists and is documented in `engineer-quickstart.md`.
- One round of weekly auto-PRs has landed at least 10 stale-row corrections.
- Threshold and freshness window are configurable per model.

## Owner

forge maintainers + one dashboard committer.

## Estimated effort

M — wrapper + CI workflow + dashboard-side PR template.
