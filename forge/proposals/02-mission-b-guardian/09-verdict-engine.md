# [forge][Mission B] Verdict engine — emit gate-passed / needs-evidence / gate-blocked labels

Refs RFC #TBD-RFC. Depends on #TBD-06, #TBD-07, #TBD-08.

## Goal

Given the extracted claims and verification results, produce a single PR-level verdict label and post a structured comment explaining the verdict.

## Verdict tiers

- **`gate-passed`** — every extracted claim has a binding, every binding produced a result, and every result PASS within tolerance.
- **`needs-evidence`** (default soft state) — at least one claim has no binding, or at least one binding could not execute (missing baseline, missing artifact, CI label not added).
- **`gate-blocked`** — at least one binding produced a FAIL with a clear policy violation. Reserved for cases where the claim is contradicted by measured evidence, OR where the churn detector positive coincides with missing CI labels for self-asserted perf claims.

## Comment format

A single forge-generated comment with:

```
## forge guardian verdict: <label>

| Claim | Binding | Result |
|---|---|---|
| ... | ... | PASS / FAIL / SKIP (reason) |

<rationale paragraph linking each FAIL/SKIP to the action the author can take>
```

Comment is updated in place on each new push (no comment spam).

## Override mechanism

Any maintainer can comment `/forge-override <gate-passed|needs-evidence>` to override the verdict. Override is logged and the comment is annotated with "Overridden by @<maintainer> on <ts>."

## Scope

- `forge/src/aiter_forge/guardian/verdict.py` — verdict aggregation + label management.
- `forge/src/aiter_forge/guardian/comment.py` — comment template + idempotent post/update.
- `/forge-override` parser + policy enforcement (only repo collaborators).

## Out of scope

- Hard rejection (PR is never auto-closed).
- Cross-PR verdict propagation (each PR is independent for MVP).

## Definition of done

- Three verdict labels exist on `ROCm/aiter` (`gate-passed`, `needs-evidence`, `gate-blocked`).
- One synthetic PR per verdict tier triggers the correct label and comment.
- The acceptance-test issue (#TBD-11) confirms `gate-blocked` is emitted on PR #3204/#3205/#3206.

## Owner

forge maintainers.

## Estimated effort

M — ~400 LOC plus label setup and one round of UX iteration with maintainers.
