# [forge][Mission B] Churn detector — author × file × time-window similarity

Refs RFC #TBD-RFC. Feeds into verdict engine #TBD-09.

## Goal

Detect when the same author repeatedly opens, closes, and re-opens cosmetically similar PRs over a short time window. Surface the signal as part of the guardian verdict so reviewers see the pattern without manually retracing PR history.

## Signal definition

A PR triggers the **churn signal** if all three hold:

1. The author has closed ≥ 2 prior PRs within the last 14 days that touched the same primary file(s) (defined as: file Jaccard overlap ≥ 0.8).
2. The diff Jaccard similarity between the new PR and the most recent closed PR is ≥ 0.7.
3. None of the closed-and-reopened PRs received ≥ 1 maintainer review with `state: APPROVED`.

The detector also emits a softer **resubmission notice** if just (1) holds, without the similarity gate.

## Real-world calibration

PR series #3197 → #3199 → #3204 / #3205 / #3206 by @peymanr on `aiter/fused_moe.py` over 2 weeks fits all three criteria. This pattern is the canonical positive case (see acceptance issue #TBD-11).

## Approach

`forge/src/aiter_forge/guardian/churn.py`:

```python
def detect_churn(pr: PR, history_days: int = 14) -> ChurnSignal: ...
```

Walks `gh pr list --author <login> --state closed --search "closed:>YYYY-MM-DD"`, computes file overlap and diff similarity, returns a `ChurnSignal` with severity, evidence list, and human-readable summary.

Diff similarity uses normalized line-set Jaccard (drop whitespace and rename comments out). Cheap, robust, no embeddings needed for MVP.

## Surfacing

Churn signal is surfaced two ways:

1. As a **section in the guardian verdict comment** when severity is non-zero.
2. As a **score input to `gate-blocked`**: churn-positive plus missing CI labels for self-asserted perf claims escalates `needs-evidence` to `gate-blocked`.

## Scope

- Churn detection module + tests.
- Wire into verdict engine (issue #TBD-09).
- One real-world fixture: PR #3204 with #3197/#3199 as prior closed history.

## Out of scope

- Cross-author churn (e.g. one author closes, a colleague re-opens) — heuristics get noisy fast.
- Cross-repo churn — AITER-only for MVP.
- Auto-flagging the author as suspicious — no individual flagging, only PR-level signal.

## Definition of done

- `detect_churn` returns positive on PR #3204 with full evidence list.
- Returns negative on a clean first-time PR.
- Verdict comment includes churn section when active.

## Owner

forge maintainers.

## Estimated effort

S — well-bounded, ~200 LOC plus calibration.
