# [forge][Mission B] Claim extraction — parse PR body / commits for perf assertions

Refs RFC #TBD-RFC. Depends on #TBD-06 (PR webhook).

## Goal

Given a PR's body, commit messages, and file diff, extract a structured list of performance claims that the PR makes. Each claim is bound to one or more verifiable measurements in the next stage.

## Claim taxonomy

Initial supported claim types:

1. **Throughput claim** — "+X% tok/s" / "Y tok/s" / "Z× speedup" against a named baseline.
   - Inputs needed: numerator (PR), denominator (baseline named or implicit), workload (model + ISL/OSL + concurrency).
2. **Kernel-selection claim** — "fast path is selected for shape X" / "kernel K is dispatched."
   - Inputs needed: shape, kernel name, dispatch path.
3. **Allocation-overhead claim** — "removes N allocator calls per step" / "alloc delta of N bytes."
   - Inputs needed: shape, allocator pattern, expected delta.
4. **Tuning-config-validity claim** — "row R in tuned_*.csv is now reachable" / "shape S now hits a tuned config."
   - Inputs needed: row identity, dispatch lookup key.
5. **Correctness claim** — "no numerical regression vs baseline on test T."
   - Inputs needed: test T identity, tolerance.

## Approach

Two-stage:

1. **Rule-based pre-pass** — regex/keyword scan to spot obvious patterns (e.g. `\+\d+(\.\d+)?% (?:vs|against)`).
2. **LLM-assisted structured extraction** — pass PR body + commit messages to a constrained-output LLM call (Claude Sonnet), schema = list of `Claim` dataclasses.

Both stages run; results are merged and de-duplicated. Discrepancies (rule-based finds X but LLM doesn't, or vice versa) are logged and contribute to a confidence score.

## Scope

- `forge/src/aiter_forge/guardian/claims.py` — `Claim` dataclass + `extract_claims(pr_body, commits, diff) -> list[Claim]`.
- Initial rule corpus: ~30 regex patterns covering the 5 claim types.
- LLM prompt template at `forge/src/aiter_forge/guardian/prompts/claim_extraction.md`.
- Unit tests with 20+ sample PR bodies (positive and negative).

## Out of scope

- Binding claims to executable tests (issue #TBD-08).
- Verdict generation (issue #TBD-09).

## Definition of done

- Extracts 4 of the 5 claims correctly from PR #3202's RFC body.
- Extracts at least one claim each from PRs #3204, #3205, #3206.
- Returns empty list on the last 5 non-perf PRs to AITER (e.g. doc-only, version-bump).

## Owner

forge maintainers + LLM-prompting engineer.

## Estimated effort

M — ~300 LOC plus prompt iteration.
