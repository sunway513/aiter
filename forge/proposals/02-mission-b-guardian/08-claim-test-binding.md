# [forge][Mission B] Claim-to-test binding library (rocprof / benchmark_serving / alloc-trace / dispatch-lookup)

Refs RFC #TBD-RFC. Depends on #TBD-07 (claim extraction).

## Goal

For each extracted `Claim`, return one or more executable verification recipes. Each recipe knows how to run on the self-hosted MI-series runner and how to interpret pass/fail.

## Binding map

| Claim type | Test binding |
|---|---|
| Throughput claim | `benchmark_serving.py` invocation with model + workload knobs from claim, plus a paired baseline run. PASS iff measured ratio is within ±10% of claimed ratio. |
| Kernel-selection claim | `rocprof --hsa-trace` on a synthetic shape-N micro, then grep for the named kernel in the trace. PASS iff kernel name appears at least once for the asserted shape. |
| Allocation-overhead claim | Allocator-trace micro that runs the asserted op N times and reports per-step delta vs caching baseline. PASS iff delta matches claim within tolerance. |
| Tuning-config-validity claim | Construct the dispatch lookup key from the claim, call `get_2stage_cfgs()` (or equivalent), assert non-empty result. PASS iff lookup hits. |
| Correctness claim | Run the named test (e.g. `pytest op_tests/<name>`) with PR head vs baseline. PASS iff output tolerance holds. |

## Approach

`forge/src/aiter_forge/guardian/bindings/` — one module per claim type. Each module exposes:

```python
def bind(claim: Claim) -> list[VerificationRecipe]: ...
def execute(recipe: VerificationRecipe, pr_head: str, baseline_ref: str) -> VerificationResult: ...
```

Recipes are serializable so they can be replayed and audited.

## Scope

- Five binding modules.
- Shared `VerificationRecipe` and `VerificationResult` schemas under `guardian/types.py`.
- Test fixtures under `forge/tests/guardian/` with at least one positive and one negative recipe per claim type.

## Out of scope

- Verdict aggregation (issue #TBD-09).
- Caching of recipe results (separate optimization).

## Definition of done

- All five binding types implemented with passing fixtures.
- Each binding type has at least one fixture sourced from a real historical PR (positive case) and one synthetic negative case.
- The acceptance-test issue (#TBD-11) shows each of the five binding types triggering on at least one claim from PR #3204/#3205/#3206.

## Owner

forge maintainers + one perf engineer for the rocprof/benchmark recipes.

## Estimated effort

L — five recipe runners plus rich fixture authoring.
