# aiter-forge Enhancement Proposals

Based on the FlyDSL RoPE backend integration experience, here are concrete proposals to make aiter-forge better suited for kernel development and backend submission workflows.

---

## Proposal 0: Pre-Submit Format Checks

### Problem
AITER CI enforces Black and Ruff code style checks. During the FlyDSL RoPE PR, the first submission failed CI because the code wasn't formatted. This wasted a CI cycle and delayed review.

### Proposed Changes

Add a `pre_submit` block to target.yaml that runs before PR creation:

```yaml
pre_submit:
  format_checks:
    - command: "black --check --diff {files}"
      fix_command: "black {files}"
      name: "Black"
    - command: "ruff check {files}"
      fix_command: "ruff check --fix {files}"
      name: "Ruff"
  files_pattern: "aiter/ops/flydsl/*.py op_tests/flydsl_tests/*.py"
```

In the `aiter-forge submit` workflow:
1. Before creating PR, run all `format_checks` commands
2. If any fail, automatically run `fix_command` and re-stage
3. Report which checks were auto-fixed
4. Never submit a PR that would fail upstream CI format checks

### Implementation
- Parse `pre_submit.format_checks` from target.yaml
- `{files}` is expanded from `files_pattern` glob
- Run sequentially (Black before Ruff, since Black changes may fix Ruff issues)
- Auto-fix is opt-in: `aiter-forge submit --auto-fix` vs `--check-only`

---

## Proposal 1: Unit Test Enforcement in Target Pipeline

### Problem
Current aiter-forge has `correctness.command` which runs a single pass/fail test. But for AITER submission, we need structured unit tests (pytest) with specific coverage requirements. There's no way to enforce "new backend must have tests" or track test coverage.

### Proposed Changes

Add a `tests` block to target.yaml:

```yaml
tests:
  # Required: unit test suite that must pass before any benchmarking
  unit:
    command: "cd $AITER_ROOT && pytest op_tests/flydsl_tests/test_flydsl_rope.py -v --tb=short"
    pass_pattern: "passed"
    min_tests: 10  # NEW: minimum number of test cases that must pass
    
  # Optional: cross-validation against another backend
  cross_check:
    command: "cd $AITER_ROOT && FLYDSL_BENCH=1 pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s"
    pass_pattern: "MATCH"
    
  # Optional: end-to-end model accuracy test
  e2e:
    command: "cd $AITER_ROOT && python e2e_test.py --model llama-3.1-8b --backend flydsl"
    pass_pattern: "accuracy within tolerance"
```

In `mini_loop.py`, enforce:
1. `tests.unit` runs before any optimization loop
2. After kernel modification, re-run `tests.unit` — if it fails, rollback
3. `tests.cross_check` runs periodically (every N iterations)
4. `tests.e2e` runs at the end before declaring "submit-ready"

### Implementation
- Extend `MiniLoop._run_correctness()` to handle multi-test blocks
- Add `min_tests` parser: count "passed" in pytest output
- Add test result to lineage metadata (which tests passed per variant)

---

## Proposal 2: Backend Submission Workflow

### Problem
aiter-forge's current workflow is "optimize existing kernel". For "add new backend" use cases, the workflow is different:
1. Start with an external kernel (e.g., from FlyDSL repo)
2. Modify it for AITER compatibility
3. Validate against existing backend
4. Create AITER wrapper + tests
5. Submit PR

aiter-forge doesn't have tooling for steps 1-2 or 4-5.

### Proposed Changes

Add a `submission` mode:

```bash
aiter-forge submit --target targets/flydsl_rope \
    --baseline-backend triton \
    --new-backend flydsl \
    --aiter-root /path/to/aiter
```

This would:
1. Run correctness tests (unit + cross-check + e2e)
2. Run benchmark comparison (new vs baseline)
3. Generate a submission report (markdown)
4. Optionally create PR with:
   - Kernel wrapper code
   - Unit tests
   - Benchmark results table
   - Performance comparison chart

### Target YAML Extension

```yaml
submission:
  baseline_backend: triton
  new_backend: flydsl
  
  # Files to include in PR
  files:
    kernel: aiter/ops/flydsl/rope_kernels.py
    tests: op_tests/flydsl_tests/test_flydsl_rope.py
    
  # Acceptance criteria
  criteria:
    performance: ">= 1.0x vs baseline"  # must not be slower
    accuracy: "bit-identical for bf16, atol<5e-3 for f16"
    test_coverage: "min 20 parametrized configs"
    e2e: "model accuracy within 0.1% of baseline"
```

---

## Proposal 3: Golden Reference Management

### Problem
Validating kernel modifications requires deterministic inputs and saved reference outputs. Currently this is all ad-hoc (our Step 0 script). aiter-forge should manage golden references as a first-class concept.

### Proposed Changes

```yaml
golden:
  # Generate and save reference tensors
  generate_command: "python generate_golden.py --shapes-file shapes.yaml"
  save_dir: "golden/"
  
  # Validate against golden after each modification
  validate_command: "python validate_golden.py --golden-dir golden/"
  
  # Bit-identical requirement (strictest)
  strict_match: true  # if false, use atol from validation.precision
```

In the optimization loop:
1. Before first iteration: generate golden reference
2. After each kernel modification: validate against golden
3. If `strict_match` and diff > 0: reject modification, rollback

---

## Proposal 4: Multi-Backend Comparison Dashboard

### Problem
When adding a new backend, you need to compare it against all existing backends (HIP, Triton, native). Currently aiter-forge only tracks one metric series.

### Proposed Changes

Add `comparison` mode that runs all backends on the same shapes:

```bash
aiter-forge compare --target targets/flydsl_rope \
    --backends triton,flydsl,hip \
    --shapes shapes.yaml \
    --output comparison_report.json
```

Output: a table like:
```
Shape           | Triton (us) | FlyDSL (us) | HIP (us) | Best
T=1, 405B TP1   |     53.9    |     36.0    |   45.2   | FlyDSL (1.50x)
T=128, 70B TP1  |     56.7    |     36.6    |   48.1   | FlyDSL (1.55x)
```

This requires:
- Multiple `benchmark.command` entries per backend
- Unified scoring across backends
- Comparison report generation

---

## Proposal 5: Operator-Dashboard Integration

### Problem
ROCm operator-dashboard contains MI300X/MI355X performance data for 20+ models. Currently we manually extract shapes from it. aiter-forge should be able to import these shapes automatically.

### Proposed Changes

```yaml
benchmark:
  shapes_source: "operator-dashboard"
  dashboard_repo: "https://github.com/ROCm/operator-dashboard"
  operator: "rope"  # filter by operator type
  # Auto-generates shapes from dashboard CSV/JSON data
```

Implementation:
- Script to parse `data/mi300x/mi300x_membound.json`
- Extract operator-specific shapes (batch, seq, heads, dim)
- Generate `shapes:` list compatible with target.yaml
- Track SOL% improvement against dashboard baselines

---

## Proposal 6: CI/CD Integration for Backend Submission

### Problem
aiter-forge has `optimize.yml` GitHub Action but no workflow for "validate new backend and create PR".

### Proposed Changes

New workflow `submit-backend.yml`:

```yaml
name: Submit New Backend
on:
  workflow_dispatch:
    inputs:
      target: { required: true }
      baseline: { default: "triton" }
      
jobs:
  validate:
    runs-on: [self-hosted, linux-aiter-forge-mi355-8]
    steps:
      - name: Run unit tests
        run: aiter-forge test --target ${{ inputs.target }}
      - name: Run benchmark comparison
        run: aiter-forge compare --target ${{ inputs.target }} --baseline ${{ inputs.baseline }}
      - name: Generate submission report
        run: aiter-forge report --target ${{ inputs.target }}
      - name: Create PR (if all checks pass)
        run: aiter-forge submit --target ${{ inputs.target }} --create-pr
```

---

## Proposal 7: End-to-End Model Accuracy Validation (Recommended Path)

### Problem
Kernel-level correctness (atol < 1e-2) does not guarantee model-level accuracy. Floating point error accumulation across 80+ layers can cause divergence. Before submitting a new backend to AITER, we need to run inference on a real model and compare the output against a known-good baseline.

### Why This Is Critical
- A kernel that passes unit tests but has a subtle numerical difference (e.g., rounding mode) could cause perplexity regression
- ROCm operator-dashboard publishes golden accuracy numbers for models — we should validate against these
- This is the **highest-confidence validation** and should be the final gate before submission

### Proposed Changes

Add an `e2e` section to target.yaml as a **recommended path**:

```yaml
e2e_validation:
  # Model to test (should use the operator being modified)
  model: "meta-llama/Llama-3.1-8B-Instruct"
  
  # Inference framework
  framework: "vllm"  # or "sglang", "transformers"
  
  # Eval task
  eval:
    task: "lm_eval"
    benchmarks: ["mmlu", "gsm8k"]  # or "perplexity" on a fixed dataset
    
  # Baseline: run with default backend first
  baseline_backend_env: "AITER_ROPE_TRITON_BACKEND=1"
  
  # New: run with new backend
  new_backend_env: "AITER_ROPE_FLYDSL_BACKEND=1"
  
  # Acceptance criteria
  criteria:
    # Accuracy must be within this tolerance of baseline
    metric: "accuracy"
    tolerance: 0.001  # 0.1% absolute difference
    
    # Or for perplexity: lower is better, tolerance is relative
    # metric: "perplexity"
    # tolerance_pct: 0.5  # within 0.5% of baseline
    
  # Reference: operator-dashboard golden numbers
  golden_reference:
    source: "https://github.com/ROCm/operator-dashboard"
    model_key: "Llama-3.1-8B"
```

### Execution Flow

```
aiter-forge e2e --target targets/flydsl_rope

1. Pull model weights (or use cached)
2. Run baseline inference:
   AITER_ROPE_TRITON_BACKEND=1 python -m vllm.entrypoints.openai.api_server ...
   → lm_eval → baseline_accuracy.json
   
3. Swap backend:
   AITER_ROPE_FLYDSL_BACKEND=1 python -m vllm.entrypoints.openai.api_server ...
   → lm_eval → new_accuracy.json
   
4. Compare:
   - accuracy_diff = abs(new - baseline)
   - If diff < tolerance: PASS
   - Also compare against operator-dashboard golden numbers
   
5. Generate report:
   "Model: Llama-3.1-8B
    Baseline (Triton): MMLU=68.2%, GSM8K=52.1%
    FlyDSL:            MMLU=68.2%, GSM8K=52.1%
    Dashboard golden:  MMLU=68.3%
    Status: PASS (within 0.1% tolerance)"
```

### Integration with aiter-forge Pipeline

The recommended validation flow becomes:

```
Unit Tests → Cross-Backend Check → Golden Reference → Performance Sweep → E2E Model Test → Submit
     ↑              ↑                    ↑                   ↑                  ↑
  (required)    (required)          (recommended)        (required)      (recommended,
                                                                         highest confidence)
```

aiter-forge should enforce this order: if `e2e_validation` is defined in target.yaml, it must pass before `aiter-forge submit` allows PR creation.

---

## Priority Ranking

| # | Proposal | Impact | Effort | Priority |
|---|----------|--------|--------|----------|
| 0 | **Pre-submit format checks** | **High — prevents CI failures** | Low | **P0** |
| 7 | **E2E model accuracy validation** | **Critical — highest confidence gate** | Medium | **P0** |
| 1 | Unit test enforcement | High — catches regressions | Low | P0 |
| 5 | Operator-dashboard integration | High — automates shape selection | Medium | P0 |
| 2 | Backend submission workflow | High — end-to-end automation | Medium | P1 |
| 3 | Golden reference management | Medium — prevents silent errors | Low | P1 |
| 4 | Multi-backend comparison | Medium — enables informed decisions | Medium | P2 |
| 6 | CI/CD for submission | Medium — automates PR creation | Medium | P2 |
