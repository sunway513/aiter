# forge — AITER's AI-agent harness

`forge/` is AITER's first-party AI-agent toolkit. It serves two missions inside the AITER project:

**Mission A — Kernel tuning harness for AITER engineers.**
Standard entry point for AI-assisted kernel optimization work on AMD MI-series GPUs. Provides the dispatcher, agent roster, predict-verify loop, learnings store, and complexity gates that turn ad-hoc "human-in-the-loop kernel evolution" into a repeatable, evidence-based workflow.

**Mission B — Guardian gate for incoming AITER PRs.**
On every incoming pull request that asserts a performance claim, forge runs claim extraction, binds each claim to an executable verification (rocprof / benchmark_serving / allocator-trace), and emits a verdict label (`gate-passed` / `needs-evidence` / `gate-blocked`) before maintainer review. AI lowered the cost of opening a PR — guardian raises the cost of opening one without evidence.

## Status

This subdirectory was folded into AITER from `sunway513/aiter-forge` on 2026-05-15. See `proposals/` for the open RFC and child issues. Proposals will be filed as GitHub issues against `ROCm/aiter` once reviewed; this folder is the staging ground.

## Quick start

### Tuning mode (Mission A)

```bash
cd forge
pip install -e ".[dev]"
cp forge.yaml.example forge.yaml      # set gpu.host, gpu.user, gpu.aiter_root
forge run --target targets/aiter_mha --mode benchmark
```

See `docs/architecture.md` and individual `targets/*/target.yaml` for the per-kernel configuration.

### Guardian mode (Mission B)

Not yet wired. See `proposals/02-mission-b-guardian/` for the design and acceptance test.

## Layout

```
forge/
  src/aiter_forge/    # dispatcher, kernel_inspect, predict_verify, complexity, learnings, agent_roster, cli
  agents/             # 7 standard agent role definitions
  targets/            # per-kernel target configs (FlyDSL FMHA, MoE FP4, RoPE, MHA)
  tasks/              # YAML experiment definitions (hypothesis-required)
  learnings/          # 4-section evidence log per experiment
  docs/               # architecture, runbooks, CK-removal tracker
  scripts/            # benchmark + dashboard refresh scripts
  proposals/          # pre-PR drafts of the RFC + 11 child issues
  .githooks/          # branch protection (pre-push)
```

## Hard rules

- Every experiment is a `tasks/*.yaml` with a non-empty `hypothesis` field. No hypothesis, no run.
- Every completed experiment appends one entry to `learnings/` with the four required sections (Hypothesis / Result / Root cause / Reusable rule).
- Complexity is a hard CI gate. PRs that exceed configured LOC / branch-per-function / env-flag limits fail CI.
- Branch protection: pre-push hook blocks direct pushes to `main`.

See `AGENTS.md` for full agent-execution policy and `CLAUDE.md` for project rules.
