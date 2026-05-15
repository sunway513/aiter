# agents/ — standard P8 roster

Each file here is a **standard agent type** the dispatcher can spawn. Format enforced by `aiter_forge.agent_roster.AgentRoster`; five required sections per agent: `Goal / Inputs / Outputs / Tools / Completion`.

The P9/P8/P7 tiering (see PUA skills and issue #28):
- **P9** — human or tech-lead agent, writes `tasks/*.yaml`, orchestrates execution, never writes code.
- **P8** — one of the five agents below. Executes one task YAML end-to-end, including calling P7 helpers if needed.
- **P7** — sub-agent spawned by P8 for a narrower task (e.g. `mlir-fix` spawned by `kernel-writer`).

## Roster

- [kernel-writer](kernel-writer.md) — modifies FlyDSL `.py` / MLIR, produces a diff.
- [bench-runner](bench-runner.md) — runs op-level and e2e benchmarks, emits structured JSON.
- [ir-inspector](ir-inspector.md) — dumps MLIR + ISA + rocprof, produces a source↔hardware bundle.
- [perf-analyzer](perf-analyzer.md) — reads bench + IR results, outputs root-cause hypothesis.
- [learning-extractor](learning-extractor.md) — writes one `learnings/*.md` entry per completed experiment.
- [dashboard-refresh](dashboard-refresh.md) — refresh ROCm/AI-Frameworks-Dashboard op-perf rows on live MI355X and open data-patch PR; supporting ritual inside **Priority 1** (see `docs/dashboard_refresh_runbook.md`). Priority 2 work (CK removal tracking + competitive perf benchmarking) uses the same harness — see `docs/ck_removal_tracker.md` + issue #42.

Any task that doesn't fit one of the five → don't spawn a bespoke agent. Either extend one of the five (update its markdown) or raise a P9 design discussion.
