# [forge][Mission A] Standard 5-agent roster — port from sunway513/aiter-forge and document

Refs RFC #TBD-RFC.

## Goal

Promote the 5-agent kernel-tuning roster from the legacy repo to first-class status inside `forge/agents/`, with one canonical responsibility per agent and a documented hand-off contract.

## The roster (already shipped under `forge/agents/`)

1. **bench-runner** — owns the perf benchmark execution. Inputs: target.yaml, kernel build. Outputs: rocprof trace, tok/s, alloc-trace, verdict on success criterion.
2. **kernel-writer** — generates kernel-source variants under a hypothesis. Inputs: source bundle from `ir-inspector`, hypothesis. Outputs: candidate diff(s).
3. **perf-analyzer** — reads rocprof traces, extracts the bottleneck claim. Inputs: rocprof. Outputs: hypothesis seeds for next round.
4. **ir-inspector** — folds source ↔ HW-resource bundle (registers, occupancy, cache hit rates). Inputs: kernel binary. Outputs: source-annotated bundle for kernel-writer.
5. **learning-extractor** — emits the 4-section `learnings/*.md` entry. Inputs: hypothesis + result + rocprof + diff. Outputs: 1 markdown file conforming to schema.

Plus two specialized members already in the tree:
- `dashboard-refresh` (Mission A supporting ritual)
- `flydsl-cleanup` (FlyDSL-internal-types-cleanup skill)

## Scope

- Audit each `forge/agents/*.md` for: clear input/output contract, single responsibility, no overlap with another role.
- Add `forge/agents/README.md` enumerating the roster, the hand-off graph, and "when to spawn what."
- Add a contract test under `forge/tests/` that verifies the roster module loads each agent and that role-name uniqueness holds.

## Out of scope

- Adding new agents (a separate issue per addition).
- Changing the dispatcher contract (separate issue).

## Definition of done

- Each `agents/*.md` follows a single template (Role / Inputs / Outputs / Hand-off).
- `agents/README.md` exists and is referenced from `engineer-quickstart.md`.
- Contract test passes in CI.

## Owner

forge maintainers.

## Estimated effort

S — mostly editorial plus a 30-line contract test.
