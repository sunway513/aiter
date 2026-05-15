# aiter-forge architecture

## Why this restructure

External expert input (partner running 100% AI-team-generated kernels on NVIDIA) distilled three principles for kernel optimization driven by agent teams. This doc records how aiter-forge operationalizes them. See issues #27 and #28 for the full context.

| Principle | Expert statement | Our enforcement mechanism |
|---|---|---|
| **Simplicity hard gate** | "Anything fast, correct, and complicated must be thrown away." | `aiter_forge.complexity` — quantified limits on env flags, function LOC, branches per function, total file LOC. PR fails CI when crossed. |
| **Evidence-based, no trial-and-error** | "Agents must produce solid data-based plans, explain with evidence why something worked/failed, update learnings, continue." | `aiter_forge.predict_verify` — tasks YAML require a non-empty hypothesis + named candidates + verify cmd before running. `aiter_forge.learnings` — every completed experiment appends a structured entry; schema enforced. |
| **Optimize for the agent team** | "If it has the right tools, feedback loop, architecture information, the agent team will do the complete job by itself." | Task-as-YAML (`tasks/template.yaml`) decouples human intent from execution. `KnowledgeBase` (static priors) + `LearningsStore` (dynamic experience) feed agents. Branch protection + CI keep humans out of the hot path. |

## Directory layout

```
aiter-forge/
├── AGENTS.md / CLAUDE.md             agent-facing policy (branch protection, workflow)
├── docs/architecture.md              this document
├── src/aiter_forge/
│   ├── knowledge/                    static architectural priors (CDNA4 ISA, etc.)
│   ├── learnings.py                  dynamic experiment-derived knowledge store
│   ├── complexity.py                 simplicity hard-gate analyzer
│   ├── predict_verify.py             hypothesis-driven task runner
│   ├── lineage/                      kernel-variant provenance (existing)
│   ├── evolution/                    optimization loop (existing)
│   └── validation/                   accuracy / perf / schema validators (existing)
├── learnings/                        experiment logs; one .md per learning
│   ├── README.md                     index + schema documentation
│   ├── moe/
│   └── infra/
├── tasks/                            task-as-YAML specs (one task = one agent prompt)
│   └── template.yaml
├── tests/                            pytest unit tests
│   ├── test_learnings.py
│   ├── test_complexity.py
│   ├── test_predict_verify.py
│   └── test_pre_push_hook.sh
└── .github/workflows/                CI (branch protection tripwire + hooks + tools)
```

## Agent lifecycle

```
    (human or P9 agent)
            |
            v
     writes tasks/<id>.yaml           <- hypothesis required up front
            |
            v
    aiter_forge.predict_verify.parse
            |   (rejects if hypothesis / candidates / verify / finalize missing)
            v
     run_candidate(spec, candidate)   <- per-candidate verify cmd
            |
            v
     learning_draft(spec, results)    <- pre-fills learnings/<id>.md
            |
            v
        human review                   <- fills in Root cause + Reusable rule
            |
            v
     learnings_lint passes             <- required sections + meta present
            |
            v
          commit + PR                  <- complexity_gate + hooks-ci + learnings_lint
```

No step is optional. Skipping "hypothesis" skips the whole workflow — agents can't declare an experiment complete without it.

## What this restructure does NOT do (yet)

- **`tools/kernel_inspect.py`** — the MLIR + ISA + rocprof four-in-one bundle for source↔hardware mapping. Scaffolded only; GPU-dependent implementation lands in a follow-up PR.
- **`agents/*.md` roster** — standard 5-agent types (kernel-writer, bench-runner, ir-inspector, perf-analyzer, learning-extractor). Comes with #28.
- **`scripts/aiter.py`** single entry point — subsumes the scattered `scripts/*.sh` into one dispatcher. Also #28.

## How to evolve this

- New domain → add a subdir under `learnings/` (e.g. `learnings/attn/`, `learnings/rope/`). The store auto-discovers.
- New complexity metric → add a dataclass field to `Limits`, wire into `check()`. UT covers the new rule.
- New task field → update `REQUIRED_FIELDS` in `predict_verify`, regenerate `tasks/template.yaml`, update the UT fixtures.
