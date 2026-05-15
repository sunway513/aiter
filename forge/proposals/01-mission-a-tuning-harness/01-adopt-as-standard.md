# [forge][Mission A] Adopt forge/ as standard AI-agent kernel tuning entry for AITER

Refs RFC #TBD-RFC.

## Goal

Make `forge/dispatcher` the standard, recommended entry point for any AITER engineer kicking off an AI-assisted kernel optimization on AMD MI-series.

## Scope

- Add a "Kernel tuning with forge" section to AITER's top-level `CONTRIBUTE.md`.
- Add a one-page quickstart at `forge/docs/engineer-quickstart.md` (covered in a separate child issue).
- Document the standard 5-agent roster invocation pattern.
- Define the canonical task lifecycle: `tasks/<name>.yaml` → dispatcher run → rocprof + benchmark → `learnings/<name>.md`.

## Out of scope

- Forcing existing in-flight kernel work to adopt forge mid-stream.
- Migrating legacy ad-hoc tuning scripts under `aiter/scripts/`.

## Definition of done

- `CONTRIBUTE.md` has a forge section.
- A new engineer can run `forge run --target targets/aiter_mha --mode benchmark` from a clean clone and produce a `learnings/` entry within 30 minutes.
- At least one in-flight kernel optimization (suggestion: an open MoE BF16 task) has been re-run through forge as a dogfooding proof.

## Owner

forge maintainers (TBD), with one AITER kernel engineer for the dogfood.

## Estimated effort

S — docs and one dogfooding run.
