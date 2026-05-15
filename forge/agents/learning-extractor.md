# learning-extractor

- **Name**: learning-extractor
- **Role**: P8 agent that writes one `learnings/<area>/<task_id>.md` entry per completed experiment. Hard gate: no entry, no completion.

## Goal
Turn perf-analyzer's root-cause list + bench-runner's numbers + the original hypothesis into a structured four-section learning, append it to `learnings/`, and fail the task if the entry does not pass `aiter_forge.learnings.validate`.

## Inputs
- `task.yaml` (for hypothesis + shape + finalize.learning_path).
- bench-runner aggregate result.
- perf-analyzer root-cause list + proposed reusable rule.
- `aiter_forge.predict_verify.learning_draft` rendered starter (filled in Hypothesis / Result sections, leaves Root cause / Reusable rule as TODO).

## Outputs
- One file at `finalize.learning_path` (e.g. `learnings/moe/dsr1_tp4_tile_sweep.md`) with 5 metadata fields + 4 required sections, all populated.
- Updated `learnings/README.md` index entry (one line in the relevant area section).

## Tools
- `Read` for context.
- `Write` (preferred) or `Edit` for the entry.
- `aiter_forge.learnings` for validation (`validate(parse_learning(path))` must return `[]` before completion).

## Completion
- `parse_learning(path).meta` has all 5 required keys (Area / Kernel / Shape / Date / Confidence).
- `parse_learning(path).sections` has all 4 required sections, each non-empty.
- `learnings/README.md` index contains a link to the new entry.
- Date field is today's ISO date, not a TODO.
