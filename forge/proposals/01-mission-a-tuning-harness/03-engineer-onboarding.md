# [forge][Mission A] Engineer onboarding doc — how to launch forge against a target

Refs RFC #TBD-RFC.

## Goal

Produce `forge/docs/engineer-quickstart.md`. A new AITER engineer should be able to read it once and run their first forge tuning run within 30 minutes.

## Required content

1. **Prerequisites**
   - One MI-series GPU node (recommended `linux-aiter-forge-mi355-{1,8}` self-hosted runner, fallback: any local MI300X/MI355X box).
   - `pip install -e ./forge[dev,llm,remote]`.
   - `forge.yaml` filled in (gpu host, user, aiter root path).

2. **First run — benchmark mode (no API key needed)**
   - Pick an existing target (`forge/targets/aiter_mha`).
   - `forge run --target targets/aiter_mha --mode benchmark`.
   - Read the resulting `report.json` and `learnings/<auto>.md`.

3. **Second run — manual mode**
   - Same target, `--mode manual --rounds 3`.
   - Inspect the generated optimization prompt at `optimization_logs/round_N_prompt.md`.
   - Apply an edit by hand, hit Enter, see the next round.

4. **Third run — auto mode**
   - Set `ANTHROPIC_API_KEY`.
   - `--mode auto --rounds 3`.
   - Walk through what the agent did via the `learnings/` entries.

5. **Authoring a new target**
   - Copy `forge/targets/template/`.
   - Fill in `target.yaml` (shapes, baseline cmd, success criterion).
   - Smoke-test with `--mode benchmark`.

6. **The hard rules** (one paragraph each)
   - Hypothesis required on every task.
   - 4-section learnings on every completed run.
   - Complexity gate.
   - Branch protection.

7. **Where to ask for help**
   - `area: forge` issue label.
   - Slack/Teams channel TBD.

## Out of scope

- Guardian mode (Mission B) — separate doc once that ships.
- Architecture deep-dive — `forge/docs/architecture.md` already covers it.

## Definition of done

- One markdown file under `forge/docs/engineer-quickstart.md`.
- One AITER engineer (not the author) follows it cold and reaches a successful first run.

## Owner

forge maintainers.

## Estimated effort

S — one focused doc-writing afternoon plus one user-test.
