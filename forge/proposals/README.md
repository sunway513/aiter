# forge/ proposals — pre-PR drafts

This folder holds the umbrella RFC, eleven child issue drafts, and one acceptance-test forensic. Once reviewed, each `*.md` here is filed verbatim as a GitHub issue body against `ROCm/aiter` via `gh issue create`. After all issues are open, this folder is removed in a follow-up PR.

## Contents

- `00-RFC-forge-subdir.md` — umbrella RFC for adopting `forge/` inside AITER

### Mission A — Kernel tuning harness (5 issues)

- `01-mission-a-tuning-harness/01-adopt-as-standard.md`
- `01-mission-a-tuning-harness/02-port-triton-targets.md`
- `01-mission-a-tuning-harness/03-engineer-onboarding.md`
- `01-mission-a-tuning-harness/04-dashboard-staleness.md`
- `01-mission-a-tuning-harness/05-agent-roster.md`

### Mission B — Guardian gate (5 issues)

- `02-mission-b-guardian/06-pr-webhook.md`
- `02-mission-b-guardian/07-claim-extraction.md`
- `02-mission-b-guardian/08-claim-test-binding.md`
- `02-mission-b-guardian/09-verdict-engine.md`
- `02-mission-b-guardian/10-churn-detector.md`

### Acceptance test (1 issue)

- `03-acceptance-test/11-must-block-3202.md` — guardian MVP must produce BLOCK verdict on PR series #3204/#3205/#3206

## Filing the issues

After this PR is reviewed and merged into `sunway513:main`, file each issue with:

```bash
for f in forge/proposals/**/*.md; do
  title=$(head -1 "$f" | sed 's/^# //')
  gh issue create --repo ROCm/aiter --title "$title" --body-file "$f"
done
```

The umbrella RFC is filed first; child issues reference it by number.
