# Lessons

Patterns captured after corrections. One entry per correction — what happened, the rule that
prevents it recurring.

## Pre-seeded (from ECC `skills/eval-harness` + `skills/ai-regression-testing`)

- **Never let a model grader gate a release on its own.** Confirm across ≥3 seeds, or demote it to
  advisory. Flaky graders in release gates is a named anti-pattern.
- **An LLM-as-judge is an unvalidated model until you measure it.** Hold out human-scored examples
  and compute agreement before trusting any judge score.
- **Cost and latency drift are regressions.** Chasing pass rates while the bill doubles is a
  failure mode, not a success.
- **Test where bugs were found, not where code looks risky.** Every production failure becomes a
  named golden case (`source: from_failure`).
- **AI reviewing its own output shares its own blind spots.** Mechanical checks run before
  judgment-based ones, always.
