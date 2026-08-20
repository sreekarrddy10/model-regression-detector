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

## From building the report suite

- **`select_autoescape(["html"])` matches the final extension.** A template named
  `foo.html.j2` has extension `.j2`, so autoescaping is off and it looks correct in review.
  Prefer `autoescape=True` when every template a loader serves is HTML.
- **A test that renders nothing passes everything.** Three tests asserted things about case
  content on reports where the fixture produced no case detail at all - including the one
  meant to prove escaping worked. Assert the fixture is non-empty before asserting about it.
- **`python - <<EOF` consumes stdin for the program.** A hook that reads a JSON payload from
  stdin cannot also take its program from stdin; use `-c`. The failure is silent.
- **Run a new suite ten times before trusting it.** Cheap, and it converts "it passed" into
  "it is stable".
