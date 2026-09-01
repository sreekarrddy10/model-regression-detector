# Loom script — 3 minutes

Everything below is a real artifact in this repository. Nothing is staged.
Open these five tabs before you hit record, in this order:

1. https://github.com/sreekarrddy10/model-regression-detector/pull/2/files
2. https://github.com/sreekarrddy10/model-regression-detector/pull/2  (scrolled to the bot comment)
3. https://github.com/sreekarrddy10/model-regression-detector/actions/runs/33532871896
4. https://github.com/sreekarrddy10/model-regression-detector/pull/1
5. `docs/sample-report.html` open in a browser

---

## 0:00 — 0:25 · The problem, stated as a diff

**Tab 1 — PR #2 files.**

> "This is a pull request that trims a classifier prompt to cut token cost. It
> removes the tie-break rules and the few-shot examples and keeps the task
> statement and the category definitions. If you review this by reading it, it
> looks like a reasonable optimisation. If you spot-check it against a handful of
> support emails, they classify identically — because easy emails don't depend on
> what was removed."

Scroll the diff once. Don't linger.

> "It is a real regression, and it is invisible to both of the things teams
> normally do."

## 0:25 — 1:05 · The gate catches it

**Tab 2 — the sticky comment on PR #2.**

> "CI runs the full eval on every pull request and posts this."

Point at the verdict line, then the table.

> "Blocked. Accuracy went from 98.8 to 96.2 against a locked golden dataset of
> eighty hand-authored cases, each run three times at temperature zero."

**Now the line that matters. Point at the McNemar warning.**

> "Read this one. The accuracy drop is not statistically significant — McNemar
> puts it at p equals 0.219. A gate that watches accuracy alone waves this
> through as noise."

Pause.

> "It blocks because one case tagged critical regressed. Critical cases block
> deterministically, regardless of significance. That is the entire argument for
> having a critical stratum, and this is it happening."

## 1:05 — 1:35 · The judge is calibrated

Still on the comment. Point at the calibration line.

> "Summary quality is scored by an LLM judge. That number is worthless unless the
> judge agrees with a human, so before any run reports quality it scores twenty
> summaries I graded myself and computes quadratic-weighted kappa against my
> scores. Kappa is 0.96 here. The floor is 0.60 — below it the run aborts rather
> than reporting a quality number nobody should believe."

> "That is the piece most eval suites skip, and skipping it is how a suite starts
> reporting a confident 4.2 that correlates with nothing."

## 1:35 — 2:05 · The report

**Tab 3 — the Actions run.** Show the `eval-report` artifact, then **Tab 5**, the
rendered HTML.

> "Every run uploads a single self-contained HTML report — no external requests,
> no scripts. Side-by-side attempts, the regressed cases named, the statistics,
> and the calibration panel."

Scroll to a regressed case.

> "It names `gen-017` — the case the trimmed prompt broke. Whoever picks this up
> tomorrow doesn't have to reproduce anything."

## 2:05 — 2:35 · The other half of the proof

**Tab 4 — PR #1, merged and green.**

> "The same pipeline on a healthy change: merged green at 98.8 percent. A gate
> that only ever blocks is a broken build, not a gate."

> "The baseline it compares against is recorded by a full run on main, cached, and
> restored by every later pull request."

## 2:35 — 3:00 · What it cost, and what it caught

> "The full suite is two minutes and about thirty cents a run. Eighty cases,
> three repeats, plus judge calibration."

Close on this — it is the strongest thing you can say:

> "Building this found five bugs in the harness itself, and two of them were the
> gate reporting green while measuring nothing. One reported PASS on a run where
> every API call had failed with a 401. The other meant no pull request could ever
> see a baseline, so every one of them passed. Both were invisible to a test suite
> at 96 percent coverage, because the cassette layer that makes tests free is
> exactly what hides the live path. They only showed up the first time this ran
> against a real provider in real CI."

---

## Don't say

- "I hand-verified every label." You audited 80 mechanically and read 12 in depth.
  Say **"audited three ways — rule consistency, live baseline, few-shot leakage."**
- Anything implying the dataset is finished. `acct-013` fails deliberately;
  68 of 80 labels have not been read case-by-case. Both are in the README.

## If asked "what would you do next"

- Read the remaining 68 cases.
- Widen the `technical` stratum — it thinned from 20 to 18 as seat and lockout
  cases moved to `account` during review.
- Fix `latest_baseline`, which filters on `dataset_hash` but not `prompt_version`,
  so `make eval-demo` becomes its own baseline if run twice.
