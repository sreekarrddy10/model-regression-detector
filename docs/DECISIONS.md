# Design Decisions

Every entry names the alternative that was rejected, because a decision without a
discarded option is not a decision. Where a claim is testable, the test that proves it
is linked.

---

## D1 — The judge is measured before it is believed

**Decision.** Twenty summaries are scored 1–5 by a human and held out. Before any run
reports a quality number, the LLM-as-judge scores those same summaries and its agreement
with the human is measured as a quadratic-weighted Cohen's κ. Below κ = 0.60 the judge's
scores are discarded and the gate blocks on the calibration failure itself.

**Why.** An LLM-as-judge is an unvalidated model. Most eval suites use one anyway and
report a confident 4.2 that has never been checked against anything a person would
recognise as quality. That number then drives decisions. Calibration is the cheapest
possible defence: twenty human judgements, once.

**Rejected: use the judge score unconditionally.** This is the default in every eval
framework I looked at. It is how an eval suite becomes theatre — it produces numbers
that move, so it looks like it is working.

**The subtle part.** A constant rater returns κ = 0.0, not 1.0. If the judge rates
everything 5, there is no variance to explain and κ is genuinely undefined; returning
0.0 means it *fails* the floor. Returning 1.0 — which a naïve "they agree on every
sample!" implementation would — lets the most useless possible judge pass calibration
perfectly. The same trap applies to the holdout itself, so
[`dataset/report.py`](../src/mrd/dataset/report.py) warns when the human scores span
fewer than three distinct values.

Proof: [`test_stats.py::test_constant_rater_yields_zero_not_a_pass`](../tests/test_stats.py),
[`test_runner.py::test_disagreeing_judge_fails_calibration`](../tests/test_runner.py).

---

## D2 — Deterministic graders gate; probabilistic graders advise

**Decision.** Category match and schema validity can block a merge on their own. Judge
scores cannot: a quality drop only blocks when it reproduces across three independent
seeds, and only when D1 passed.

**Why.** ECC's `eval-harness` skill names "allowing flaky graders in release gates" as
an anti-pattern, and it is right. A gate that blocks unpredictably gets routed around
within two sprints, and then the whole system is decoration. A blocked merge must be
explainable in one sentence to the person it blocked.

**Rejected: one blended quality score.** Cleaner to report, impossible to act on. "Your
score went from 8.1 to 7.4" tells a developer nothing about what to fix.

**The cost.** Some real quality regressions only warn. That is the deliberate trade:
under-blocking preserves the gate's credibility, over-blocking destroys it.

---

## D3 — Significance and effect size must both trip

**Decision.** A case counts as regressed only when it fails a majority of its three
repeats. Aggregate degradation blocks when it clears an 8% effect-size floor **or** when
McNemar's exact test on the discordant pairs returns p < 0.05 with net-negative
direction.

**Why.** The source guide proposes flat thresholds — warn at 3%, block at 8%. It also
poses the right question and then fails to answer it: *"If 2 out of 80 cases flipped, is
that signal or noise?"* A fixed percentage cannot tell. It answers identically at n = 20
and n = 2,000.

McNemar's exact test is the correct instrument here. Only discordant pairs carry
information — cases that passed both runs, or failed both, say nothing about whether the
change mattered — and under the null each discordant pair is equally likely to fall
either way. That is a two-sided binomial test on the discordant count, computed exactly.

**Concretely.** Six regressions and zero improvements out of a hundred is a 6% drop,
*under* the guide's 8% block threshold, so a flat gate waves it through. p = 0.031, so
this gate blocks it. Inversely, one regression in twenty is a 5% drop with p = 1.0, and
only warns.

**Rejected: p-value alone.** At large n, statistically significant differences become
trivially attainable and the gate starts blocking on noise-sized real effects. Both
conditions exist to catch the other's failure mode.

**Rejected: the guide's flat thresholds as the whole test.** Kept as the effect-size
half, discarded as the entirety.

Proof:
[`test_compare.py::test_significant_regression_blocks_even_under_the_effect_floor`](../tests/test_compare.py),
[`test_stats.py`](../tests/test_stats.py) — every statistic is checked against a
hand-computed value, not only against itself.

---

## D4 — The dataset hash is part of the comparison key

**Decision.** Every run records a SHA-256 over the semantic content of the golden set.
Runs with differing hashes are refused for comparison, loudly. `dataset.lock.json`
freezes the hash, and CI fails if ground truth drifted without a version bump.

**Why.** This is the quiet failure that makes an eval suite actively harmful. Someone
relabels three ambiguous cases to match what the model already does — reasonably, in
good faith — and the next run shows a 4% improvement. Nothing in a timestamp-ordered
comparison can detect that. The model did not change; the ruler did.

**Rejected: compare by timestamp, most recent baseline wins.** The obvious
implementation, and it silently reports ground-truth edits as model improvements.

**Deliberate scope.** The hash covers semantic content, not file bytes. Reordering lines
or reflowing JSON keeps a baseline valid; changing any label, email, or difficulty tag
invalidates it. A hash that broke on whitespace would be re-locked reflexively until it
meant nothing.

Proof: [`test_dataset.py::test_any_label_change_changes_the_hash`](../tests/test_dataset.py),
[`test_compare.py::test_different_ground_truth_refuses_to_diff`](../tests/test_compare.py).

---

## D5 — The judge runs through the provider layer, not a framework

**Decision.** No DeepEval, no RAGAS, no `instructor`. The judge is a prompt plus a
structured-output schema, sent through the same provider abstraction as the feature under
test.

**Why.** Each of those libraries patches vendor clients directly and returns a parsed
object, bypassing the normalized `Response` that carries token counts, latency, and cost.
The gate treats cost and latency drift as regressions — ECC's `eval-harness` skill names
"ignoring cost and latency drift while chasing pass rates" as an anti-pattern — so
losing that accounting on judge calls means half the spend becomes invisible.

Keeping the judge on the provider layer also makes it cassette-replayable, so judge logic
is tested offline with no spend.

**Rejected: adopt the framework and reconcile two accounting paths.** More code than
writing the ~80-line judge, and the reconciliation is exactly where a subtle cost-tracking
bug would live.

**Also rejected: RAGAS specifically.** Its metrics (faithfulness, context precision,
answer relevancy) are retrieval-specific. This is a classification task with no retrieved
context. The metrics would have been decorative.

**What was kept.** `scipy`. Hand-rolled statistics gating merges is precisely where
silent wrongness lives — the dependency is worth it. The hand-computed values in
[`test_stats.py`](../tests/test_stats.py) match scipy exactly, which validates my
understanding as much as the code.

---

## D6 — A bad model output is data, not an exception

**Decision.** `classify()` never raises on malformed output. It returns `parse_error`
alongside the preserved raw response. Provider outages are recorded per attempt and do
not abort a run. A judge outage degrades the advisory dimension without failing a run
whose deterministic graders succeeded.

**Why.** Schema validity is a blocking gate signal, so an exception would destroy the
exact evidence the gate needs. And a run that dies on case 7 of 80 tells you nothing,
whereas a run with three recorded errors tells you precisely where the provider
struggled.

**Rejected: fail fast.** Correct for a request path, wrong for a measurement harness.
The harness's job is to observe failure, not to propagate it.

---

## D7 — Smoke tier selection is deterministic

**Decision.** The 20-case smoke subset is a pure function of the dataset, stratified over
category and difficulty, with every `critical` case included regardless of the size
target.

**Why.** With random sampling, cases enter and leave the subset between commits, and the
diff reports sampling churn as model change. The gate would fire on noise it generated
itself.

**Rejected: random sample per run.** Better expected coverage over many runs, useless for
run-to-run comparison — which is the only thing this system does.

**Rejected: first 20 by id.** Deterministic, but `tc_0001`–`tc_0020` are whatever got
written first, which in practice means the easy ones.

Proof: [`test_cli.py::test_smoke_selection_is_deterministic`](../tests/test_cli.py),
[`test_cli.py::test_critical_cases_are_not_dropped_to_hit_the_size`](../tests/test_cli.py).

---

## D8 — Ground truth is written by hand, and nothing in the codebase can generate it

**Decision.** No code path produces golden cases. The loader additionally rejects any
case that duplicates a few-shot example from any prompt version.

**Why.** A model-generated golden set measures whether the model agrees with itself. And
a case the model was shown in its own prompt measures recall of the prompt, not
capability — that one is easy to introduce by accident when a good few-shot example
seems like an obvious test case.

**Rejected: LLM-generated cases with human review.** Reviewing a plausible-looking
generated label is a far weaker act than writing the label from scratch; the anchor does
most of the work. This is the most expensive constraint in the project and the one that
makes the rest of it mean anything.

Proof: [`test_dataset.py::test_few_shot_leakage_rejected`](../tests/test_dataset.py) —
verified live against the real `v001` prompt.

---

## Deviations from the source guide, in one table

| Guide says | Built instead | Reason |
|---|---|---|
| RAGAS or DeepEval | Judge on the provider layer | D5 — accounting and fit |
| `instructor` for structured output | Native `json_schema` / forced tool use | Same objection as D5 |
| Warn >3%, block >8% | Effect size **and** McNemar exact | D3 — a percentage cannot answer "signal or noise?" |
| `notes` field | `notes` **required, non-empty** | An uninterpretable case is worse than no case |
| — | Judge calibration gate | D1 — the guide never validates its judge |
| — | Dataset hash in the comparison key | D4 — the guide has no defence against ground-truth drift |
| — | Tiered CI + cassettes | Full evals on every push cost more than they catch |
