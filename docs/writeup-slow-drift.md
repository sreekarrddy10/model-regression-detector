# Your eval suite is probably blind to the way models actually get worse

Most teams that build an eval suite build the same one. A golden dataset, a pass rate, and
a comparison against the previous run. Prompt changes, accuracy drops four points, someone
gets alerted. It works, and it catches the thing it was designed to catch.

It is also structurally incapable of noticing the most common way a production LLM feature
degrades.

## Two different failures

A **per-run regression** is a step change. Someone edits a prompt, twelve cases flip, and
the diff against yesterday is unmissable. This is the failure everyone builds for, and
per-run diffing catches it well.

**Drift** is not a step. It is a slope. Accuracy goes 97, 96, 96, 95, 94, 93, 92. No
single run-to-run comparison exceeds a one-point threshold, so no alert ever fires. Six
weeks later the feature is five points worse than it was and nobody can point at the
change that did it — because there wasn't one. There were twenty, each individually
defensible.

The causes are mundane and mostly not your code:

- A provider silently updates the model behind a version alias.
- Twenty prompt tweaks each trade a little accuracy for a little latency.
- Real traffic shifts under a prompt that was tuned for last quarter's traffic.
- Few-shot examples that were representative in March aren't in September.

Each of those is invisible to a diff that only ever looks one run back.

## The fix is three lines and a decision

The mechanism is trivial — an exponentially weighted moving average over recent runs, with
a floor:

```python
def ewma(values, *, alpha=0.3):
    current = values[0]
    for value in values[1:]:
        current = alpha * value + (1 - alpha) * current
    return current
```

If the trend falls below the floor, warn — regardless of whether the current run regressed
against its baseline. The interesting part isn't the maths. It's three decisions around it.

**Only compare runs scored against identical ground truth.** A trend line is a claim about
the model over time. If someone relabels a few ambiguous cases in week three, the trend
silently becomes a claim about the model *and* the ruler. My runs carry a SHA-256 of the
golden dataset's semantic content, and the trend query filters on it. Runs against
different ground truth aren't averaged in — they're skipped.

**Drift warns, it never blocks.** There is no commit to blame. Blocking a PR because
accuracy has been sliding for a month punishes whoever happened to open it, and the gate
loses credibility the first time it does that. The trend is a signal to investigate, not
a verdict on a change.

**Weight recent runs more.** A flat mean over the last N runs is dominated by history and
reacts late. `alpha = 0.3` means the most recent run contributes 30%, and the trend turns
within a few runs of a real change rather than a few dozen.

## The half nobody builds

Here is the part I'd argue matters more than the EWMA.

Almost every eval suite has an LLM-as-judge scoring some open-ended quality dimension.
Almost none check whether that judge agrees with a human about anything.

An unvalidated judge doesn't fail loudly. It reports a stable, confident 4.2 that
correlates with nothing, and because the number is stable it looks like evidence that
quality is holding. You can watch a completely uncalibrated judge report a flat trend line
across a genuine six-point regression. The drift detector is measuring the judge's
consistency, not the model's quality — and consistency is exactly what a bad judge has
plenty of.

So: score twenty summaries by hand, once. Before each run reports a quality number, have
the judge score those same twenty and compute agreement — I use a quadratic-weighted
Cohen's κ, so a 5-vs-4 disagreement costs far less than a 5-vs-1. Below κ = 0.60, throw
the judge's scores away and say so.

Two traps in there, both of which I hit:

**A constant rater must score zero, not one.** If the judge rates everything 5, there's no
variance to explain and κ is genuinely undefined. Return 1.0 — which a naïve
"they-agree-on-every-sample" implementation does — and the most useless possible judge
passes calibration perfectly. Return 0.0 and it fails the floor, which is the safe
direction.

**The holdout needs a deliberate spread of quality.** Twenty summaries you'd all score 5
produce an undefined κ for the same reason. Include summaries that are wrong, that
hallucinate a detail, and that are technically accurate but useless. Agreement statistics
need disagreement to measure.

## What this changes about the alerting

Once drift is separate from per-run regression, alerts start carrying different meanings,
which is the actual win:

| Signal | What it means | Action |
|---|---|---|
| Critical case regressed | A specific change broke a specific thing | Block the merge |
| Significant aggregate drop | This change is worse, and it's not noise | Block the merge |
| Trend below floor | Something has been eroding for weeks | Investigate, don't block |
| Judge uncalibrated | You cannot interpret any quality number | Block, and fix the judge |

That last row is the one I'd defend hardest. Blocking because you *can't measure* feels
wrong the first time you write it. But a quality number you can't interpret is worse than
no quality number, because people act on it.

---

*From [model-regression-detection-system](../README.md) — CI/CD for model behavior. The
statistics live in [`stats.py`](../src/mrd/stats.py), the gate in
[`compare.py`](../src/mrd/compare.py), and the reasoning behind each threshold in
[DECISIONS.md](DECISIONS.md).*
