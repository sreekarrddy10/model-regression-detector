# Authoring the Golden Dataset

Ground truth is written by hand. Never generate these cases with an LLM — the entire
premise of the project is that the eval bar is human-verified. A model-generated golden
set measures whether the model agrees with itself.

## Workflow

```bash
python -m mrd.dataset new --id tc_0007 >> data/golden/emails.jsonl   # blank row
$EDITOR data/golden/emails.jsonl                                     # fill it in
make dataset-validate                                                # errors, with line numbers
make dataset-report                                                  # coverage gaps
make dataset-lock VERSION=v1                                         # freeze it
```

## Fields

| Field | Notes |
|---|---|
| `id` | `tc_NNNN`, stable forever. Never renumber — run history references it. |
| `input_email` | The raw email. Write it as a real person would, typos included. |
| `expected_category` | `billing` · `technical` · `account` · `general` |
| `expected_summary` | One sentence, customer's point of view, under 25 words. |
| `difficulty` | `easy` · `ambiguous` · `adversarial` |
| `critical` | `true` = a regression here blocks the merge on its own. Reserve for behavior that must never break. |
| `source` | `handwritten` now; `from_failure` for cases harvested from production failures later. |
| `notes` | Required. Why this case exists. Write it for whoever debugs the failure in six months. |
| `added_at` | ISO 8601 UTC. |

## Coverage targets

| Target | Count |
|---|---|
| Total cases | 80 |
| Per category | ≥ 12 |
| Ambiguous | ≥ 12 |
| Adversarial | ≥ 8 |
| Critical | ≥ 10 |
| Judge holdout | 20 |

`make dataset-report` shows progress against each.

## Edge cases worth writing deliberately

- Two categories genuinely apply (a bug caused a wrong charge — billing or technical?)
- Five words total, no context
- Heavy typos, autocorrect damage, phone-keyboard mangling
- Mixed language, or English written by a non-native speaker
- Sarcastic or angry, where the literal reading misleads
- Empty body, subject line only
- Long thread with the actual request buried at the bottom
- A request the taxonomy genuinely does not cover

Every edge case needs a `notes` line explaining the intended reading. Where the tie-break
rules in `prompts/classifier/v001.yaml` decide the answer, say which rule and why.

## Two rules the validator enforces

**No leakage.** A case may not duplicate a few-shot example from any prompt version. The model
was shown those answers, so such a case measures recall of the prompt, not capability.

**No duplicate emails.** Near-identical inputs inflate the apparent sample size and skew the
significance test in Phase 3.

## The judge holdout

`judge_holdout.jsonl` holds 20 summaries scored 1–5 by a human. It exists to measure whether the
LLM-as-judge agrees with a person; below κ = 0.60 the judge is noise and the eval run aborts.

Score a **spread** of quality deliberately — include summaries that are wrong, that hallucinate a
detail, and that are technically accurate but useless. A holdout scored 5/5 across the board gives
an undefined κ, and the judge would pass calibration without ever being tested. The validator warns
when the spread is too narrow.
