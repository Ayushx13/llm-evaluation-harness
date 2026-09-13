# Support Ticket Classifier Evaluation

This project tests how well an LLM sorts customer support tickets, and where it breaks.

The system under test is a small classifier. It takes one e-commerce support ticket and returns JSON with four fields:

```json
{
  "category": "payment",
  "priority": "high",
  "confidence": 0.92,
  "evidence_span": "My card was charged twice for the same order."
}
```

Most of the code is the evaluation harness around it. The harness scores the classifier on hand-labelled tickets and on inputs built to trick it. It also checks whether the model invents evidence and whether its confidence matches how often it is right. It groups the failures by cause and compares two prompt versions case by case.

---

## What it measures

| Question | How the harness answers it | Code |
|---|---|---|
| Does it pick the right label? | Accuracy, per-class precision, recall and F1, and a confusion matrix | [evals/metrics.py](evals/metrics.py) |
| Does it hold up against tricky inputs? | Six families of adversarial inputs, each aimed at one suspected weakness | [evals/adversarial.py](evals/adversarial.py) |
| Does it make up evidence? | Checks that `evidence_span` really appears in the ticket | [evals/hallucination.py](evals/hallucination.py) |
| Does its confidence mean anything? | Calibration: when it says 0.9, is it right about 90% of the time? | [evals/calibration.py](evals/calibration.py) |
| Why does it fail? | Tags each failure with signals and assigns it to a named cluster with a proposed fix | [evals/clustering.py](evals/clustering.py) |
| Did a prompt change break anything? | Per-case comparison of two prompt versions | [evals/regression.py](evals/regression.py) |

---

## The task

**Categories:** `payment`, `delivery`, `account`, `product_defect`, `refund`, `other`.

**Priorities:**

| Priority | Rule |
|---|---|
| high | Money lost, access blocked, account compromised, item unusable or unsafe, or a deadline within 48 hours |
| medium | A real problem, but nothing is lost or blocked yet |
| low | Questions, feedback, and minor complaints |

**Evidence span.** The model must copy a piece of the ticket, character for character, that justifies its category. This makes grounding testable. If the span is not in the ticket, the model made up its own justification. The classifier never repairs a bad span, so the harness always sees what the model really returned.

---

## Project layout

```
app/
  schema.py          Output schema (Pydantic): category, priority, confidence, evidence_span
  prompts.py         Versioned system prompts (v1 baseline, v2 fixes). Never edited in place.
  classifier.py      Gemini call, response cache, rate limiting, retries
data/
  ground_truth.json  52 hand-labelled tickets, each with a written reason and a difficulty tag
  adversarial.json   222 adversarial cases generated from the ground truth
evals/
  adversarial.py     Builds data/adversarial.json
  runner.py          Runs the full suite and writes results/
  metrics.py         Accuracy, per-class scores, confusion matrix
  hallucination.py   Evidence span grounding check
  calibration.py     ECE, MCE, reliability diagram
  clustering.py      Failure clusters with hypotheses and fixes
  regression.py      v1 vs v2, case by case
report/
  evaluation_report.ipynb   Reads results/ and presents the findings. Never calls the model.
results/             Output of the runs (raw results, metrics, clusters, plots, regression report)
```

---

## Ground truth

[data/ground_truth.json](data/ground_truth.json) has 52 tickets. Each one has:

- the expected `category` and `priority`,
- a `label_reason` explaining the decision, and
- a `difficulty` tag: `easy`, `arguable` or `hard`.

Some tickets are hard on purpose. For example, "I did not receive the item, and I have already been charged for it" has both a payment and a delivery signal. The label policy settles such cases by root cause: here the parcel is missing, so the label is `delivery`.

One person labelled all 52 cases. Treat the `arguable` and `hard` labels as a judgement call, not a fact.

---

## Adversarial inputs

Each family targets one suspected weakness. A derived case always inherits its label from the human-labelled base case, so no LLM ever chooses a label.

| Family | Cases | What it changes | Weakness it targets |
|---|---|---|---|
| paraphrase | 52 | Rewords the ticket (LLM-written, cached) | Depending on exact wording |
| irrelevant_context | 52 | Adds chatter full of other categories' words ("my friend got a refund...") | Getting pulled by keywords |
| contradiction | 38 | Appends a fact that conflicts with the complaint | Handling conflicting evidence |
| negation | 20 | Adds a rejected keyword ("This is not a refund request.") | Reacting to a word, not its meaning |
| typos | 52 | Swaps and drops characters | Brittle inputs and broken span copying |
| ood | 8 | Replaces the ticket with an off-topic input (coding, trivia, gibberish, a prompt injection) | Being confident outside the task |

Everything except the paraphrases is template-based with a fixed seed (`SEED = 7`), so the same code produces the same cases.

---

## Hallucination check

A classifier with fixed labels can only make up two things: a label outside the schema, or an evidence span that is not in the ticket. The span check has three outcomes:

| Verdict | Meaning | Counts as hallucination? |
|---|---|---|
| `supported` | The span appears in the ticket, ignoring case and extra spaces | No |
| `paraphrased_unverified` | Not an exact match, but every content word is in the ticket and fuzzy similarity is at least 0.82 | No, but reported separately |
| `fabricated` | The span contains words that are not in the ticket | Yes |

**Limits:** a span can be copied correctly and still not justify the label, and the check cannot catch that. It also does not treat "400" and "four hundred" as the same.

---

## Reproducibility

- **Pinned model.** The default is `gemini-3.5-flash-lite`, not a `-latest` alias that could change without warning.
- **Temperature 0 and a fixed seed (42)**, with thinking set to the lowest level the API allows.
- **Disk cache.** Every response is saved in `.cache/`. The cache key covers the model, prompt text, input, generation settings and output schema. If any of those change, the harness makes a new call. It never reuses a stale answer.
- **Raw results first.** The runner writes every raw answer to `results/raw_results_<version>.json`, then computes metrics from that file. Fixing a bug in the metrics code does not need new API calls.
- **Prompts are append-only.** A prompt change adds a new version, so the v1 baseline always means the same thing.

---

## Setup

Requires Python 3.11 or newer, because the schema uses `StrEnum`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill it in:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_RPM=14
```

`GEMINI_RPM` caps live calls per minute. The free tier allows 15, and the default of 14 leaves some margin.

---

## How to run

```bash
# 1. Build the adversarial suite (calls the API for paraphrases only)
python -m evals.adversarial

# 2. Run the suite for each prompt version
python -m evals.runner --prompt v1
python -m evals.runner --prompt v2

# 3. Compare the two versions case by case
python -m evals.regression

# 4. Open the report
jupyter notebook report/evaluation_report.ipynb
```

Useful flags:

- `python -m evals.runner --normal-only` runs only the 52 ground-truth cases.
- `python -m evals.runner --limit 20` runs only the first 20 cases.
- `python -m evals.adversarial --no-llm` skips paraphrases, for use without an API key. This drops 52 cases and shifts every `adv_` ID, so the output will not match the committed `data/adversarial.json`.

The classifier also works on its own:

```bash
python -m app.classifier "I was charged twice for order #4471."
```

---

## Results

Both runs used `gemini-3.5-flash-lite` with seed 42.

### v1 (baseline prompt): complete, 274 of 274 cases scored

| Metric | Value |
|---|---|
| Category accuracy (all) | 92.0% |
| Category accuracy, clean inputs | 92.3% |
| Category accuracy, adversarial inputs | 91.9% |
| Priority accuracy | 61.3% |
| Both fields correct | 57.3% |
| Hallucination rate | 0.4% (1 fabricated span) |
| ECE (expected calibration error) | 0.031 |

Category accuracy by input type:

| Input type | Cases | Category accuracy | Mean confidence |
|---|---|---|---|
| typos | 52 | 86.5% | 0.94 |
| paraphrase | 52 | 90.4% | 0.93 |
| contradiction | 38 | 92.1% | 0.87 |
| normal | 52 | 92.3% | 0.94 |
| irrelevant_context | 52 | 94.2% | 0.91 |
| negation | 20 | 100% | 0.95 |
| ood | 8 | 100% | 0.69 |

Main findings on v1:

- **Priority is the weak spot, not category.** The model gets the category right 92% of the time but the priority only 61% of the time.
- **Typos hurt the most.** Typos are the worst input type, yet confidence stays at 0.94. The model does not notice when it struggles.
- **Failures look confident.** 21 of the 22 category failures were stated at confidence 0.85 or higher.
- **Most failures share one cause.** 14 of the 22 trace back to 4 hard clean tickets where two categories overlap (for example an app crash read as `product_defect`). Their paraphrase, typo and distractor variants failed the same way. The automatic clusterer files most of them under "surface-form brittleness", which the report corrects by hand.

### v2 (targeted prompt): complete, 274 of 274 cases scored

The v2 prompt adds explicit rules for negation, distractor chatter, contradictions, off-topic inputs, and root-cause tie-breaks between categories.

| Metric | v1 | v2 |
|---|---|---|
| Category accuracy (all) | 92.0% | 98.9% |
| Category accuracy, clean inputs | 92.3% | 100% |
| Category accuracy, adversarial inputs | 91.9% | 98.7% |
| Priority accuracy | 61.3% | 78.8% |
| Both fields correct | 57.3% | 78.1% |
| Hallucination rate | 0.4% (1 case) | 0.4% (1 case) |
| ECE | 0.031 | 0.129 |

Category accuracy by input type:

| Input type | Cases | v1 | v2 | v2 mean confidence |
|---|---|---|---|---|
| typos | 52 | 86.5% | 100% | 0.88 |
| paraphrase | 52 | 90.4% | 100% | 0.88 |
| contradiction | 38 | 92.1% | 97.4% | 0.79 |
| normal | 52 | 92.3% | 100% | 0.90 |
| irrelevant_context | 52 | 94.2% | 98.1% | 0.88 |
| negation | 20 | 100% | 100% | 0.95 |
| ood | 8 | 100% | 87.5% | 0.32 |

Regression check (`python -m evals.regression`): **21 cases fixed, 2 regressed**, and priority 58 fixed against 10 broken.

Main findings on v2:

- **The tie-break rules worked.** They fixed all 14 failures that traced back to overlapping categories, plus 7 others.
- **Priority is still the weakest field.** 44 of v2's 58 priority errors rate a ticket lower than its label, for example a refund two weeks overdue (`gt_035`) rated medium instead of high.
- **An empty ticket gets a confident, made-up answer.** For `adv_222` (input `""`), v2 answers `delivery` / `high` at 0.95 confidence, and its evidence span is a sentence copied from its own system prompt. This is the only hallucination in the v2 run and one of the two regressions.
- **Calibration got worse (ECE 0.031 → 0.129).** The prompt's "off-topic → confidence at most 0.3" rule also fired on 15 correct, in-scope `other` tickets such as job and internship questions. The other regression, `adv_118`, comes from the same rule.
- **These results are not held out.** The v2 rules were written after reading v1's failures on these same cases, so v2's 100% on clean inputs shows the rules fix the cases they were written for, not that they generalise.

The full analysis, including five failure clusters with causes and proposed fixes, is in [report/evaluation_report.ipynb](report/evaluation_report.ipynb).

---

## Limitations

- **Small dataset.** 52 base tickets and one labeller. One case is worth about 2% of clean accuracy.
- **One model.** All results are for `gemini-3.5-flash-lite`. Other models may fail differently.
- **Hand-picked attacks.** Six adversarial families cannot cover everything. Tickets that mix languages (for example Hindi and English), very long threads and multi-turn conversations are not tested.
- **LLM-written paraphrases.** The same model family that is being tested wrote them. They may be easier for it than human rewrites would be.
- **Shallow span check.** It confirms a span exists in the ticket, not that the span supports the label.
- **Calibration on few cases.** On v1, 233 of 274 answers have confidence of 0.9 or above, and only 7 fall below 0.8. The low buckets hold 1 to 3 cases each, so ECE and especially MCE are noisy.
- **Rule-based clusters.** Clusters come from keyword and input-type rules. A failure can land in a cluster whose hypothesis does not really explain it.
- **v2 tuned on the test data.** Its rules were written after reading v1's failures on the same 274 cases. A held-out set is needed to know whether the gains generalise.
- **LLM-drafted ground truth.** The 52 tickets were drafted with an LLM, then every case was reviewed and corrected by hand by one person.
- **Determinism is guaranteed by the cache, not measured.** Temperature 0 and a fixed seed are used, but the run-to-run variance of fresh API calls was not measured.
