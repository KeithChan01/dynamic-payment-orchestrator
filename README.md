# Dynamic Payment Orchestrator

**A decision helper for “which payment provider should we try first?”**

When a customer pays, a business often has several options: Stripe, PayPal, a bank transfer, Wise, and so on. They are not interchangeable. One may be cheaper, one more likely to go through, one faster at actually delivering the money.

This project looks at a payment (amount, currency, country, deposit vs withdrawal) and returns:

1. **The best first choice** for that payment
2. **Backup options** if the first one fails
3. **A plain-language reason** (success chance, fee, and how long settlement usually takes)

It does **not** take card numbers, move money, or talk to a real bank. Think of it as a ranking engine sitting *next to* a payment system, not the payment system itself.

> All scores come from **fake (synthetic) training data**. They are not real Stripe / PayPal / Wise performance. Do not use them as production benchmarks.

## A concrete example

Someone in Germany pays **$8,200 USD**, and the merchant wants euros.

The orchestrator does not blindly pick Stripe. It:

- drops providers that are not allowed for this corridor
- estimates “will this go through?”, “what will it cost?”, and “how long until the money arrives?”
- ranks the rest using a policy you choose: **balanced**, **cheapest**, **fastest**, or **highest approval**

On the same payment those policies can disagree — cheap often means slow (bank transfer), fast often costs more. That disagreement is the point of the demo.

Try it in the dashboard (see Quick start): pick a scenario such as “Weekend JPY withdrawal” and you will see some providers marked ineligible, with a reason.

## How a decision is made

```
Incoming payment
    → Who is even allowed?          (simple rules: currency, country, deposit vs withdrawal)
    → Score the allowed options     (machine learning + a fee table)
    → Rank them                     (you pick: cheap, fast, reliable, or a mix)
    → Return #1, then backups
```

Two kinds of logic on purpose:

| Kind | What it covers | Why it is not “all AI” |
|---|---|---|
| **Written rules** | “Square does not take JPY withdrawals in this demo.” Fee % + fixed fee. Ranking weights. | Coverage and pricing are business policy. They should not be guessed from a fake history. |
| **Learned patterns** | Which provider fits this context; chance of success; typical vs slow settlement time | Those are prediction problems, trained from historical-style tables. |

The machine learning library is scikit-learn. The orchestrator still makes the final call.

## What the data is

The project trains on the public [PayMind reference dataset](https://huggingface.co/datasets/navk8690/paymind-reference-data) on Hugging Face. It is three spreadsheets that look like payment logs, not one combined file:

| Spreadsheet | Question it helps answer |
|---|---|
| Which method was used | “Which provider looks relevant for this kind of payment?” |
| Did it succeed | “How likely is this provider to work?” |
| How long until arrival | “How long might settlement take?” (typical and a slower, more conservative estimate) |

The code loads each file from Hugging Face automatically. You do not download a zip.

Rows are trained on **older dates** and tested on **newer dates**. Mixing them at random would let the model peek at the future, which looks good in a screenshot and teaches the wrong lesson.

License: **GPL-3.0** (same as the dataset).

## Honest limits (read this)

The fake data is weakly informative. In this set almost every provider already succeeds about 90% of the time, so **cost and speed usually decide the ranking, not approval rate**. That is shown in the metrics below rather than hidden.

Do not quote these rankings as “Adyen is better than Stripe in the real world.” Retrain on your own processor logs before anything customer-facing.

Inspired by [PayMind](https://github.com/navjotk8690/paymind). This is a separate product-shaped demo, not a rename of that project.

---

## For developers

### Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Dashboard (easiest way to see it):

```powershell
streamlit run dashboard/app.py
```

API:

```powershell
dpo-api
# http://127.0.0.1:8080/docs
# POST /v1/route   rank one payment
# POST /v1/compare same payment, all four policies
# POST /v1/simulate 200 fake payments vs “always Stripe”, cheapest, and fastest
```

Until you train, it uses built-in averages so the demo still runs. To train from Hugging Face (files are cached; no manual CSV download):

```powershell
dpo-inspect
dpo-train
dpo-train --sample 20000   # faster iteration
pytest
```

Example `POST /v1/route` body:

```json
{
  "transaction_type": "deposit",
  "amount": 8200,
  "currency": "USD",
  "local_currency": "EUR",
  "country": "United States",
  "ip_country": "Germany",
  "is_cross_border": true,
  "policy": "balanced"
}
```

### Loading the three tables

They do not share columns. Loading the Hub repo as one dataset fails; load each file:

```python
from dpo.data import load_table

success = load_table("success")
arrival = load_table("arrival")
routes  = load_table("payment_method")
```

### Layout

```
config/     human-editable fees, eligibility, ranking policies
src/dpo/    data load, models, ranking, FastAPI
dashboard/  Streamlit UI
tests/      rules, ranking, API
```

### What training actually learned

Full tables, older→newer split:

| Model | What “good” means here | Result | Naive baseline |
|---|---|---|---|
| Which provider | Share of times it names the historical choice | 19% accuracy (52% in top 3) | Always guess the most common provider: 14% |
| Will it succeed | Ranking successful vs failed payments | AUC 0.58 | Coin-flip is 0.50 |
| Settlement time | Average error on typical arrival (minutes) | 113 | Always guess the median: 132 |

Weak lift is expected with this synthetic set. Fees and settlement still produce a useful ranking product.