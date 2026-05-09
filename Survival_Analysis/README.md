# Homework 3: Survival Analysis and Customer Lifetime Value

## Author

Anna Asatryan

---

## Project Overview

This project analyzes **subscriber churn risk** using survival analysis and customer lifetime value estimation.

The analysis applies **parametric Accelerated Failure Time (AFT) models**, which model how customer characteristics affect expected survival time before churn. Four AFT distributions are fitted and compared: Weibull, Log-Normal, Log-Logistic, and Generalized Gamma Regression. The final model is selected based on model fit, interpretability, and decision usefulness. It is then used to estimate customer-level CLV and identify valuable and at-risk segments.

In this assignment, churn is treated as the event of interest:

* **tenure** – the observed customer lifetime in months
* **churn** – whether the subscriber churned (1) or was censored (0)
* **subscriber features** – demographic, service, and customer-category variables used to explain churn risk

---

## Repository Structure

```
DS223/
│
├── A_B_Testing/
├── Bass_Model/
└── Survival_Analysis/
    ├── survival_analysis.py        # Full pipeline: data prep, model fitting, CLV, report
    ├── analysis.ipynb              # Integrated notebook: code + results + report
    ├── telco.csv                   # Telecom subscriber dataset
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # This file
    ├── img/                        # Generated figures
    │   ├── aft_survival_curves.png
    │   ├── clv_by_segment.png
    │   └── risk_by_segment.png
    └── output/                     # Model results and final report
        ├── model_comparison.csv
        ├── final_model_summary.csv
        ├── coefficient_interpretation.csv
        ├── customer_clv.csv
        ├── segment_clv_summary.csv
        ├── retention_budget.csv
        └── report.md
```

---

## Dataset

Telecom subscriber dataset (`Survival_Analysis/telco.csv`).

| Column | Description |
|---|---|
| ID | Subscriber ID |
| region | Region code |
| tenure | Customer lifetime (months) |
| age | Subscriber age |
| marital | Marital status |
| address | Years at current address |
| income | Annual income (K) |
| ed | Education level |
| retire | Retirement status |
| gender | Gender |
| voice | Voice service |
| internet | Internet service |
| forward | Call forwarding |
| custcat | Customer category |
| churn | Churn indicator |

---

## Methodology

1. Load and clean the telecom subscriber dataset
2. Convert churn into a survival event indicator (1 = churned, 0 = censored)
3. Fit parametric AFT survival models (Weibull, Log-Normal, Log-Logistic, Generalized Gamma)
4. Compare models using AIC, BIC, log-likelihood, and concordance index
5. Visualize average survival curves in one plot
6. Select the final model based on fit and interpretability
7. Retain statistically significant predictors (α = 0.05)
8. Estimate customer-level CLV using conditional predicted survival probabilities
9. Explore CLV and churn risk across customer segments
10. Estimate an annual retention budget for at-risk subscribers

**Note on model coverage:** Piecewise exponential regression was not included because it requires externally specified breakpoints rather than a single distributional AFT form — a different modeling approach rather than an additional AFT distribution.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/anna-asatryan/DS223.git
cd DS223
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Mac / Linux
```

Install required packages:

```bash
pip install -r Survival_Analysis/requirements.txt
```

---

## Running the Project

**Option 1 — run the pipeline script:**

```bash
python Survival_Analysis/survival_analysis.py
```

**Option 2 — run the notebook** (shows code + outputs inline):

```bash
jupyter notebook Survival_Analysis/analysis.ipynb
```

Both options reproduce the full analysis end-to-end.

---

## Outputs

| File | Description |
|---|---|
| `output/report.md` | Written report with interpretations |
| `output/model_comparison.csv` | AIC/BIC/concordance for all fitted models |
| `output/final_model_summary.csv` | Coefficients of the final model |
| `output/coefficient_interpretation.csv` | Time ratios and directions |
| `output/customer_clv.csv` | Per-customer CLV and churn risk |
| `output/segment_clv_summary.csv` | Segment-level CLV and risk summaries |
| `output/retention_budget.csv` | At-risk count and recommended budget |
| `img/aft_survival_curves.png` | All AFT survival curves on one plot |
| `img/clv_by_segment.png` | Top segments by mean 36-month CLV |
| `img/risk_by_segment.png` | Top segments by 12-month churn risk |

---

## Course

**DS223 – Marketing Analytics** | Survival Analysis and CLV Assignment
