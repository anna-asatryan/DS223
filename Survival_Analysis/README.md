# Homework 3: Survival Analysis and Customer Lifetime Value

## Author

Anna Asatryan

---

## Project Overview

This project analyzes **subscriber churn risk** using survival analysis and customer lifetime value estimation.

The analysis applies **parametric Accelerated Failure Time (AFT) models**, which model how customer characteristics affect expected survival time before churn.

In this assignment, churn is treated as the event of interest:

* **tenure** – the observed customer lifetime.
* **churn** – whether the subscriber churned or was censored.
* **subscriber features** – demographic, service, and customer-category variables used to explain churn risk.

Several AFT distributions are fitted and compared. The final model is selected based on model fit, interpretability, and decision usefulness. The selected model is then used to estimate customer-level CLV and identify valuable and at-risk customer segments.

---

## Repository Structure

```
DS223/
│
├── A_B_Testing/
├── Bass_Model/
└── Survival_Analysis/
    ├── survival_analysis.py        # Main pipeline script
    ├── telco.csv                   # Telecom subscriber dataset
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Project documentation
    ├── img/                        # Generated figures and plots
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

The dataset is a telecom subscriber dataset stored in:  `Survival_Analysis/telco.csv`.


Main variables include:

* **ID** – subscriber ID
* **region** – region code
* **tenure** – customer lifetime
* **age** – subscriber age
* **marital** – marital status
* **address** – number of years living at the same address
* **income** – subscriber annual income
* **ed** – education level
* **retire** – retirement status
* **gender** – subscriber gender
* **voice** – voice service usage
* **internet** – internet service usage
* **forward** – call forwarding service
* **custcat** – customer category
* **churn** – churn indicator

---

## Methodology

The project follows these steps:

1. Load and clean the telecom subscriber dataset.
2. Convert churn into a survival event indicator.
3. Fit available parametric AFT survival models.
4. Compare models using AIC, BIC, log-likelihood, and concordance index.
5. Visualize average survival curves in one plot.
6. Select the final model from a decision-maker perspective.
7. Retain statistically significant features.
8. Estimate customer-level CLV using predicted survival probabilities.
9. Explore CLV and churn risk across customer segments.
10. Estimate an annual retention budget for at-risk subscribers.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/anna-asatryan/DS223.git
cd DS223
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment.

Mac / Linux:

```bash
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r Survival_Analysis/requirements.txt
```

---

## Running the Project

Run the full analysis:

```bash
python Survival_Analysis/survival_analysis.py
```

The script reproduces the entire analysis:

- loading and preparing the dataset
- fitting AFT survival models
- comparing model performance
- generating survival curve visualizations
- selecting the final model
- keeping significant features
- calculating customer-level CLV
- analyzing valuable and risky segments
- estimating the retention budget
- writing the final report

---

## Output

The project generates model comparison tables, CLV estimates, segment summaries, plots, and a markdown report.
Generated figures are stored in:

```
Survival_Analysis/img/
```

Generated output tables and the final report are stored in:

```
Survival_Analysis/output/
```


Main outputs include:

- AFT model comparison
- survival curves for fitted models
- final model summary
- coefficient interpretation table
- customer-level CLV estimates
- segment-level CLV and churn-risk summaries
- annual retention budget estimate
- final written report


The final report is available at:
```
Survival_Analysis/output/report.md
```

---

## Course
**DS223 – Marketing Analytics**

Survival Analysis and CLV Assignment
