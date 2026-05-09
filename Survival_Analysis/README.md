# Homework 3: Survival Analysis and Customer Lifetime Value

This project analyzes telecom subscriber churn using survival analysis.

The goal is to:
1. Fit parametric Accelerated Failure Time (AFT) survival models.
2. Compare available AFT distributions.
3. Select a final model based on statistical fit, interpretability, and business usefulness.
4. Keep statistically significant features.
5. Estimate customer lifetime value (CLV) using predicted survival probabilities.
6. Explore valuable and at-risk customer segments.
7. Estimate an annual retention budget based on at-risk customers.

## Dataset

The dataset is `telco.csv`.

Main variables:
- `tenure`: subscriber lifetime / survival duration.
- `churn`: event indicator; 1 means churned, 0 means censored.
- Demographics and service variables: region, age, marital status, income, education, gender, voice, internet, forwarding, customer category, etc.

## How to run

```bash
pip install -r requirements.txt
python survival_analysis.py