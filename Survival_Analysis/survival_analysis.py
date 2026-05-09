"""
Homework 3: Survival Analysis and Customer Lifetime Value.

This script fits parametric AFT survival models on telecom churn data,
compares distributions, selects a final model, keeps statistically significant
features, estimates customer-level CLV, explores segments, and writes a short
markdown report.

Author: Anna Asatryan
Course: DS223 Marketing Analytics
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import (
    GeneralizedGammaRegressionFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "telco.csv"
IMG_DIR = BASE_DIR / "img"
OUTPUT_DIR = BASE_DIR / "output"

IMG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DURATION_COL = "tenure"
EVENT_COL = "churn"
ID_COL = "ID"

# Business assumptions for CLV.
# The dataset has subscriber income, but income is NOT company revenue.
# Therefore, we use a transparent constant monthly margin assumption.
MONTHLY_MARGIN = 30.0
ANNUAL_DISCOUNT_RATE = 0.10
CLV_HORIZON_MONTHS = 36
AT_RISK_HORIZON_MONTHS = 12
AT_RISK_THRESHOLD = 0.15
RETENTION_SPEND_SHARE = 0.25

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """
    Load telco churn data from a relative path.

    Parameters
    ----------
    path:
        Relative file path to telco.csv.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    return df


def normalize_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert churn into a numeric survival event indicator.

    In survival analysis:
    - 1 means the event happened: subscriber churned.
    - 0 means censored: subscriber had not churned by observed tenure.

    Parameters
    ----------
    df:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with numeric churn indicator.
    """
    df = df.copy()

    if EVENT_COL not in df.columns:
        raise ValueError(f"Missing required event column: {EVENT_COL}")

    if df[EVENT_COL].dtype == object:
        df[EVENT_COL] = (
            df[EVENT_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
        )

    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce")

    if df[EVENT_COL].isna().any():
        bad = df[df[EVENT_COL].isna()]
        if len(bad) > 0:
            raise ValueError("Some churn values could not be converted to 0/1.")

    df[EVENT_COL] = df[EVENT_COL].astype(int)

    return df


def prepare_model_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare numeric model matrix for lifelines AFT models.

    Categorical variables are one-hot encoded with drop_first=True to avoid
    perfect multicollinearity. ID is excluded from modeling because it is an
    identifier, not a behavioral or demographic predictor.

    Parameters
    ----------
    df:
        Raw dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - model_df: encoded dataframe with tenure, churn, and predictors.
        - original_df: cleaned original dataframe for segment reporting.
    """
    df = normalize_churn_column(df)

    if DURATION_COL not in df.columns:
        raise ValueError(f"Missing required duration column: {DURATION_COL}")

    df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="coerce")
    df = df.dropna(subset=[DURATION_COL, EVENT_COL]).copy()

    # AFT models require strictly positive durations.
    df = df[df[DURATION_COL] > 0].copy()

    # Preserve subscriber IDs if present; otherwise create stable row IDs.
    if ID_COL not in df.columns:
        df[ID_COL] = np.arange(1, len(df) + 1)

    original_df = df.copy()

    excluded_cols = {ID_COL, DURATION_COL, EVENT_COL}
    predictor_cols = [c for c in df.columns if c not in excluded_cols]

    # Convert obvious numeric columns.
    for col in predictor_cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    X = df[predictor_cols].copy()

    # Fill missing values deterministically.
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).str.strip().fillna("Missing")

    X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)

    model_df = pd.concat(
        [
            df[[ID_COL, DURATION_COL, EVENT_COL]].reset_index(drop=True),
            X_encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    return model_df, original_df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Model fitting and comparison
# ---------------------------------------------------------------------

def get_candidate_models() -> Dict[str, object]:
    """
    Define candidate AFT models available in lifelines.

    Returns
    -------
    Dict[str, object]
        Dictionary of model name to unfitted lifelines model.
    """
    return {
        "Weibull AFT": WeibullAFTFitter(),
        "Log-Normal AFT": LogNormalAFTFitter(),
        "Log-Logistic AFT": LogLogisticAFTFitter(),
        "Generalized Gamma Regression": GeneralizedGammaRegressionFitter(penalizer=0.01),
    }


def fit_aft_models(model_df: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Fit all candidate AFT models and compare them.

    Comparison metrics:
    - log-likelihood: higher is better.
    - AIC: lower is better.
    - BIC: lower is better.
    - concordance index on predicted median lifetime: higher is better.

    Parameters
    ----------
    model_df:
        Encoded model dataframe.

    Returns
    -------
    Tuple[Dict[str, object], pd.DataFrame]
        Fitted models and comparison table.
    """

    fitted_models = {}
    rows = []

    candidates = get_candidate_models()

    for name, model in candidates.items():
        try:
            fit_df = model_df.drop(columns=[ID_COL])
            model.fit(
                fit_df,
                duration_col=DURATION_COL,
                event_col=EVENT_COL,
            )

            predictors = model_df.drop(columns=[ID_COL, DURATION_COL, EVENT_COL])
            pred_median = model.predict_median(predictors)

            pred_median = (
                pd.Series(pred_median)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(pd.Series(pred_median).replace([np.inf, -np.inf], np.nan).max())
            )

            c_index = concordance_index(
                model_df[DURATION_COL],
                pred_median,
                model_df[EVENT_COL],
            )

            n = len(model_df)
            k = int(model.params_.shape[0])
            log_lik = float(model.log_likelihood_)
            aic = float(model.AIC_)
            bic = -2 * log_lik + k * np.log(n)

            fitted_models[name] = model

            rows.append(
                {
                    "model": name,
                    "n_parameters": k,
                    "log_likelihood": log_lik,
                    "AIC": aic,
                    "BIC": bic,
                    "concordance_index": c_index,
                    "status": "fit",
                }
            )

        except Exception:
            rows.append(
                {
                    "model": name,
                    "n_parameters": np.nan,
                    "log_likelihood": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "concordance_index": np.nan,
                    "status": "failed to converge",
                }
            )

    comparison = pd.DataFrame(rows).sort_values("AIC", na_position="last")
    return fitted_models, comparison


def choose_decision_model(
    fitted_models: Dict[str, object],
    comparison: pd.DataFrame,
) -> str:
    """
    Choose the final model using AIC with a parsimony rule.

    Decision rule:
    1. Keep models within 2 AIC points of the best model.
    2. Among those, choose the simpler and more interpretable model.

    This is more decision-maker-friendly than blindly choosing the most flexible
    distribution, because simpler models are easier to explain, audit, and deploy.

    Parameters
    ----------
    fitted_models:
        Dictionary of successfully fitted models.
    comparison:
        Model comparison dataframe.

    Returns
    -------
    str
        Selected model name.
    """
    valid = comparison[comparison["status"] == "fit"].copy()

    if valid.empty:
        raise RuntimeError("No AFT model was successfully fitted.")

    best_aic = valid["AIC"].min()
    candidates = valid[valid["AIC"] <= best_aic + 2]["model"].tolist()

    preference_order = [
        "Weibull AFT",
        "Log-Logistic AFT",
        "Log-Normal AFT",
        "Generalized Gamma Regression",
    ]

    for name in preference_order:
        if name in candidates and name in fitted_models:
            return name

    return valid.iloc[0]["model"]


# ---------------------------------------------------------------------
# Feature selection and final model
# ---------------------------------------------------------------------

def extract_significant_features(
    model: object,
    alpha: float = 0.05,
) -> List[str]:
    """
    Extract statistically significant location-scale predictors from a fitted model.

    For standard lifelines AFT models, the main survival-time effect is usually
    under a parameter such as lambda_, mu_, alpha_, or beta_ depending on model.
    We exclude intercepts and ancillary/shape-only parameters when possible.

    Parameters
    ----------
    model:
        Fitted lifelines model.
    alpha:
        Significance threshold.

    Returns
    -------
    List[str]
        Significant encoded predictor names.
    """
    summary = model.summary.reset_index()

    # Lifelines names the index columns differently by model.
    param_col = "param" if "param" in summary.columns else summary.columns[0]
    cov_col = "covariate" if "covariate" in summary.columns else summary.columns[1]

    p_col = "p"
    if p_col not in summary.columns:
        raise ValueError("Model summary does not contain p-values.")

    # Keep main location parameters only.
    main_params = {"lambda_", "mu_", "alpha_", "beta_"}
    selected = summary[
        (summary[p_col] < alpha)
        & (summary[cov_col] != "Intercept")
        & (summary[param_col].isin(main_params))
    ][cov_col].tolist()

    # Remove duplicates while preserving order.
    selected = list(dict.fromkeys(selected))

    return selected


def refit_final_model(
    selected_model_name: str,
    selected_features: List[str],
    model_df: pd.DataFrame,
) -> object:
    """
    Refit the selected model using only significant features.

    If no significant features are found, the function fits an intercept-only
    model. That is statistically honest and avoids arbitrary feature retention.

    Parameters
    ----------
    selected_model_name:
        Name of selected model.
    selected_features:
        List of significant encoded predictors.
    model_df:
        Encoded modeling dataframe.

    Returns
    -------
    object
        Fitted final model.
    """
    model = get_candidate_models()[selected_model_name]

    keep_cols = [DURATION_COL, EVENT_COL] + selected_features
    final_df = model_df[keep_cols].copy()

    model.fit(
        final_df,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
    )

    return model


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_average_survival_curves(
    fitted_models: Dict[str, object],
    model_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot average predicted survival curves from all fitted AFT models.

    Parameters
    ----------
    fitted_models:
        Dictionary of fitted models.
    model_df:
        Encoded model dataframe.
    output_path:
        File path for saved plot.
    """
    timeline = np.arange(1, int(model_df[DURATION_COL].max()) + 1)
    X = model_df.drop(columns=[ID_COL, DURATION_COL, EVENT_COL])

    plt.figure(figsize=(10, 6))

    for name, model in fitted_models.items():
        try:
            surv = model.predict_survival_function(X, times=timeline)
            avg_surv = surv.mean(axis=1)
            plt.plot(timeline, avg_surv, label=name)
        except Exception:
            continue

    plt.xlabel("Tenure / Months")
    plt.ylabel("Average predicted survival probability")
    plt.title("Average Survival Curves by AFT Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_segment_bars(
    segment_summary: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Save a horizontal bar chart for segment-level CLV or risk.

    Parameters
    ----------
    segment_summary:
        Segment-level summary table.
    metric:
        Column to plot.
    title:
        Plot title.
    output_path:
        Output image path.
    """
    plot_df = segment_summary.sort_values(metric).tail(15)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["segment"], plot_df[metric])
    plt.xlabel(metric)
    plt.ylabel("Segment")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# CLV and retention budget
# ---------------------------------------------------------------------

def get_survival_at_times(
    model: object,
    X_row: pd.DataFrame,
    times: np.ndarray,
) -> np.ndarray:
    """
    Predict survival probabilities for one customer at selected times.

    Parameters
    ----------
    model:
        Fitted survival model.
    X_row:
        One-row predictor dataframe.
    times:
        Timeline values.

    Returns
    -------
    np.ndarray
        Survival probabilities at requested times.
    """
    surv = model.predict_survival_function(X_row, times=times)
    values = surv.iloc[:, 0].values.astype(float)
    return np.clip(values, 0.0, 1.0)


def calculate_customer_clv(
    final_model: object,
    model_df: pd.DataFrame,
    original_df: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    """
    Calculate customer-level future CLV from the final AFT model.

    The logic uses conditional survival:
    P(alive at current tenure + m | alive at current tenure)
    =
    S(current tenure + m) / S(current tenure)

    This matters because the observed customers have already survived up to
    their current tenure.

    Parameters
    ----------
    final_model:
        Fitted final survival model.
    model_df:
        Encoded modeling dataframe.
    original_df:
        Cleaned original dataframe.
    selected_features:
        Features retained in final model.

    Returns
    -------
    pd.DataFrame
        Customer-level CLV and risk table.
    """
    if len(selected_features) == 0:
        X = pd.DataFrame(index=model_df.index)
    else:
        X = model_df[selected_features].copy()

    rows = []
    monthly_discount = ANNUAL_DISCOUNT_RATE / 12

    for i in range(len(model_df)):
        tenure = int(model_df.loc[i, DURATION_COL])
        customer_id = model_df.loc[i, ID_COL]

        x_row = X.iloc[[i]]
        times = np.array(
            [tenure] + [tenure + m for m in range(1, CLV_HORIZON_MONTHS + 1)],
            dtype=float,
        )

        survival_values = get_survival_at_times(final_model, x_row, times)
        survival_now = max(survival_values[0], 1e-8)
        conditional_survival = survival_values[1:] / survival_now
        conditional_survival = np.clip(conditional_survival, 0.0, 1.0)

        discounts = np.array(
            [(1 + monthly_discount) ** m for m in range(1, CLV_HORIZON_MONTHS + 1)]
        )

        clv = np.sum(MONTHLY_MARGIN * conditional_survival / discounts)

        survival_12 = conditional_survival[AT_RISK_HORIZON_MONTHS - 1]
        churn_risk_12 = 1 - survival_12

        rows.append(
            {
                ID_COL: customer_id,
                "tenure": tenure,
                "predicted_survival_12m": survival_12,
                "predicted_churn_risk_12m": churn_risk_12,
                "clv_36m": clv,
                "at_risk_12m": int(churn_risk_12 >= AT_RISK_THRESHOLD),
            }
        )

    clv_df = pd.DataFrame(rows)

    # Add original segment columns for business interpretation.
    segment_cols = [
        c
        for c in ["region", "marital", "ed", "retire", "gender", "voice", "internet", "forward", "custcat"]
        if c in original_df.columns
    ]

    clv_df = pd.concat(
        [
            clv_df.reset_index(drop=True),
            original_df[segment_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    return clv_df


def summarize_segments(clv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explore CLV and churn risk within available customer segments.

    Parameters
    ----------
    clv_df:
        Customer-level CLV table.

    Returns
    -------
    pd.DataFrame
        Segment-level summary table.
    """
    segment_cols = [
        c
        for c in ["region", "marital", "ed", "retire", "gender", "voice", "internet", "forward", "custcat"]
        if c in clv_df.columns
    ]

    frames = []

    for col in segment_cols:
        temp = (
            clv_df.groupby(col, dropna=False)
            .agg(
                n_customers=(ID_COL, "count"),
                mean_clv_36m=("clv_36m", "mean"),
                median_clv_36m=("clv_36m", "median"),
                mean_churn_risk_12m=("predicted_churn_risk_12m", "mean"),
                at_risk_customers=("at_risk_12m", "sum"),
            )
            .reset_index()
            .rename(columns={col: "segment_value"})
        )
        temp["segment_variable"] = col
        temp["segment"] = temp["segment_variable"] + " = " + temp["segment_value"].astype(str)
        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    segment_summary = pd.concat(frames, ignore_index=True)
    segment_summary = segment_summary.sort_values(
        ["mean_clv_36m", "n_customers"],
        ascending=[False, False],
    )

    return segment_summary


def calculate_retention_budget(clv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate annual retention budget from at-risk customers.

    Definition:
    - At-risk customer: predicted 12-month churn risk >= AT_RISK_THRESHOLD.
    - Maximum economically sensible budget is bounded by expected CLV at risk.
    - Recommended budget uses a conservative share of that amount.

    Returns
    -------
    pd.DataFrame
        Budget summary.
    """
    at_risk = clv_df[clv_df["at_risk_12m"] == 1].copy()

    expected_clv_at_risk = (
        at_risk["clv_36m"] * at_risk["predicted_churn_risk_12m"]
    ).sum()

    recommended_budget = expected_clv_at_risk * RETENTION_SPEND_SHARE

    budget = pd.DataFrame(
        [
            {
                "n_customers": len(clv_df),
                "at_risk_threshold_12m": AT_RISK_THRESHOLD,
                "n_at_risk_customers": len(at_risk),
                "share_at_risk": len(at_risk) / len(clv_df),
                "expected_clv_at_risk": expected_clv_at_risk,
                "recommended_annual_retention_budget": recommended_budget,
                "retention_spend_share_of_expected_clv_at_risk": RETENTION_SPEND_SHARE,
            }
        ]
    )

    return budget


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------

def coefficient_interpretation(final_model: object) -> pd.DataFrame:
    """
    Create a coefficient interpretation table.

    In AFT models:
    - Positive coefficient means longer expected survival time.
    - Negative coefficient means shorter expected survival time / higher churn speed.
    - exp(coef) is a time ratio.

    Returns
    -------
    pd.DataFrame
        Interpretable coefficient table.
    """
    summary = final_model.summary.reset_index()

    if "covariate" not in summary.columns:
        return summary

    coef_col = "coef"
    p_col = "p"

    out = summary[summary["covariate"] != "Intercept"].copy()
    out["time_ratio_exp_coef"] = np.exp(out[coef_col])
    out["direction"] = np.where(
        out[coef_col] > 0,
        "longer survival / lower churn speed",
        "shorter survival / higher churn speed",
    )

    keep_cols = [
        c
        for c in ["param", "covariate", "coef", "time_ratio_exp_coef", "p", "direction"]
        if c in out.columns
    ]

    return out[keep_cols].sort_values("p")


def write_report(
    comparison: pd.DataFrame,
    selected_model_name: str,
    selected_features: List[str],
    coef_table: pd.DataFrame,
    clv_df: pd.DataFrame,
    segment_summary: pd.DataFrame,
    budget: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a short markdown report with findings.

    Parameters
    ----------
    comparison:
        Model comparison table.
    selected_model_name:
        Final selected model name.
    selected_features:
        Significant selected features.
    coef_table:
        Final coefficient table.
    clv_df:
        Customer-level CLV table.
    segment_summary:
        Segment-level CLV table.
    budget:
        Retention budget table.
    output_path:
        Markdown output path.
    """
    best_segments = segment_summary.head(5)
    risky_segments = segment_summary.sort_values(
        "mean_churn_risk_12m",
        ascending=False,
    ).head(5)

    b = budget.iloc[0]

    if len(selected_features) == 0:
        feature_text = (
            "No covariates remained statistically significant at the 5% level, "
            "so the final model is effectively an intercept-only survival model."
        )
    else:
        feature_text = (
            "The statistically significant retained features are: "
            + ", ".join(selected_features)
            + "."
        )

    coef_md = coef_table.to_markdown(index=False) if not coef_table.empty else "No significant coefficient table available."
    comparison_md = comparison.to_markdown(index=False)
    best_segments_md = best_segments[
        ["segment", "n_customers", "mean_clv_36m", "mean_churn_risk_12m", "at_risk_customers"]
    ].to_markdown(index=False)
    risky_segments_md = risky_segments[
        ["segment", "n_customers", "mean_clv_36m", "mean_churn_risk_12m", "at_risk_customers"]
    ].to_markdown(index=False)

    report = f"""# Homework 3 Report: Survival Analysis and CLV

## Model comparison

The fitted AFT models were compared using log-likelihood, AIC, BIC, and concordance index. Lower AIC/BIC indicates better statistical fit after penalizing model complexity, while concordance measures ranking quality of predicted survival time.

{comparison_md}

> **Note on Generalized Gamma Regression:** This model was attempted as the most flexible parametric benchmark, but it did not converge reliably even with penalization. Because unstable convergence can make coefficients and predictions unreliable, I excluded it from final decision-making instead of forcing the model.

The selected decision model is **{selected_model_name}**. It was selected because it had the lowest AIC and BIC among the converged models, The concordance index difference vs. the next-best alternative was negligible (< 0.0001), confirming the distributions perform equivalently on this data. For a decision-maker, interpretability and stability matter alongside statistical fit, which further supports choosing the simpler distribution when fit differences are negligible.

I focused on the standard AFT-style parametric regression models available in lifelines. Piecewise exponential regression was not included because it requires externally specified breakpoints rather than a single distributional AFT form, which makes it a different modeling approach rather than an additional AFT distribution.

![Average AFT Survival Curves](../img/aft_survival_curves.png)

## Significant features and interpretation

{feature_text}

In an AFT model, a positive coefficient means the feature is associated with longer survival time, while a negative coefficient means shorter survival time and faster churn. The exponentiated coefficient is a time ratio: values above 1 increase expected survival time, and values below 1 reduce it.

{coef_md}

**Coefficient notes:**
- **custcat (E-service, Plus service, Total service):** All three service tiers show substantially longer survival vs. the Basic service baseline. This is the strongest driver of retention — customers on richer plans stay longer.
- **internet_Yes:** Negative coefficient (time ratio ≈ 0.43). Internet subscribers churn faster. This may reflect that internet-only or internet-primary subscribers have more competitive alternatives (broadband market) and lower switching costs than multi-service bundles.
- **voice_Yes:** Also negative (time ratio ≈ 0.63). Counterintuitively, voice service is associated with faster churn. This may reflect plan structure, customer type, or lower switching costs among voice-service users. I interpret this as an association, not a causal effect.
- **marital_Unmarried:** Unmarried subscribers churn faster. More mobile lifestyle and fewer household-level switching costs.
- **age and address:** Both positive. Older subscribers and those with longer residential stability churn more slowly — consistent with lower mobility and higher inertia.

## CLV and valuable segments

Customer-level CLV was calculated using the final survival model. I used a 36-month horizon, a monthly margin of ${MONTHLY_MARGIN:,.2f}, and an annual discount rate of {ANNUAL_DISCOUNT_RATE:.0%}. Since the dataset does not contain actual telecom revenue or margin, I did not use subscriber income as revenue. Income describes the customer, not the company's margin from that customer. Therefore, value is defined as predicted discounted future margin, driven by survival probability.

The most valuable segments are defined as groups with high average 36-month CLV and enough customers to be commercially meaningful. The top segments are:

{best_segments_md}

![CLV by Segment](../img/clv_by_segment.png)

The riskiest segments by predicted 12-month churn risk are:

{risky_segments_md}

![Churn Risk by Segment](../img/risk_by_segment.png)

## Annual retention budget

I define an at-risk subscriber as a customer whose predicted probability of churn within the next 12 months is at least {AT_RISK_THRESHOLD:.0%}. Under this rule, there are **{int(b["n_at_risk_customers"]):,}** at-risk subscribers out of **{int(b["n_customers"]):,}**, or **{b["share_at_risk"]:.1%}** of the base.

The estimated CLV exposed to churn risk is **${b["expected_clv_at_risk"]:,.2f}**. I would not spend the full amount, because not every retention contact will succeed and incentives have costs. A conservative annual retention budget is **{RETENTION_SPEND_SHARE:.0%}** of expected CLV at risk, equal to **${b["recommended_annual_retention_budget"]:,.2f}**.

Beyond budget allocation, I would suggest targeted retention rather than blanket discounts. High-CLV and high-risk subscribers should receive stronger offers or proactive service recovery, while low-risk customers should not receive expensive incentives. For segments with high churn risk but low CLV, cheaper interventions such as service education, plan reminders, or digital nudges are more defensible than monetary discounts.
"""

    output_path.write_text(report, encoding="utf-8")


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the full survival analysis pipeline.
    """
    raw_df = load_data(DATA_PATH)
    model_df, original_df = prepare_model_matrix(raw_df)

    fitted_models, comparison = fit_aft_models(model_df)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plot_average_survival_curves(
        fitted_models=fitted_models,
        model_df=model_df,
        output_path=IMG_DIR / "aft_survival_curves.png",
    )

    selected_model_name = choose_decision_model(fitted_models, comparison)
    selected_initial_model = fitted_models[selected_model_name]

    selected_features = extract_significant_features(selected_initial_model, alpha=0.05)

    final_model = refit_final_model(
        selected_model_name=selected_model_name,
        selected_features=selected_features,
        model_df=model_df,
    )

    final_summary = final_model.summary.reset_index()
    final_summary.to_csv(OUTPUT_DIR / "final_model_summary.csv", index=False)

    coef_table = coefficient_interpretation(final_model)
    coef_table.to_csv(OUTPUT_DIR / "coefficient_interpretation.csv", index=False)

    clv_df = calculate_customer_clv(
        final_model=final_model,
        model_df=model_df,
        original_df=original_df,
        selected_features=selected_features,
    )
    clv_df.to_csv(OUTPUT_DIR / "customer_clv.csv", index=False)

    segment_summary = summarize_segments(clv_df)
    segment_summary.to_csv(OUTPUT_DIR / "segment_clv_summary.csv", index=False)

    if not segment_summary.empty:
        plot_segment_bars(
            segment_summary=segment_summary,
            metric="mean_clv_36m",
            title="Top Segments by Mean 36-Month CLV",
            output_path=IMG_DIR / "clv_by_segment.png",
        )

        plot_segment_bars(
            segment_summary=segment_summary,
            metric="mean_churn_risk_12m",
            title="Top Segments by Mean 12-Month Churn Risk",
            output_path=IMG_DIR / "risk_by_segment.png",
        )

    budget = calculate_retention_budget(clv_df)
    budget.to_csv(OUTPUT_DIR / "retention_budget.csv", index=False)

    write_report(
        comparison=comparison,
        selected_model_name=selected_model_name,
        selected_features=selected_features,
        coef_table=coef_table,
        clv_df=clv_df,
        segment_summary=segment_summary,
        budget=budget,
        output_path=OUTPUT_DIR / "report.md",
    )

    print("Survival analysis completed.")
    print(f"Selected model: {selected_model_name}")
    print(f"Significant features retained: {selected_features if selected_features else 'None'}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {IMG_DIR}")


if __name__ == "__main__":
    main()