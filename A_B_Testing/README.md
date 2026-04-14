# A/B Testing — Multi-Armed Bandits

## Author

Anna Asatryan

---

## Project Overview

This project implements two multi-armed bandit algorithms for an A/B testing scenario with four advertisement options.

The bandits have true mean rewards `[1, 2, 3, 4]` with Gaussian noise (known precision `τ = 1`). Each algorithm runs for **20,000 trials**.

Two strategies are compared:

* **Epsilon-Greedy** — explores with probability `ε = 1/t` (decaying from `ε₀ = 1.0`); exploits the best-estimated arm otherwise.
* **Thompson Sampling** — Bayesian approach using the Gaussian-Gaussian conjugate model with known precision. Posterior update follows the derivation from the course slides:
  * `λ = τ·N + λ₀`
  * `m = (τ·Σxᵢ + λ₀·m₀) / λ`

---

## Repository Structure

```
DS223/
│
├── Bass_Model/
│
└── A_B_Testing/
    │
    ├── Bandit.py
    │   Main script containing all classes and the experiment.
    │
    ├── requirements.txt
    │   Python dependencies required to run the project.
    │
    ├── img/
    │   Generated figures and plots.
    │   ├── epsilongreedy_learning.png
    │   ├── thompsonsampling_learning.png
    │   └── comparison.png
    │
    ├── bandit_rewards.csv
    │   Per-trial log (Bandit, Reward, Algorithm).
    │
    └── README.md
```

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
pip install -r A_B_Testing/requirements.txt
```

---

## Running the Project

```bash
python A_B_Testing/Bandit.py
```

The script runs both algorithms and produces:

* learning-process plots (linear and log scale) for each algorithm
* a side-by-side comparison of cumulative reward and cumulative regret
* a CSV file with per-trial records
* cumulative reward and regret summaries printed via loguru

---

## Output

Generated figures are stored in:

```
A_B_Testing/img/
```

The per-trial reward log is saved as:

```
A_B_Testing/bandit_rewards.csv
```

---

## Bonus: Better Implementation Plan

1. **UCB1 (Upper Confidence Bound)** — a deterministic, parameter-free alternative that achieves logarithmic regret without tuning ε or a prior. Adding UCB1 as a third baseline would strengthen the comparison.

2. **Sliding-window or discounted estimators** — in real ad campaigns, reward distributions shift over time (non-stationary). Replacing the global mean with an exponentially-weighted moving average makes both algorithms robust to drift.

3. **Contextual bandits (LinUCB)** — incorporating user context features (demographics, device, time-of-day) via a contextual bandit turns the experiment into a personalised ad-serving system, closer to production A/B testing at scale.

4. **Batched / delayed feedback** — real systems do not update after every impression. Simulating batch updates (e.g., every 100 trials) and measuring the regret penalty would surface a practical deployment concern that pure sequential simulation misses.

---

## Course

**DS223 – Marketing Analytics**
Bayesian A/B Testing Assignment