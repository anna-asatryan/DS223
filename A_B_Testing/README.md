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

The current setup works well for demonstrating the core ideas behind bandit algorithms, but it is still quite simplified compared to real-world A/B testing systems. There are a few extensions that would make the experiment both more robust and more realistic.


First, it would be useful to include UCB1 as an additional baseline. Right now, both Epsilon-Greedy and Thompson Sampling rely on choices that introduce some subjectivity, such as the decay rate of epsilon or the prior assumptions in Thompson Sampling. UCB1 avoids this entirely by using a deterministic rule that balances exploration and exploitation through a confidence bound. Because it does not require tuning or prior assumptions and still achieves strong theoretical performance, it serves as a clean benchmark. Including it would make the comparison more balanced and less dependent on parameter choices.


Second, the current model assumes that reward distributions stay constant over time. In practice, this is rarely true. User behavior changes, ads lose effectiveness, and external factors shift outcomes. To account for this, the model could be extended using either a sliding window or a discounted average, where recent observations are weighted more heavily than older ones. This allows the algorithm to adapt to changes instead of relying on outdated information. Without this adjustment, even well-performing algorithms can degrade over time in non-stationary environments.


Another important extension is to move toward contextual bandits. At the moment, the model treats all users as identical, which is a strong limitation. In real advertising systems, decisions depend heavily on user-specific information such as device type, time of day, or demographics. By incorporating context into the decision process, for example through a method like LinUCB, the system can learn not just which ad is best overall, but which ad is best for a particular type of user. This shifts the problem from general optimization to personalized decision-making, which is much closer to how modern systems operate.


Finally, the current implementation assumes immediate feedback after every action, which is also unrealistic. In practice, data is often collected and processed in batches, and conversions can be delayed. Simulating batched updates, where the model is only updated after a fixed number of trials, would introduce this constraint. It would also highlight an important trade-off: delayed updates slow down learning and can increase regret. Understanding this effect is important for evaluating how these algorithms perform in deployment settings.


Overall, these improvements extend the experiment from a clean theoretical setup to something that better reflects real-world conditions. Instead of only asking which algorithm performs best under ideal assumptions, the focus shifts to how different approaches behave under uncertainty, changing environments, and practical limitations.
---

## Course

**DS223 – Marketing Analytics**
Bayesian A/B Testing Assignment