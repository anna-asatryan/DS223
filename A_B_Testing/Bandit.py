"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
BANDIT_REWARDS = [1, 2, 3, 4]
NUM_TRIALS = 20000
RANDOM_SEED = 42
IMG_DIR = os.path.join(os.path.dirname(__file__), "img")


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


#--------------------------------------#


class Visualization():
    """Plotting utilities for bandit experiments."""
 
    @staticmethod
    def plot1(rewards, num_trials, bandit_rewards, algorithm_name):
        """Visualise the learning process: cumulative average reward.
 
        Two sub-plots are produced: one with a linear x-axis and one
        with a logarithmic x-axis so the early exploration is visible.
 
        Parameters
        ----------
        rewards : list[float]
            Per-trial rewards collected during the experiment.
        num_trials : int
            Total number of trials.
        bandit_rewards : list[float]
            True mean rewards (used for reference lines).
        algorithm_name : str
            Label used in title and filename.
        """
        cumulative_avg = np.cumsum(rewards) / (np.arange(1, num_trials + 1))
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
        # Linear scale
        axes[0].plot(cumulative_avg, label="Cumulative Average")
        for m in bandit_rewards:
            axes[0].axhline(y=m, linestyle="--", alpha=0.4,
                            label=f"Bandit mean = {m}")
        axes[0].set_title(f"{algorithm_name} — Learning Process (linear)")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Cumulative Average Reward")
        axes[0].legend()
 
        # Log scale
        axes[1].plot(cumulative_avg, label="Cumulative Average")
        for m in bandit_rewards:
            axes[1].axhline(y=m, linestyle="--", alpha=0.4,
                            label=f"Bandit mean = {m}")
        axes[1].set_xscale("log")
        axes[1].set_title(f"{algorithm_name} — Learning Process (log scale)")
        axes[1].set_xlabel("Trial (log)")
        axes[1].set_ylabel("Cumulative Average Reward")
        axes[1].legend()
 
        plt.tight_layout()
        fname = os.path.join(
            IMG_DIR,
            f"{algorithm_name.replace(' ', '_').lower()}_learning.png",
        )
        plt.savefig(fname, dpi=150)
        plt.close()
        logger.info(f"Saved learning-process plot -> {fname}")
 
    @staticmethod
    def plot2(eg_rewards, ts_rewards, eg_regrets, ts_regrets):
        """Compare cumulative rewards and cumulative regrets.
 
        Parameters
        ----------
        eg_rewards : list[float]
            Per-trial rewards from Epsilon-Greedy.
        ts_rewards : list[float]
            Per-trial rewards from Thompson Sampling.
        eg_regrets : list[float]
            Per-trial regret (mu* - mu_chosen) from Epsilon-Greedy.
        ts_regrets : list[float]
            Per-trial regret (mu* - mu_chosen) from Thompson Sampling.
        """
        eg_cum_reward = np.cumsum(eg_rewards)
        ts_cum_reward = np.cumsum(ts_rewards)
        eg_cum_regret = np.cumsum(eg_regrets)
        ts_cum_regret = np.cumsum(ts_regrets)
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
        # Cumulative reward
        axes[0].plot(eg_cum_reward, label="Epsilon-Greedy")
        axes[0].plot(ts_cum_reward, label="Thompson Sampling")
        axes[0].set_title("Cumulative Reward Comparison")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].legend()
 
        # Cumulative regret
        axes[1].plot(eg_cum_regret, label="Epsilon-Greedy")
        axes[1].plot(ts_cum_regret, label="Thompson Sampling")
        axes[1].set_title("Cumulative Regret Comparison")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Cumulative Regret")
        axes[1].legend()
 
        plt.tight_layout()
        fname = os.path.join(IMG_DIR, "comparison.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        logger.info(f"Saved comparison plot -> {fname}")
 
 
# ---------------------------------------------------------------------------
# Epsilon-Greedy
# ---------------------------------------------------------------------------
 
class EpsilonGreedy(Bandit):
    """Epsilon-greedy bandit algorithm with 1/t epsilon decay.
 
    Parameters
    ----------
    p : list[float]
        True mean rewards for each arm.
    initial_epsilon : float, optional
        Starting epsilon value (default 1.0).
    """
 
    def __init__(self, p, initial_epsilon=1.0):
        """Initialise the algorithm with one arm per element in ``p``.
 
        Parameters
        ----------
        p : list[float]
            True mean rewards for each arm.
        initial_epsilon : float, optional
            Starting epsilon value (default 1.0).
        """
        super().__init__(p)
        self.p = p                              # true means
        self.initial_epsilon = initial_epsilon
        self.n_bandits = len(p)
 
        # Per-arm tracking
        self.p_estimate = [0.0] * self.n_bandits  # running mean estimate
        self.N_pulls = [0] * self.n_bandits        # times each arm pulled
 
        # Experiment results (populated after experiment())
        self.rewards = []
        self.arms_chosen = []
        self.regrets = []  # per-trial regret: mu* - mu_chosen
 
    def __repr__(self):
        """Return a summary string.
 
        Returns
        -------
        str
        """
        lines = [f"EpsilonGreedy(epsilon_0={self.initial_epsilon})"]
        for i in range(self.n_bandits):
            lines.append(
                f"  Arm {i}: true_mean={self.p[i]}, "
                f"estimate={self.p_estimate[i]:.4f}, "
                f"pulls={self.N_pulls[i]}"
            )
        return "\n".join(lines)
 
    def pull(self, arm):
        """Draw a reward from arm's distribution N(p[arm], 1).
 
        Parameters
        ----------
        arm : int
            Index of the arm to pull.
 
        Returns
        -------
        float
            Observed reward.
        """
        return np.random.randn() + self.p[arm]
 
    def update(self, arm, reward):
        """Update the running mean estimate for the given arm.
 
        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Observed reward from that arm.
        """
        self.N_pulls[arm] += 1
        n = self.N_pulls[arm]
        self.p_estimate[arm] = ((n - 1) * self.p_estimate[arm] + reward) / n
 
    def experiment(self):
        """Run the full epsilon-greedy experiment.
 
        Epsilon decays as ``initial_epsilon / t`` at each trial ``t``.
        """
        self.rewards = []
        self.arms_chosen = []
        self.regrets = []
        best_mean = max(self.p)
 
        for t in range(1, NUM_TRIALS + 1):
            epsilon = self.initial_epsilon / t
 
            if np.random.random() < epsilon:
                chosen = np.random.randint(self.n_bandits)
            else:
                chosen = int(np.argmax(self.p_estimate))
 
            reward = self.pull(chosen)
            self.update(chosen, reward)
            self.rewards.append(reward)
            self.arms_chosen.append(chosen)
            self.regrets.append(best_mean - self.p[chosen])
 
        logger.info("Epsilon-Greedy experiment complete.")
 
    def report(self):
        """Print summary, generate plot, and return CSV rows.
 
        Returns
        -------
        list[dict]
            Row-dicts with keys Bandit, Reward, Algorithm.
        """
        cumulative_reward = sum(self.rewards)
        cumulative_regret = sum(self.regrets)
 
        logger.info(f"[EpsilonGreedy] Cumulative Reward: {cumulative_reward:.2f}")
        logger.info(f"[EpsilonGreedy] Cumulative Regret: {cumulative_regret:.2f}")
        logger.info(f"\n{self}")
 
        Visualization.plot1(
            self.rewards, NUM_TRIALS, self.p, "EpsilonGreedy"
        )
 
        rows = [
            {"Bandit": arm, "Reward": r, "Algorithm": "EpsilonGreedy"}
            for arm, r in zip(self.arms_chosen, self.rewards)
        ]
        return rows
 
 
# ---------------------------------------------------------------------------
# Thompson Sampling  (known precision, Gaussian-Gaussian conjugate)
# ---------------------------------------------------------------------------
 
class ThompsonSampling(Bandit):
    """Thompson Sampling with Gaussian reward and known precision.
 
    Uses the conjugate normal-normal model. For each arm the posterior
    on the mean is N(m, 1/lambda) where the update rules (from slides)
    are::
 
        lambda = tau * N + lambda_0
        m      = (tau * sum_x + lambda_0 * m_0) / lambda
 
    Parameters
    ----------
    p : list[float]
        True mean rewards for each arm.
    precision : float, optional
        Known precision tau = 1/sigma^2 (default 1.0).
    """
 
    def __init__(self, p, precision=1.0):
        """Initialise the algorithm.
 
        Parameters
        ----------
        p : list[float]
            True mean rewards for each arm.
        precision : float, optional
            Known precision of the reward distribution (default 1.0).
        """
        super().__init__(p)
        self.p = p
        self.n_bandits = len(p)
        self.tau = precision  # known precision of the likelihood
 
        # Per-arm posterior hyper-parameters: prior is N(m_0=0, 1/lambda_0=1)
        self.m = [0.0] * self.n_bandits         # posterior mean
        self.lambda_ = [1.0] * self.n_bandits    # posterior precision
        self.sum_x = [0.0] * self.n_bandits      # running sum of observations
        self.N_pulls = [0] * self.n_bandits
 
        # Experiment results
        self.rewards = []
        self.arms_chosen = []
        self.regrets = []  # per-trial regret: mu* - mu_chosen
 
    def __repr__(self):
        """Return a summary string.
 
        Returns
        -------
        str
        """
        lines = [f"ThompsonSampling(tau={self.tau})"]
        for i in range(self.n_bandits):
            lines.append(
                f"  Arm {i}: true_mean={self.p[i]}, "
                f"m={self.m[i]:.4f}, "
                f"lambda={self.lambda_[i]:.4f}, "
                f"pulls={self.N_pulls[i]}"
            )
        return "\n".join(lines)
 
    def pull(self, arm):
        """Draw a reward from arm's distribution N(p[arm], 1/tau).
 
        Parameters
        ----------
        arm : int
            Index of the arm to pull.
 
        Returns
        -------
        float
            Observed reward.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.p[arm]
 
    def update(self, arm, reward):
        """Bayesian posterior update for the given arm.
 
        Follows the update rule from the lecture slides:
            lambda = tau * N + lambda_0
            m      = (tau * sum_x + lambda_0 * m_0) / lambda
 
        Since m_0 = 0 and lambda_0 = 1 the prior term vanishes.
 
        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        reward : float
            Observed reward.
        """
        self.N_pulls[arm] += 1
        self.sum_x[arm] += reward
        self.lambda_[arm] = self.tau * self.N_pulls[arm] + 1  # tau*N + lambda_0
        self.m[arm] = (self.tau * self.sum_x[arm]) / self.lambda_[arm]
 
    def sample(self, arm):
        """Draw a sample from the posterior on arm's mean.
 
        Parameters
        ----------
        arm : int
            Index of the arm.
 
        Returns
        -------
        float
            A sample from N(m, 1/lambda).
        """
        return np.random.randn() / np.sqrt(self.lambda_[arm]) + self.m[arm]
 
    def experiment(self):
        """Run the full Thompson Sampling experiment."""
        self.rewards = []
        self.arms_chosen = []
        self.regrets = []
        best_mean = max(self.p)
 
        for _ in range(NUM_TRIALS):
            # sample from each arm's posterior, pick the best
            samples = [self.sample(i) for i in range(self.n_bandits)]
            chosen = int(np.argmax(samples))
 
            reward = self.pull(chosen)
            self.update(chosen, reward)
            self.rewards.append(reward)
            self.arms_chosen.append(chosen)
            self.regrets.append(best_mean - self.p[chosen])
 
        logger.info("Thompson Sampling experiment complete.")
 
    def report(self):
        """Print summary, generate plot, and return CSV rows.
 
        Returns
        -------
        list[dict]
            Row-dicts with keys Bandit, Reward, Algorithm.
        """
        cumulative_reward = sum(self.rewards)
        cumulative_regret = sum(self.regrets)
 
        logger.info(f"[ThompsonSampling] Cumulative Reward: {cumulative_reward:.2f}")
        logger.info(f"[ThompsonSampling] Cumulative Regret: {cumulative_regret:.2f}")
        logger.info(f"\n{self}")
 
        Visualization.plot1(
            self.rewards, NUM_TRIALS, self.p, "ThompsonSampling"
        )
 
        rows = [
            {"Bandit": arm, "Reward": r, "Algorithm": "ThompsonSampling"}
            for arm, r in zip(self.arms_chosen, self.rewards)
        ]
        return rows
 
 
# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
 
def comparison(eg, ts):
    """Visually compare Epsilon-Greedy and Thompson Sampling.
 
    Generates cumulative reward and cumulative regret plots
    side by side and logs a short summary.
 
    Parameters
    ----------
    eg : EpsilonGreedy
        Completed Epsilon-Greedy experiment.
    ts : ThompsonSampling
        Completed Thompson Sampling experiment.
    """
    Visualization.plot2(eg.rewards, ts.rewards, eg.regrets, ts.regrets)
 
    eg_total_reward = sum(eg.rewards)
    ts_total_reward = sum(ts.rewards)
    eg_total_regret = sum(eg.regrets)
    ts_total_regret = sum(ts.regrets)
 
    logger.info("=" * 55)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 55)
    logger.info(
        f"  Epsilon-Greedy  | Reward: {eg_total_reward:>10.2f}  "
        f"| Regret: {eg_total_regret:>10.2f}"
    )
    logger.info(
        f"  Thompson Sampl. | Reward: {ts_total_reward:>10.2f}  "
        f"| Regret: {ts_total_regret:>10.2f}"
    )
    logger.info("=" * 55)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
 
    np.random.seed(RANDOM_SEED)
 
    logger.info("Starting A/B Testing Bandit Experiment")
    logger.info(f"Bandit true means: {BANDIT_REWARDS}")
    logger.info(f"Number of trials : {NUM_TRIALS}")
 
    # Ensure img/ directory exists
    os.makedirs(IMG_DIR, exist_ok=True)
 
    # ---- Epsilon-Greedy ----
    logger.info("-" * 40)
    logger.info("Running Epsilon-Greedy (epsilon_0=1.0, decay=1/t)")
 
    eg = EpsilonGreedy(BANDIT_REWARDS, initial_epsilon=1.0)
    eg.experiment()
    eg_rows = eg.report()
 
    # ---- Thompson Sampling ----
    logger.info("-" * 40)
    logger.info("Running Thompson Sampling (precision=1.0)")
 
    ts = ThompsonSampling(BANDIT_REWARDS, precision=1.0)
    ts.experiment()
    ts_rows = ts.report()
 
    # ---- Save combined CSV ----
    csv_path = os.path.join(os.path.dirname(__file__), "bandit_rewards.csv")
    all_rows = eg_rows + ts_rows
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(all_rows)} records -> {csv_path}")
 
    # ---- Visual comparison ----
    comparison(eg, ts)
 
    logger.info("Experiment complete.")
 