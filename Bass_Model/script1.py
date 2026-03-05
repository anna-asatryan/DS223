"""
Script 1: Estimate Bass diffusion parameters using Apple Watch sales data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from helper_functions import bass_model, peak_adoption_time


def main():

    # Load dataset
    data = pd.read_csv("data/apple_watch_sales.csv")

    # Time index
    t = np.arange(1, len(data) + 1)

    # Sales values
    sales = data["shipments"].values

    # Estimate parameters
    initial_guess = [0.03, 0.4, 500]

    params, _ = curve_fit(bass_model, t, sales, p0=initial_guess)

    p_est, q_est, M_est = params

    print("Estimated Bass parameters:")
    print(f"p (innovation): {p_est:.4f}")
    print(f"q (imitation): {q_est:.4f}")
    print(f"M (market potential): {M_est:.2f} million")

    # Peak adoption time
    t_peak = peak_adoption_time(p_est, q_est)

    launch_year = data["year"].min()
    peak_year = launch_year + t_peak

    print("\nPeak adoption:")
    print(f"Years after launch: {t_peak:.2f}")
    print(f"Estimated year: {peak_year:.2f}")

    # Predicted sales
    predicted_sales = bass_model(t, p_est, q_est, M_est)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(data["year"], sales, marker="o", label="Actual Sales")
    plt.plot(data["year"], predicted_sales, marker="s", label="Bass Model Fit")

    plt.title("Bass Model Fit vs Apple Watch Sales")
    plt.xlabel("Year")
    plt.ylabel("Units Sold (millions)")
    plt.legend()
    plt.grid(True)

    plt.savefig("Bass_Model/img/bass_fit.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()