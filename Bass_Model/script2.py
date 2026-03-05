"""
Script 2: Forecast diffusion of Meta Ray-Ban Display using Bass model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper_functions import bass_model


def main():

    # Parameters estimated from Apple Watch model
    p_est = 0.0102
    q_est = 0.4887
    M_est = 387.98

    # Market potential adjustment
    M_rayban = 0.35 * M_est

    # Forecast horizon
    years_future = np.arange(0, 20)

    # Predicted adoption
    rayban_sales = bass_model(years_future, p_est, q_est, M_rayban)

    launch_year = 2025
    calendar_years = launch_year + years_future

    forecast = pd.DataFrame({
        "Year": calendar_years,
        "Yearly Adoption (millions)": rayban_sales,
        "Cumulative Adoption (millions)": np.cumsum(rayban_sales)
    })

    print(forecast.round(2))

    # Plot yearly adoption
    plt.figure(figsize=(10, 5))
    plt.plot(calendar_years, rayban_sales, marker="o")
    plt.title("Forecasted Yearly Adoption of Meta Ray-Ban Display")
    plt.xlabel("Year")
    plt.ylabel("New Adopters (millions)")
    plt.grid(True)

    plt.savefig("Bass_Model/img/rayban_yearly_adoption.png", dpi=300)
    plt.show()

    # Plot cumulative adoption
    plt.figure(figsize=(10, 5))
    plt.plot(calendar_years, np.cumsum(rayban_sales), marker="o")
    plt.title("Forecasted Cumulative Adoption of Meta Ray-Ban Display")
    plt.xlabel("Year")
    plt.ylabel("Total Adopters (millions)")
    plt.grid(True)

    plt.savefig("Bass_Model/img/rayban_cumulative_adoption.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()