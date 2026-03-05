# Bass Model Innovation Diffusion Analysis

## Author

Anna Asatryan

---

## Project Overview

This project analyzes the potential diffusion of the **Meta Ray-Ban Display with Neural Band**, an innovation featured in *TIME's Best Inventions of 2025*.

The analysis applies the **Bass Diffusion Model**, a widely used framework for modeling how new technologies spread through a population.

The Bass model describes adoption through two mechanisms:

* **Innovation (p)** – adoption driven by external influences such as advertising or media exposure.
* **Imitation (q)** – adoption driven by social influence and word-of-mouth.

Because the Meta Ray-Ban Display is a new product without historical adoption data, the **Apple Watch smartwatch market** is used as a historical analogue to estimate the Bass model parameters.

These parameters are then used to forecast the potential adoption path of the Meta Ray-Ban Display.

---

## Repository Structure

```
DS223/
│
├── A_B_Testing/
│
└── Bass_Model/
    │
    ├── MA_HW1.ipynb
    │   Main notebook containing the full analysis.
    │
    ├── helper_functions.py
    │   Utility functions including the Bass diffusion model.
    │
    ├── script1.py
    │   Script used to estimate Bass model parameters from Apple Watch data.
    │
    ├── script2.py
    │   Script used to forecast diffusion of the Meta Ray-Ban Display.
    │
    ├── requirements.txt
    │   Python dependencies required to run the project.
    │
    ├── data/
    │   Dataset used in the analysis.
    │
    ├── img/
    │   Generated figures and plots used in the report.
    │
    ├── report/
    │   Final project report files.
    │   ├── report_source.md
    │   └── report.pdf
    │
    └── README.md
```

---

## Data Source

Apple Watch shipment data was obtained from Statista:

https://www.statista.com/statistics/1421546/apple-watch-sales-worldwide/

Annual shipments are used as a proxy for the number of adopters per period.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DS223.git
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
pip install -r Bass_Model/requirements.txt
```

---

## Running the Project

### Run the notebook

```bash
jupyter notebook Bass_Model/MA_HW1.ipynb
```

The notebook reproduces the entire analysis:

* loading the dataset
* estimating Bass model parameters
* forecasting adoption
* visualizing results

### Run the scripts

Estimate Bass model parameters:

```bash
python Bass_Model/script1.py
```

Forecast innovation diffusion:

```bash
python Bass_Model/script2.py
```

---

## Output

The project generates:

* Apple Watch historical adoption plots
* Bass model parameter estimation
* predicted yearly adoption curves
* cumulative diffusion forecasts

Generated figures are stored in:

```
Bass_Model/img/
```

The final report is available at:

```
Bass_Model/report/report.pdf
```

---

## Course

**DS223 – Marketing Analytics**
Bass Model Innovation Diffusion Assignment