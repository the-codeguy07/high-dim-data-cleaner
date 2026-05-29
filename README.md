# High-Dimensional Data Cleaning Pipeline

## Overview
This project is an automated, enterprise-level data cleaning pipeline built entirely with **Python, Pandas, and NumPy**. It is designed to handle high-dimensional datasets (50+ columns) typically found in machine learning and data engineering workflows. 

Rather than relying on pre-built machine learning libraries like `scikit-learn`, this project implements raw mathematical transformations and native Pandas operations to process the data efficiently.

## Key Features
* **Dynamic Column Separation:** Automatically distinguishes between standard demographics and high-dimensional continuous features.
* **Algorithmic Imputation:** Uses linear interpolation and median fallback to estimate missing values based on surrounding data points.
* **Native Feature Scaling:** Implements Z-Score Normalization (Standard Scaling) using pure mathematical operations `(Value - Mean) / Standard Deviation`.
* **Outlier Capping:** Dynamically detects and caps extreme multivariate outliers at the 1st and 99th percentiles.
* **Memory Optimization:** Downcasts 64-bit integers and floats to 32-bit/16-bit to dramatically reduce RAM usage during large-scale processing.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy
* **Environment:** VS Code

## How to Run
1. Ensure `pandas` and `numpy` are installed.
2. Place your raw dataset in the project directory.
3. Update the file paths at the bottom of the script.
4. Run `python code.py`.