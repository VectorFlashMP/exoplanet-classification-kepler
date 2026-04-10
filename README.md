# Exoplanet Detection using Kepler Data

This project explores how machine learning can be used to detect exoplanets 
from Kepler light curve data.

## Overview
Stars exhibit small dips in brightness when a planet passes in front of them. 
These signals are subtle and often buried in noise.

This project focuses on:
- Visualizing light curve data
- Preprocessing noisy and imbalanced datasets
- Training models to classify potential exoplanet signals

## Methods
- Data visualization (flux vs time)
- Preprocessing: normalization, smoothing
- Handling class imbalance using SMOTE
- Models tested:
  - Multi-layer Perceptron (Neural Network)
  - Random Forest
  - Logistic Regression

## Current Status
This project is exploratory and focuses on understanding data challenges 
and comparing model approaches rather than final optimization.

## Key Insight
Detecting exoplanets is not just a modeling problem—it is a data problem.

## Future Work
- Improve feature extraction
- Optimize neural network architecture
- Apply deep learning (CNN on time-series)
