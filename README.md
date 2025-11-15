# AI_Project
AI-Based Music Aesthetic Evaluation
Project Overview

This project focuses on predicting the aesthetic qualities of music using audio embeddings and regression models. The goal is to evaluate multiple musical aspects — coherence, musicality, memorability, clarity, and naturalness — using machine learning models trained on audio features.

The project leverages Wav2Vec2 embeddings, MFCC features, dimensionality reduction, and ensemble regression models to achieve accurate predictions.

Features

Audio Feature Extraction

Wav2Vec2 embeddings from raw audio.

MFCC features (optional for further improvement).

Preprocessing

Feature scaling using StandardScaler.

Dimensionality reduction with PCA to retain most variance.

Machine Learning Models

Ridge Regression with hyperparameter tuning.

ElasticNet Regression.

XGBoost Regressor.

Ensemble of models for improved performance.

Evaluation Metrics

Mean Squared Error (MSE)

R² score

Linear Correlation Coefficient (LCC)

Spearman’s Rank Correlation Coefficient (SRCC)

Kendall’s Rank Correlation Coefficient (KRCC)

Top-Tier Accuracy (TTA)

Visualization

Scatter plots of actual vs predicted scores for each musical aspect.

PDF export of prediction plots.

Dataset

Audio dataset with corresponding human-annotated scores for the five target attributes.

Splits into training and test sets.

Features are extracted using Wav2Vec2 and optionally MFCC.
