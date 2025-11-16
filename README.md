# Automatic Song Aesthetics Evaluation
AI-Based Music Aesthetic Evaluation
Project Overview

This project focuses on predicting the aesthetic qualities of music using audio embeddings and regression models. The goal is to evaluate multiple musical aspects — coherence, musicality, memorability, clarity, and naturalness — using machine learning models trained on audio features.

The project leverages Wav2Vec2 embeddings, MFCC features, dimensionality reduction, and ensemble regression models to achieve accurate predictions.

# Links to notebooks (code history)
Assignment 1 (EDA) : 
https://colab.research.google.com/drive/14gQzUPjkXsWF_N9gNKz_hyb2WwbuOg8Y?usp=sharing

Assignment 2 (Baselines) : <br>
MFCC Regression, Wav2vec2 + ridge, audiobox aesthetics : https://colab.research.google.com/drive/1HPiM030LOjpUwMpKmX914nE2w4CfFfh_?usp=sharing <br>
MFCC + randomforestregressor : https://colab.research.google.com/drive/1gvAkLjIjye5VQsRgsKgRd1y8D9NPJ5PV?usp=sharing <br>
Wav2vec2 + randomforestregressor : https://colab.research.google.com/drive/1xi6uR4RwILHh7ycVbBo5ucPoVtZuomD9?usp=sharing

Assignment 3 (Final Improved Ensemble Model):
https://colab.research.google.com/drive/1mbXPqywbqeK7gKmYDeYiVNwzxfu4GivY?usp=sharing

# Model Performance Overview

We evaluated multiple approaches for predicting SongEval aesthetic scores. Overall, we see a clear progression: simple MFCC-based baselines struggle, pretrained models improve things significantly, and the final ensemble achieves the strongest results.

# MFCC Baselines

The MFCC linear regression baseline performs the weakest, with high MSE values (>1.0) and low R² scores (~0.18–0.20). RandomForest on MFCCs improves this slightly, but overall MFCC features still lack the high-level structure needed for aesthetic prediction.

# Wav2Vec2 Models

 Switching to Wav2Vec2 embeddings leads to a major jump in performance. Both Linear Regression and RandomForest cut MSE values nearly in half and achieve strong correlations (LCC ≈ 0.82–0.84). This shows how well pretrained audio representations capture musical and perceptual features.

# Audiobox Aesthetics Model

Fine-tuning the Audiobox model did not outperform the Wav2Vec2 baselines. Some targets show high error or even negative R², suggesting domain mismatch or insufficient training. While correlations are decent, the overall regression accuracy is lower than expected.

# Final Ensemble Model

The final ensemble produces the best results across all metrics. It consistently reaches the lowest MSE (0.37–0.43 range), the highest R² (up to ~0.70), and strong ranking correlations. This model achieves the highest weighted score (0.7215), making it the best-performing system in our pipeline.

# Features
Audio Feature Extraction
Wav2Vec2 embeddings from raw audio.
MFCC features (optional for further improvement).
Preprocessing
Feature scaling using StandardScaler.
Dimensionality reduction with PCA to retain most variance.

# Machine Learning Models

Ridge Regression with hyperparameter tuning.
ElasticNet Regression.
XGBoost Regressor.
Ensemble of models for improved performance.

# Evaluation Metrics

Mean Squared Error (MSE)
R² score
Linear Correlation Coefficient (LCC)
Spearman’s Rank Correlation Coefficient (SRCC)
Kendall’s Rank Correlation Coefficient (KRCC)
Top-Tier Accuracy (TTA)

# Visualization

Scatter plots of actual vs predicted scores for each musical aspect.
PDF export of prediction plots.

# Dataset

Features are extracted using Wav2Vec2 and optionally MFCC.
