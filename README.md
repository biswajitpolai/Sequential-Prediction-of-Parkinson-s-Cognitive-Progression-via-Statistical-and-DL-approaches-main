# Parkinson’s Cognitive Decline Prediction

This project uses machine learning and deep learning to predict the future cognitive state (Normal, MCI, Dementia) of Parkinson’s Disease patients 1, 2, and 3 years ahead. We compare three approaches: a Markov‑style logistic regression, an LSTM recurrent neural network, and a state‑of‑the‑art Temporal Fusion Transformer (TFT). The models are evaluated using the Inverse Probability Weighted F1‑score (IPW‑F1) to handle severe class imbalance.

---

## Table of Contents
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Dataset
We use data from the **Parkinson’s Progression Markers Initiative (PPMI)**. The main file is `final_imputed_data.csv` (9,543 visits, 1,260 patients). The target variable is `cogstate` (1 = Normal, 2 = MCI, 3 = Dementia). Input features include:

- `age_at_visit`
- `EDUCYRS`, `BMI`, `moca`, `MSEADLG`
- `td_pigd`, `NP1COG`, `duration_yrs`, `diabetes_flag`
- current `cogstate`

The dataset is heavily imbalanced: **84% Normal**, **13.5% MCI**, and **1.5% Dementia**. Therefore we apply **Inverse Probability Weighting (IPW)** and evaluate using IPW‑F1.

---

## Models

### 1. Markov / Logistic Regression (Baseline)
- Treats each visit as an independent transition.
- Multinomial logistic regression.
- Serves as a simple, interpretable baseline.

### 2. LSTM (Recurrent Neural Network)
- Architecture: Single LSTM layer (16 units) with masking for variable‑length sequences.
- Uses **weighted categorical cross‑entropy** loss with IPW weights.
- Trained with 5‑fold cross‑validation (patient‑grouped folds).
- Predicts the next 3 visits from sequences of 1‑3 past visits.

### 3. Temporal Fusion Transformer (TFT)
- Implemented using `pytorch_forecasting`.
- Handles both static (e.g., `SEX`, `COHORT`, `subgroup`) and time‑varying features.
- Uses multi‑head attention for interpretability.
- Multi‑task: predicts both `cogstate` (classification) and `moca` (regression) using a composite loss.
- Trained on a single train‑validation split (80/20 by patient ID).

---

## Evaluation Metrics
Because accuracy is misleading due to class imbalance, we use:

- **IPW‑F1 Score** – weighted average of per‑class F1, with weights inversely proportional to class frequencies:

| Class | IPW Weight |
|-------|------------|
| Normal (1) | 0.022 |
| MCI (2)    | 0.101 |
| Dementia (3)| 0.877 |

- **Per‑class F1** – to diagnose which classes are hardest to predict.
- **Confusion / Transition Matrices** – to visualise prediction patterns.

---

## Results

### IPW‑F1 Comparison (Higher is better)

| Model | IPW‑F1 | Validation Method |
|-------|--------|-------------------|
| **Logistic (Markov)** | 0.3177 | 5‑fold Grouped CV |
| **LSTM** | 0.2784 | 5‑fold CV |
| **TFT** | **0.3413** | Validation split |

**TFT achieves the best IPW‑F1, outperforming both baselines.**

### Per‑Class F1 (TFT vs LSTM)

| Class | LSTM F1 | TFT F1 |
|-------|---------|--------|
| Normal | 0.8734 | **0.9132** |
| MCI    | 0.4189 | 0.3977 |
| Dementia | 0.2587 | **0.3256** |

TFT is superior on the most critical classes (Normal and Dementia), though MCI remains challenging for both.

### Transition Matrix (LSTM – all folds)
