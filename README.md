# MLB Trade Deadline Impact Model: An Interrupted Time Series Framework
**Developed by:** [Your Name/GitHub Handle] & [Jack Behnfeldt](https://github.com/JackBehnfeldt)  
**Date:** April 2026  

---

## 📌 Executive Summary
[cite_start]This repository contains a machine learning Interrupted Time Series (ITS) framework designed to quantify the true "excess returns" of the 2025 MLB Trade Deadline[cite: 3]. [cite_start]By isolating team profiles at the July 31 trade boundary [cite: 7, 50][cite_start], the model constructs a historical counterfactual—predicting how a team would have performed over the second half of the season had they stood pat[cite: 23, 234]. 

[cite_start]Instead of relying on naive historical averages or basic run-differential regressions [cite: 31, 41][cite_start], this framework uses optimized **CatBoost** models to capture non-linear matchup features [cite: 4, 41, 83][cite_start], giving front offices an empirical look at the marginal value of trade deadline acquisitions[cite: 33, 58].

---

## 📊 Technical Architecture & Data Engineering
[cite_start]To prevent target leakage, all team performance vectors are frozen on July 31[cite: 50]. [cite_start]The pipeline engineers **135 features** per game [cite: 47, 62][cite_start], focusing on team momentum, consistency, and roster depth[cite: 48]:

* [cite_start]**WLS Trend Slopes:** Weighted Least Squares curves that weight recent games more heavily, tracking a team's true velocity entering the deadline[cite: 71, 72].
* [cite_start]**Split Pitching Metrics:** Explicitly separates Starter ERA from Bullpen ERA and measures pitching staff volume distribution (Inning Share)[cite: 67, 68].
* [cite_start]**Scoring Volatility:** Utilizes the Coefficient of Variation (`cv_run_diff`) to evaluate a team's scoring consistency independently of their raw run average[cite: 74].
* [cite_start]**Opponent Mirroring:** Automatically mirrors the complete feature matrix for the projected opponent to account for strength of schedule in the second half[cite: 49, 78].

---

## 📉 Statistical Performance & Validation (2024 Test Set)
[cite_start]Before scoring the 2025 season [cite: 234][cite_start], the pipeline's predictive accuracy was rigorously validated against a baseline expectation on a completely held-out **2024 test set ($n = 1,596$ games)**[cite: 131].

### Model Performance vs. Pythagorean Baseline:
* [cite_start]**Runs Scored Model:** Outperformed the baseline in **53.7%** of games (Paired t-test: $t = -2.10, p = 0.018$)[cite: 145].
* [cite_start]**Runs Allowed Model:** Highly significant improvement in runs allowed prediction (Paired t-test: $t = -2.93, p = 0.0017$)[cite: 146].
* [cite_start]**Win/Loss Classifier:** Achieved **57.96% accuracy** (compared to the baseline's 54.70%) [cite: 138, 141][cite_start], yielding a highly significant **McNemar's test** result ($\chi^2 = 7.31, p = 0.0069$)[cite: 147].

<p align="center">
  <img src="visuals/confusion_matrix.png" width="45%" alt="Confusion Matrix" />
  <img src="visuals/game_scatter.jpg" width="45%" alt="Game Error Scatter" />
</p>
[cite_start]<p align="center"><em>Figure 1: Side-by-side performance breakdown and per-game absolute error reduction on the 2024 held-out test set[cite: 131].</em></p>

---

## 🚀 2025 Post-Mortem & Micro Findings
[cite_start]Applying the validated models to the second half of the 2025 season reveals who truly unlocked "excess returns" at the deadline and who fell victim to standard regression[cite: 56, 59].

<p align="center">
  <img src="visuals/2025_deadline_impact_summary.jpg" width="95%" alt="2025 Deadline Impact Summary" />
</p>
[cite_start]<p align="center"><em>Figure 2: League-wide 2025 Trade Deadline Excess Returns across Wins, Runs Scored, and Runs Allowed[cite: 173].</em></p>

### ⚾ Case Study 1: New York Mets (Actual: 83 Wins | Counterfactual Projection: 91 Wins | Impact: -8 Wins)
[cite_start]The Mets took an aggressive buying approach, bringing in high-leverage assets[cite: 186].
* [cite_start]**The Story:** The bats responded beautifully, surging entirely out of their historical peer interval to score an incredible **+0.89 runs/game above projection**[cite: 187, 238]. 
* [cite_start]**The Downfall:** This historic offensive explosion was entirely neutralized by an unexpected pitching collapse, with the staff surrendering **+0.89 runs/game more than projected**, ultimately costing the Mets 8 wins relative to their pre-deadline trajectory[cite: 188, 238].

<p align="center">
  <img src="visuals/team_deep_dives/NYM_2025_its_panel_FINAL.jpg" width="95%" alt="Mets 2025 ITS Panel" />
</p>

### 🐯 Case Study 2: Detroit Tigers (Actual: 87 Wins | Counterfactual Projection: 93.5 Wins | Impact: -7 Wins)
[cite_start]Detroit attempted to solidify their playoff presence by adding veteran arms[cite: 192, 193]. 
* [cite_start]**The Story:** Despite the deadline additions, the counterfactual model shows the Tigers regressed sharply relative to their blistering first-half pace[cite: 193]. [cite_start]The offense went completely cold, dropping **-0.62 runs/game below projection** [cite: 193][cite_start], resulting in an overall negative impact against their expected baseline[cite: 192].

<p align="center">
  <img src="visuals/team_deep_dives/DET_2025_its_panel_FINAL.jpg" width="95%" alt="Detroit 2025 ITS Panel" />
</p>

---

## 🛠️ Reproduction & Dependencies

> ⚠️ **Data Dependency Notice:** To respect data provider Terms of Service, this repository does **not** host raw StatHead database tables. A personal subscription to StatHead is required to pull the game logs and populate the `/data` folder schemas before executing the scripts.

### R Package Prerequisites
[cite_start]The `catboost` package must be compiled directly from GitHub as it does not live on CRAN[cite: 37]:
```R
# 1. Install developer tools
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")

# 2. Compile CatBoost R-Package
devtools::install_github("catboost/catboost", subdir = "catboost/R-package")

# 3. Required pipeline libraries
library(dplyr)
library(catboost)
library(ggplot2)
library(patchwork)