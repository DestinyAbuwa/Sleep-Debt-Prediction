# Sleep Debt & Screen Time Prediction (CS 462 Semester Project)

Predicting severe sleep debt and next-day fatigue risk from late-night smartphone usage, blue light exposure, caffeine intake, and overnight sleep metrics using supervised machine learning.

* **Dataset:** [Sleep Debt & Screen Time: Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits) (8,500 records, 18 attributes)

---

## Research Questions

1. **Screen Exposure & Brightness:** How strongly do bedtime phone duration (`bedtime_phone_minutes`) and screen brightness (`screen_brightness_pct`) drive predicted sleep debt risk according to learned feature weights and importances?
2. **Blue Light Mitigation:** Does activating a blue light filter (`blue_light_filter_active = 1`) measurably reduce predicted fatigue risk during high screen time or high brightness sessions?
3. **App Category Arousal:** Does short-form video consumption (`TikTok / Reels`) lead to higher predicted fatigue compared to lower-arousal activities (`News / Reading` or `Messaging / Chat`)?
4. **Algorithm Comparison:** Which supervised learning model (KNN, Logistic Regression, Naïve Bayes, Decision Tree, Random Forest, or Gradient Boosting) achieves the highest classification performance (ROC-AUC and F1-score) and best runtime trade-off?
5. **Pre-Sleep Habits vs. Overnight Sleep Metrics:** How accurately can models predict sleep debt severity using **only pre-sleep behavioral habits** (screen time, app choice, blue light filter, caffeine, activity) compared to including **overnight sleep-tracking metrics** (`total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `sleep_latency_min`)?

---

## Project Execution Roadmap

### Phase 1: Data Exploration & Preprocessing (`01_eda_and_preprocessing.ipynb`)
- [ ] Load `bedtime_screentime_sleep_debt.csv` and verify missing values, duplicates, and data types.
- [ ] Drop `user_id` (non-predictive identifier).
- [ ] Prevent data leakage: drop `next_day_fatigue_score` when predicting `sleep_debt_category` (and run a separate behavioral-only feature comparison).
- [ ] Define binary target (`0` = Optimal/Mild vs. `1` = Moderate/Severe Sleep Debt) and visualize class distribution.
- [ ] One-hot encode categorical features (`gender`, `occupation_type`, `chronotype`, `primary_bedtime_app`).
- [ ] Save a reproducible stratified train/test split (`random_state=42`) so all models are evaluated on identical data.

### Phase 2: Model Training & Hyperparameter Tuning
*All scaling (`StandardScaler`) and tuning (`GridSearchCV` with 10-fold Stratified CV) must be fit on training folds only to prevent data leakage.*
- [ ] **Single Classifiers (`02_baseline_models.ipynb`):**
  - [ ] K-Nearest Neighbors (tune `n_neighbors`)
  - [ ] Logistic Regression (tune `C`)
  - [ ] Gaussian Naïve Bayes
  - [ ] Decision Tree (tune `max_depth`, `criterion`)
- [ ] **Ensemble Models (`03_ensemble_models.ipynb`):**
  - [ ] Random Forest (tune `n_estimators`, `max_depth`)
  - [ ] Gradient Boosting (tune `learning_rate`, `n_estimators`, `max_depth`)
- [ ] Record Accuracy, Precision, Recall, F1-Score, ROC-AUC, and runtime (seconds) for every model.
- [ ] Export high-resolution Confusion Matrices and combined ROC curves to `figures/`.

### Phase 3: Final Report Writing (CS 462 Rubric Order)
- [ ] **Section 3:** Dataset Description & Preparation (feature dictionary, class counts, preprocessing log)
- [ ] **Section 2:** Methodology (algorithm explanations, equations with defined symbols, validation design)
- [ ] **Section 4:** Experiments & Results (comparison table, confusion matrices, ROC curves, error/trade-off analysis)
- [ ] **Section 5:** Discussion & Conclusion (best model selection, answers to research questions, limitations, future work)
- [ ] **Section 1:** Introduction (problem background, objective, model selection rationale, research questions)
- [ ] **Front/Back Matter:** Title Page, Table of Contents, Abstract (150–250 words), and References (APA/IEEE)

---

## Planned Repository Structure

```text
Sleep-Debt-Prediction/
├── data/
│   ├── raw/                  # Original bedtime_screentime_sleep_debt.csv
│   └── processed/            # Cleaned & encoded train/test datasets
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_ensemble_models.ipynb
├── figures/                  # Exported plots (class distribution, ROC curves, confusion matrices)
└── README.md
