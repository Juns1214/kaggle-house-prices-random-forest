# Kaggle : House Prices - Advanced Regression Techniques

Predicting the sale prices of houses in Ames, Iowa, using an optimised **Random Forest Regression** model.

This project was completed in two parts:

1. **Assignment** - Individual regression assignment for *MAI5043 Applied Machine Learning* (Master in Artificial Intelligence, Sunway University, September 2026 semester).
2. **Kaggle submission** - The same approach applied to the official [Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), predicting prices for the competition's `test.csv`.

**Kaggle public score: 0.13939** (RMSE between the log of predicted and actual prices; lower is better)

---

## Assignment vs Kaggle submission

Both parts use the same preprocessing and the same optimised Random Forest. The difference is in what data is used and how the model is judged.

| | Assignment | Kaggle submission |
|---|---|---|
| **Goal** | Build, evaluate and optimise a regression model, and explain it in a report | Predict prices for houses whose real prices are hidden |
| **Training data** | 80% of `train.csv` (1,166 houses) | All of `train.csv` (1,458 houses after removing 2 outliers) |
| **Test data** | The other 20% of `train.csv` (292 houses), because `test.csv` has no prices | `test.csv` (1,459 houses), scored by Kaggle |
| **Evaluation** | MAE, MSE, RMSE and R² in dollars, calculated locally | RMSE on log prices, calculated by Kaggle |
| **Extra preprocessing** | - | Fills 15 columns that only have missing values in `test.csv` (e.g. `KitchenQual`, `GarageCars`, `MSZoning`) |
| **Output** | Report, figures and evaluation results | `submission/submission.csv` uploaded to Kaggle |
| **Script** | `src/assignment.py` | `src/kaggle_submission.py` |

---

## Approach

**1. Exploratory Data Analysis**
- Checked the distribution of `SalePrice` (right-skewed, skewness 1.88), which was reduced to 0.12 with a log transformation.
- Used scatter plots, histograms, box plots, count plots, a correlation heatmap and a missing-value chart to study the features.
- `OverallQual` (r = 0.79) and `GrLivArea` (r = 0.71) showed the strongest correlation with price.

**2. Data preprocessing**
- Checked for duplicates (none found).
- Reviewed houses flagged by the IQR rule and Z-score, and removed two partial sales with very large living areas but very low prices (Id 1299 and 524).
- Handled missing values based on their meaning: `'None'` or 0 when a feature is absent, neighbourhood median for `LotFrontage`, and the training-set mode otherwise. Dropped `PoolQC`, `MiscFeature`, `Alley`, `Fence` and `GarageYrBlt`.
- Encoded categorical features using ordinal encoding (14 quality-type features), binary encoding (`CentralAir`, `Street`) and one-hot encoding (nominal features).
- Applied `log1p` to `SalePrice` and 16 skewed features, then `StandardScaler` to non-binary features.
- All imputation, encoding and scaling were fitted on the training data only to avoid data leakage.

**3. Model**
- Baseline: `RandomForestRegressor` with default settings.

**4. Optimisation**
- A `Pipeline` combining `SelectFromModel` (keeps the top-N features by importance) and a Random Forest.
- `RandomizedSearchCV` with 40 combinations and 5-fold cross-validation.
- Best settings: 150 features, 500 trees, `max_features = 0.25`, `max_depth = 20`, `min_samples_split = 5`, `min_samples_leaf = 1`, `bootstrap = False`.

---

## Results

**Assignment (20% held-out test set)**

| Model | MAE | RMSE | R² | Features |
|---|---|---|---|---|
| Original Random Forest | $16,288 | $23,704 | 0.8983 | 246 |
| Optimised Random Forest | $15,379 | $22,061 | 0.9119 | 150 |

**Kaggle**

| | Score (RMSE on log prices) |
|---|---|
| 5-fold cross-validation on `train.csv` | 0.1285 |
| Kaggle public leaderboard | **0.13939** |

> Random Forest results can vary slightly between scikit-learn versions, so re-running the scripts may give marginally different numbers.

---

## Repository structure

```
├── data/
│   ├── train.csv                 # Kaggle training data (with SalePrice)
│   ├── test.csv                  # Kaggle test data (without SalePrice)
│   ├── sample_submission.csv     # Kaggle submission format
│   └── data_description.txt      # Description of all 79 features
├── src/
│   ├── assignment.py             # Assignment: EDA, preprocessing, training, evaluation, optimisation
│   └── kaggle_submission.py      # Kaggle: trains on full train.csv and predicts test.csv
├── figures/                      # All plots saved by assignment.py
├── report/
│   └── MAI5043_22025530_Assignment1_Report.docx   # Assignment report
├── submission/
│   └── submission.csv            # Predictions submitted to Kaggle
├── requirements.txt
└── README.md
```

---

## How to run

```bash
pip install -r requirements.txt

# Assignment (saves all plots to figures/)
python src/assignment.py

# Kaggle submission (creates submission/submission.csv)
python src/kaggle_submission.py
```

Run both scripts from the repository root so they can find the `data/` folder. The randomised search in `assignment.py` takes several minutes.

---

## Possible improvements

- Feature engineering, such as total floor area and house age at sale.
- Gradient boosting models (XGBoost, LightGBM), which usually score better on this competition.
- A two-stage hyperparameter search (random search followed by grid search).

---

## Data

The dataset is from the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Please refer to the competition page for its data and usage rules.

## Author

Chong Jun Wei
