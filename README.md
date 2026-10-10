# housing-price-xgboost
End-to-end housing price regression using XGBoost, leak-free scikit-learn Pipelines, and custom feature engineering (MAE: ~$15.1k).
# Ames Housing Price Prediction with XGBoost 🏠

An end-to-end Machine Learning pipeline designed to predict residential property values using the Ames Housing dataset. This project focuses on leak-free preprocessing, feature engineering, and hyperparameter tuning to systematically reduce prediction error.

---

## 📊 Model Performance & Progression

The model's performance was evaluated using 5-fold cross-validation on Mean Absolute Error (MAE):

| Model Stage | Technical Focus | Average MAE |
| :--- | :--- | :--- |
| **Baseline Model** | Initial Random Forest Regressor | ~$17,636 |
| **Pipeline Iteration** | XGBoost + `ColumnTransformer` Preprocessing | ~$15,260 |
| **Final Optimized Pipeline** | XGBoost + Feature Engineering (`TotalSF`, `HouseAge`) | **~$15,138** |

> **Total Error Reduction:** Improved prediction accuracy by **~$2,498 MAE** over the baseline model.

---

## 🛠️ Key Engineering Practices

* **Data Leakage Prevention:** Built using `scikit-learn` `Pipeline` and `ColumnTransformer` objects to ensure all missing value imputation (median for numerical) and categorical encoding (One-Hot Encoding) were computed strictly within cross-validation folds.
* **Feature Engineering:** Derived high-signal composite domain features prior to model evaluation:
  * `TotalSF`: Sum of total basement, first floor, and second floor square footage (`TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`).
  * `HouseAge`: Derived property age at the time of sale (`YrSold` - `YearBuilt`).
* **Target & ID Dropping:** Explicitly dropped non-predictive identifiers (`Id`) and target variable columns (`SalePrice`) before model training to guarantee strict data integrity.
* **Hyperparameter Tuning:** Configured gradient boosting parameters via `XGBRegressor` (`learning_rate=0.03`, `n_estimators=1000`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`) to prevent overfitting.

---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.x
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`
* **Workflow:** Jupyter Notebooks / Kaggle API
