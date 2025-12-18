# Movie Revenue Prediction - Project Details

## Problem Statement & Objectives

The movie industry faces high financial risk due to uncertainty in box office performance. Producers, investors, and marketing teams need data-driven estimates of expected revenue before and shortly after a movie's release to guide budget allocation and strategic decisions.

**Objectives**

- Predict movie revenue using available metadata and engagement-based features.
- Identify which factors (e.g., budget, votes, popularity) most strongly influence revenue.
- Build a transparent, reproducible ML pipeline that can be extended or deployed.

## Technical Approach / Methodology

- Framed the task as a supervised regression problem with revenue as the target variable.
- Performed Exploratory Data Analysis (EDA) to understand distributions, missingness, and relationships among features.
- Applied data preprocessing: missing value imputation, encoding of categorical variables, and scaling for relevant models.
- Trained and compared multiple baseline and advanced regression models (e.g., Linear Regression, Regularized models, Tree-based models).
- Evaluated performance using standard regression metrics such as R-squared and RMSE and inspected residuals for model diagnostics.

## Implementation Details

- Organized the project into clearly separated directories:
  - `data/` for raw/processed data or data loading instructions.
  - `notebooks/` for EDA, feature engineering, and modeling (6 sequential notebooks).
  - `models/` for any saved trained models.

- Structured notebooks to follow a logical sequence:
  1. **01_eda_movie_revenue.ipynb** - Initial data exploration, loading structure, basic statistics
  2. **02_eda_movie_revenue.ipynb** - Value counts, data types, and initial distribution analysis
  3. **03_eda_movie_revenue.ipynb** - Feature analysis, correlations, and outlier detection
  4. **04_eda_movie_revenue.ipynb** - Feature importance exploration and actual vs. predicted plots
  5. **05_eda_movie_revenue.ipynb** - Ridge and Lasso regression models and evaluation
  6. **06_eda_movie_revenue.ipynb** - Random Forest and advanced ensemble models

- Used Git and conventional commit practices to track incremental improvements.

If a scripts-based pipeline is added (e.g., `src/`):

- Encapsulate data prep, training, and inference into modular Python scripts for reuse.
- Enable configuration via constants or config files instead of hard-coded paths.

## Key Learnings & Challenges Solved

- Learned how correlated financial and popularity features (budget, votes, engagement metrics) drive revenue predictions and how to avoid data leakage.
- Dealt with noisy, skewed financial data and outliers by applying appropriate transformations (log, sqrt) and robust evaluation practices.
- Improved model generalization by balancing model complexity with interpretability and avoiding overfitting through cross-validation and regularization.
- Strengthened skills in communicating model assumptions, limitations, and actionable insights to business stakeholders.
- Demonstrated the importance of reproducible workflows: clear documentation, version control, and modular code for team collaboration.

## Tools & Libraries Used

- **Language**: Python 3.9+
- **Data & ML**: pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Environment**: Jupyter Notebook / Google Colab for experimentation
- **Version Control**: Git + GitHub
- **Model Persistence**: joblib/pickle for saving trained models
- **Additional**: scipy (if needed for statistical tests)

## Results & Deliverables

- A functioning regression pipeline capable of predicting movie revenue from structured input features.
- Six Jupyter notebooks documenting the full analytical process from EDA to model evaluation and comparison.
- A repository structured for professional portfolio use, with clear documentation, appropriate ignore rules, and space for extension (API, dashboard, etc.).
- Discussion-ready insights on which factors drive movie revenue and how this model could support stakeholders in planning and risk management.
- Clean commit history with conventional commit messages (docs:, feat:, refactor:) for professional presentation.

## Project Structure

```
movie-revenue-prediction/
├─ data/
├─ notebooks/
│  ├─ 01_eda_movie_revenue.ipynb
│  ├─ 02_eda_movie_revenue.ipynb
│  ├─ 03_eda_movie_revenue.ipynb
│  ├─ 04_eda_movie_revenue.ipynb
│  ├─ 05_eda_movie_revenue.ipynb
│  └─ 06_eda_movie_revenue.ipynb
├─ models/
├─ README.md
├─ PROJECT.md
├─ USAGE.md
├─ .gitignore
├─ requirements.txt
└─ LICENSE (optional)
```

## Next Steps & Future Work

- Validate model performance on held-out test data and production scenarios.
- Gather feedback from domain experts (producers, studios) on feature relevance and predictions.
- Scale to larger datasets or real-time prediction scenarios.
- Explore deployment options (Flask API, Streamlit dashboard) for stakeholder use.
