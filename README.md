# Movie Revenue Prediction

Data Scientist & ML Engineer · End-to-end regression pipeline for business decision support

## 📋 Overview

Movie Revenue Prediction is a regression-based machine learning project that estimates box office revenue using pre-release and metadata features such as budget, votes, popularity, and engagement signals. The goal is to showcase a complete data science workflow from exploratory analysis to model deployment readiness for real-world decision-making in the film industry.

This repository is structured to be portfolio-ready, highlighting clean notebooks, modular code, and clear documentation suitable for recruiters and collaborators.

## 🎯 Key Features

- End-to-end ML pipeline: data loading, cleaning, feature engineering, model training, and evaluation.
- Supervised regression models (e.g., Linear Regression, Tree-based models, Ensembles) for revenue prediction.
- Exploratory Data Analysis (EDA) with visualizations for distributions, correlations, and outliers.
- Feature importance analysis to understand key drivers of movie revenue (budget, votes, popularity).
- Robust preprocessing: handling missing values, encoding categorical variables, and scaling numerical features.
- Reproducible experimentation via organized Jupyter notebooks in the `notebooks/` directory.
- Clear separation of data, notebooks, and source code for maintainability.

## 📊 Project Artifacts / Deliverables

- `notebooks/`: step-by-step EDA, feature engineering, and model training notebooks.
- `data/`: raw/processed datasets or data loading instructions.
- `models/` (if present): serialized models for reuse or deployment.
- `PROJECT.md`: in-depth technical and methodological documentation.
- `USAGE.md`: examples for running notebooks and using trained models.
- `.gitignore`: Python-focused ignore rules for a clean repository.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ (or your project's version)
- Git
- Recommended: virtual environment (`venv` or `conda`)

### Installation

```bash
# Clone the repository
git clone https://github.com/abdurrabdadkhan2003/movie-revenue-prediction.git
cd movie-revenue-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

- Open Jupyter/Colab and run notebooks in the `notebooks/` folder in order:
  1. `01_eda_movie_revenue.ipynb` - Initial exploration and statistics
  2. `02_eda_movie_revenue.ipynb` - Value counts and distributions
  3. `03_eda_movie_revenue.ipynb` - Feature analysis and correlations
  4. `04_eda_movie_revenue.ipynb` - Feature importance and predictions
  5. `05_eda_movie_revenue.ipynb` - Ridge and Lasso regression
  6. `06_eda_movie_revenue.ipynb` - Random Forest and advanced models

For more detailed usage and commands, see `USAGE.md`.

## 🛠 Technologies & Skills Demonstrated

**Technologies**

- Python (pandas, NumPy, matplotlib/seaborn)
- Scikit-learn for regression modeling and evaluation
- Jupyter/Colab notebooks for experimentation
- Git & GitHub for version control
- joblib/pickle for model persistence

**Data & ML Skills**

- Problem framing for supervised regression
- Data cleaning, feature engineering, and exploratory analysis
- Model selection, tuning, and performance evaluation (R², RMSE)
- Interpreting feature importance and business impact
- Reproducible, documented ML workflows

**Soft Skills**

- Clear technical communication through structured documentation
- Experiment design and evidence-based model comparison
- Ownership of end-to-end project lifecycle

## 📈 Results & Impact

- Built regression models capable of explaining variance in revenue using key predictors such as budget, votes, and engagement metrics.
- Demonstrated how data-driven insights support producers and investors in forecasting box office performance and making informed budgeting decisions.
- Showcased a reusable template for future business-focused ML regression projects.

*Note: Replace with actual metrics once final evaluation is complete (e.g., "Best model achieved R² = 0.85, RMSE = $50M on test set")*

## 🔍 Future Enhancements

- Integrate additional features such as genre embeddings, cast/crew statistics, and social media sentiment.
- Experiment with advanced models (Gradient Boosting, XGBoost, neural networks) for improved performance.
- Wrap the model in a simple API or dashboard for non-technical stakeholders.
- Add automated evaluation scripts and unit tests for data and model pipelines.
- Containerize the project (Docker) for easier deployment.

## 👤 About This Project

This project was designed and implemented as part of a personal data science portfolio to demonstrate:

- Ability to take real-world business problems and translate them into predictive modeling tasks.
- Competence in building and documenting full ML workflows rather than isolated notebooks.
- Practical, industry-aligned thinking about how predictive models support decision-making in media and entertainment.

---

**Questions or suggestions?** Feel free to open an issue or reach out via GitHub.
