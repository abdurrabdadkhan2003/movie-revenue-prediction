# Usage Guide - Movie Revenue Prediction

This guide shows how to use the Movie Revenue Prediction project, including running notebooks, loading data, and making predictions.

## Prerequisites

Ensure you have completed the setup steps from the main README.md:

1. Cloned the repository
2. Created and activated a virtual environment
3. Installed dependencies: `pip install -r requirements.txt`

## Running Jupyter Notebooks

### Launch Jupyter Server

```bash
# In the project root directory with venv activated
jupyter notebook
```

This opens Jupyter in your default browser at `http://localhost:8888`.

### Notebook Execution Order

Run the notebooks in this order for best results:

#### 1. **01_eda_movie_revenue.ipynb**
   - Load movie dataset
   - Display basic info (shape, columns, dtypes)
   - Show first few rows and basic statistics
   - **Purpose**: Understand data structure and baseline statistics

#### 2. **02_eda_movie_revenue.ipynb**
   - Value counts for categorical columns
   - Data type summary
   - Initial distribution plots
   - **Purpose**: Identify data types and categorical distributions

#### 3. **03_eda_movie_revenue.ipynb**
   - Correlation heatmaps
   - Feature relationships and patterns
   - Outlier detection
   - **Purpose**: Understand feature relationships and data quality

#### 4. **04_eda_movie_revenue.ipynb**
   - Feature importance from baseline models
   - Actual vs. predicted plots
   - Residual analysis
   - **Purpose**: Identify key predictive features

#### 5. **05_eda_movie_revenue.ipynb**
   - Ridge Regression with regularization
   - Lasso Regression with feature selection
   - Model comparison and evaluation
   - **Purpose**: Test regularized regression models

#### 6. **06_eda_movie_revenue.ipynb**
   - Random Forest Regressor
   - Ensemble methods
   - Final model evaluation and visualization
   - **Purpose**: Build and evaluate advanced ensemble models

## Working with the Data

### Loading Data Manually

If you want to load the movie revenue data in your own script:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/movies.csv')  # Replace with actual filename

# Display basic info
print(df.head())
print(df.info())
print(df.describe())
```

### Data Columns

Common columns in the dataset (adapt to your actual data):

- `revenue`: Target variable (box office revenue in dollars)
- `budget`: Production budget
- `popularity`: Popularity score
- `vote_count`: Number of votes/ratings
- `vote_average`: Average rating (1-10 scale)
- `release_date`: Movie release date
- `genres`: Movie genres
- `runtime`: Movie duration in minutes

## Making Predictions

### Using a Trained Model

If models are saved (e.g., with joblib), load and use them:

```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('models/random_forest_model.pkl')

# Prepare your data (example)
X_new = np.array([
    [50000000, 2000, 8.5, 5000],  # Example: budget, popularity, rating, vote_count
])

# Make predictions
predictions = model.predict(X_new)
print(f"Predicted Revenue: ${predictions[0]:,.0f}")
```

### Feature Engineering

To replicate feature engineering from the notebooks:

```python
# Example: Log-transform budget for skewed distributions
df['log_budget'] = np.log1p(df['budget'])

# Example: Create interaction features
df['budget_x_popularity'] = df['budget'] * df['popularity']

# Example: Normalize ratings to 0-1
df['rating_normalized'] = df['vote_average'] / 10.0
```

## Interpreting Results

### Model Evaluation Metrics

- **R² (Coefficient of Determination)**: Proportion of variance explained (0-1 scale)
  - Higher is better. R² = 0.8 means model explains 80% of variance.

- **RMSE (Root Mean Squared Error)**: Average prediction error in dollars
  - Lower is better. Measured in same units as target (revenue).

- **MAE (Mean Absolute Error)**: Average absolute prediction error
  - Lower is better. More interpretable than RMSE.

### Example Model Output

```
Linear Regression
  R² Score: 0.72
  RMSE: $45,000,000
  MAE: $35,000,000

Random Forest
  R² Score: 0.85
  RMSE: $28,000,000
  MAE: $20,000,000
```

## Troubleshooting

### Issue: "Module not found" error

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Notebooks won't run

**Solution**: Check that Jupyter kernel matches your Python environment:
```bash
python -m ipykernel install --user --name venv --display-name "venv"
```

Then select the "venv" kernel in Jupyter.

### Issue: Data file not found

**Solution**: Ensure data files are in the `data/` directory or update the file path in notebooks:
```python
df = pd.read_csv('./data/your_file.csv')  # Use relative path from notebook location
```

## Advanced Usage

### Cross-Validation for Model Evaluation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    cv=5,  # 5-fold cross-validation
    scoring='r2'
)

print(f"CV R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=3,
    scoring='r2'
)

grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best R² Score: {grid_search.best_score_:.4f}")
```

## Next Steps

1. **Explore the Notebooks**: Open each notebook sequentially and understand the analysis.
2. **Experiment**: Modify feature engineering steps and retrain models.
3. **Deploy**: Wrap the best model in a Flask/Streamlit app for predictions.
4. **Share**: Document findings and results in a presentation or report.

For more technical details, see `PROJECT.md`.
