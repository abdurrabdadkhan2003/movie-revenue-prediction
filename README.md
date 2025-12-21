# Movie Revenue Prediction: Forecasting Box Office Success

**One-Line Pitch:** An end-to-end regression pipeline that predicts movie revenue using metadata and engagement signals to support data-driven decision-making in the film industry.

## Background & Motivation
The movie industry is a high-stakes business where production costs often exceed hundreds of millions of dollars. Predicting a film's financial success (Box Office Revenue) is critical for producers, investors, and distributors. This project leverages machine learning to identify the key drivers of revenue and provides a framework for forecasting performance before or shortly after release.

## Dataset Description
The analysis is based on a dataset of **1,000 movies** with **18 features**, including:
- **Financials:** Budget (highly correlated with revenue).
- **Engagement:** Trailer views, trailer likes, and trailer engagement rates.
- **Metadata:** Genre, Director, Runtime, MPAA Rating, and Release Date.
- **Audience Feedback:** Vote average and vote count.
- **Cast:** Cast popularity scores.

## The Regression Modeling Process (Simplified)
Regression is a statistical method used to understand the relationship between a dependent variable (Revenue) and one or more independent variables (like Budget or Trailer Views).
- **The Goal:** To find a \"line of best fit\" that predicts revenue based on the input features.
- **How it works:** The model assigns \"weights\" (coefficients) to each feature. For example, if the budget has a high positive weight, it means an increase in budget generally leads to an increase in predicted revenue.
- **Evaluation:** We measure how far off our predictions are from the actual values using metrics like **MAE** (average error) and **R²** (how much of the variance we successfully captured).

## Model Overview & Approach
I implemented a structured data science workflow:
1. **Exploratory Data Analysis (EDA):** Analyzed distributions, identified outliers, and mapped correlations (e.g., Budget vs. Revenue).
2. **Preprocessing:** 
   - Handled categorical variables using **One-Hot Encoding** (Genre, Director).
   - Applied **Standard Scaling** to numerical features to ensure model convergence.
3. **Regression Modeling:** 
   - **Baseline:** Linear Regression.
   - **Regularization:** Ridge and Lasso regression to prevent overfitting.
   - **Advanced:** Random Forest Regressor for capturing non-linear relationships.

## Evaluation Metrics & Results
The models were evaluated using standard regression metrics:
- **Best Model:** Ridge Regression
- **R-Squared (R²):** **0.53** (The model explains 53% of the variance in revenue).
- **Mean Absolute Error (MAE):** ~4.6B (Note: Revenue is in local currency/unscaled units as per the dataset).
- **Root Mean Squared Error (RMSE):** ~7.3B.

*Practical Value:* Budget and trailer views are the most significant predictors. This suggests that financial investment and early audience engagement are primary levers for box office success.

## Setup, Installation & How to Run
### Prerequisites
- Python 3.9+
- Jupyter Notebook or Google Colab

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abdurrabdadkhan2003/movie-revenue-prediction.git
   cd movie-revenue-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
Navigate to the `notebooks/` folder and run the notebooks in sequence:
1. `01_Exploratory_Analysis.ipynb`: Data cleaning and initial exploration.
2. `02_Regression_Modeling.ipynb`: Baseline and regularized regression models.
3. `03_Model_Evaluation.ipynb`: Feature importance and result visualization.

## Example Output & Interpretations
### Visualizations
![Actual vs Predicted Revenue](https://via.placeholder.com/600x400?text=Actual+vs+Predicted+Revenue+Scatter+Plot)
*Interpretation: The scatter plot shows a strong positive correlation, with the model performing well for mid-budget films but occasionally underestimating blockbuster breakouts.*

![Feature Importance](https://via.placeholder.com/600x400?text=Top+10+Feature+Importances)
*Interpretation: Budget and Trailer Views dominate the feature importance, suggesting that financial investment and marketing reach are the primary levers for revenue.
