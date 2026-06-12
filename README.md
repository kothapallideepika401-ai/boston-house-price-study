# boston-house-price-study
Machine Learning project for predicting Boston house prices using regression models, data preprocessing, feature analysis, and visualization techniques.
# Boston House Price Prediction using Machine Learning

## Project Overview

The Boston House Price Prediction project uses machine learning techniques to estimate housing prices based on various economic, environmental, and demographic factors. The project demonstrates data preprocessing, exploratory analysis, regression modeling, and performance evaluation.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

## Machine Learning Algorithm

* Random Forest Regressor

## Dataset Features

* CRIM (Crime Rate)
* ZN (Residential Land Zone)
* INDUS (Industrial Area)
* CHAS (Charles River Proximity)
* NOX (Nitric Oxide Concentration)
* RM (Average Rooms)
* AGE (Property Age)
* DIS (Distance to Employment Centers)
* TAX (Property Tax Rate)
* PTRATIO (Student-Teacher Ratio)
* LSTAT (Lower Status Population Percentage)

Target Variable:

* MEDV (Median House Value)

## Project Workflow

1. Load Dataset
2. Data Cleaning
3. Missing Value Handling
4. Feature Selection
5. Train-Test Split
6. Random Forest Regression
7. Model Evaluation
8. Visualization of Results

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

## Installation

pip install -r requirements.txt

## Run the Project

python boston_house_price_prediction.py

## Results

* House Price Prediction
* Actual vs Predicted Visualization
* Performance Metrics
* CSV Export of Predictions

## Future Enhancements

* XGBoost Regressor
* Hyperparameter Tuning
* Streamlit Dashboard
* Real-Time House Price Estimation

## Author

Machine Learning Internship Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("housing.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================
# TARGET COLUMN
# ==========================

target_column = "MEDV"

X = df.drop(columns=[target_column])
y = df[target_column]

# ==========================
# NUMERIC FEATURES
# ==========================

numeric_features = X.columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer",
         SimpleImputer(strategy="median"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num",
         numeric_transformer,
         numeric_features)
    ]
)

# ==========================
# MODEL
# ==========================

model = Pipeline(
    steps=[
        ("preprocessor",
         preprocessor),

        ("regressor",
         RandomForestRegressor(
             n_estimators=300,
             random_state=42
         ))
    ]
)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================

model.fit(X_train, y_train)

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("\nMean Squared Error:", mse)
print("\nR2 Score:", r2)

# ==========================
# R2 SCORE VISUALIZATION
# ==========================

plt.figure(figsize=(6,4))

plt.bar(
    ["R2 Score"],
    [r2 * 100],
    color="green"
)

plt.ylabel("Score (%)")
plt.title("Boston House Price Prediction")

plt.text(
    0,
    r2 * 100 + 1,
    f"{r2*100:.2f}%"
)

plt.show()

# ==========================
# ACTUAL VS PREDICTED
# ==========================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title(
    "Actual vs Predicted House Prices"
)

plt.show()

# ==========================
# SAVE PREDICTIONS
# ==========================

results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

results.to_csv(
    "house_price_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
