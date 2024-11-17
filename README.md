# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1. Import Libraries: 
* Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
#### 2. Load Dataset: 
* Load the dataset containing car prices and relevant features.
#### 3. Data Preprocessing: 
* Handle missing values and perform feature selection if necessary.
#### 4. Split Data: 
* Split the dataset into training and testing sets.
#### 5. Train Model: 
* Create a linear regression model and fit it to the training data.
#### 6. Make Predictions: 
* Use the model to make predictions on the test set.
#### 7. Evaluate Model: 
* Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
#### 8. Check Assumptions: 
* Plot residuals to check for homoscedasticity, normality, and linearity.
#### 9. Output Results: 
* Display the predictions and evaluation metrics.

## Program:
```python
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: SANJUSHRI A
RegisterNumber:  212223040187
*/

# Import necessary libraries
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import shapiro
import numpy as np

# Load the dataset from the file path
data = pd.read_csv(r"C:\Users\admin\Downloads\CarPrice_Assignment (1).csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Data Preprocessing
# Handle missing values (if any)
data = data.dropna()  # Drop rows with missing values

# Select features and target variable
# Assume 'price' is the target variable and 'horsepower', 'curbweight', 'enginesize', and 'highwaympg' are features
X = data[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Assumption 1: Linearity
# Check residuals vs predicted values for linearity
plt.scatter(y_pred, y_test - y_pred, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Linearity Check)')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Assumption 2: Homoscedasticity
# Homoscedasticity means the residuals should have constant variance.
plt.scatter(y_pred, y_test - y_pred, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Homoscedasticity Check')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()

# Assumption 3: Normality
# Use the Shapiro-Wilk test for normality of residuals
residuals = y_test - y_pred
stat, p_value = shapiro(residuals)
print("Shapiro-Wilk Test:")
print(f"Statistic: {stat}, P-value: {p_value}")
if p_value > 0.05:
    print("Residuals are normally distributed.")
else:
    print("Residuals are not normally distributed.")

# Assumption 4: Multicollinearity
# Check Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_with_const = add_constant(X)  # Add a constant column for VIF calculation
vif_data = pd.DataFrame({
    "Feature": X_with_const.columns,
    "VIF": [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
})
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# Plotting Residual Distribution for Normality
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

```


## Output:

![image](https://github.com/user-attachments/assets/8bdd87b5-648b-40ef-b80f-0ba3b0e7b8a6)

![image](https://github.com/user-attachments/assets/e2bbb19f-7acc-4c91-ab66-fbfd4f1e2cba)

![image](https://github.com/user-attachments/assets/d93524d6-5d47-4e6b-9652-142af5959fcd)

![image](https://github.com/user-attachments/assets/cf12df94-957c-45a8-b6e3-c1f6c8aea59b)

![image](https://github.com/user-attachments/assets/2a2bb44a-4f0a-4614-ae54-58cd692e85a8)





## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
