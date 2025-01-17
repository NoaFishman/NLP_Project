import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# Load the dataset
file_path = 'filtered_with_review.csv'
df = pd.read_csv(file_path)

# Drop rows with missing values in the target column - even though in our data there are no lines like this.
df = df.dropna(subset=['revenue'])

# Prepare target variable and remove outliers
print(f'Number of lines before applying IQR method: {len(df)}')
y = df['revenue']
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
mask = (y >= (Q1 - 1.5 * IQR)) & (y <= (Q3 + 1.5 * IQR))
df_cleaned = df[mask]
y = df_cleaned['revenue']
X = df_cleaned.drop('revenue', axis=1)
print(f'Number of lines after applying IQR method: {len(df_cleaned)}\n')

# Split the data into train and test sets
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

# Compute the average revenue from the training set
average_revenue = train_df['revenue'].mean()
print(f"Average Revenue is ${average_revenue:,.2f}")

# Predict the average revenue for both training and test sets
train_baseline_preds = [average_revenue] * len(train_df)
test_baseline_preds = [average_revenue] * len(test_df)

actual = test_df['revenue']
predicted = pd.Series(test_baseline_preds, index=test_df.index)

# Evaluate the Baseline Model
baseline_train_mse = mean_squared_error(train_df['revenue'], train_baseline_preds)
baseline_test_mse = mean_squared_error(test_df['revenue'], test_baseline_preds)
baseline_test_mae = mean_absolute_error(test_df['revenue'], test_baseline_preds)

print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE):        ${baseline_test_mae:,.2f}")
print(f"Mean Squared Error (MSE):         ${baseline_test_mse:,.2f}")
print(f"Root Mean Squared Error (RMSE):   ${np.sqrt(baseline_test_mse):,.2f}")


# Select a random point from the test set and compare predicted vs actual
random_index = random.choice(test_df.index)
random_actual = test_df.loc[random_index, 'revenue']
random_predicted = predicted.loc[random_index]

print("\n\nRandom Point Evaluation:")
print(f"Actual Revenue: {random_actual:.2f}")
print(f"Predicted Revenue: {random_predicted:.2f}")

# Plotting the predictions vs actual revenue
plt.figure(figsize=(10, 6))
plt.scatter(actual, predicted, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color='red', linestyle='--', label='Perfect '
                                                                                                    'Prediction Line')
plt.title("Actual vs Predicted Revenue - model 1")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
# Set axis limits
plt.xlim(-20000000, 650000000)
plt.ylim(-20000000, 790000000)
# Make sure the intervals are consistent
plt.gca().set_aspect('auto')  # Allows the axes to scale automatically
plt.legend()
plt.show()