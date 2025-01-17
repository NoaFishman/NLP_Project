import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


def clean_text(text):
    """Clean text data by removing special characters and converting to lowercase"""
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.lower().strip()


def combine_text_features(row, text_columns):
    """Combine multiple text columns into a single string with spacing"""
    return ' . '.join(clean_text(str(row[col])) for col in text_columns if pd.notna(row[col]))


def process_dates(X, date_feature):
    """Process dates into day, month, year, and day_of_week features and fill the missing values with the median"""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[date_feature])

    dates = pd.to_datetime(X[date_feature], errors='coerce')

    date_features = pd.DataFrame({
        'day': dates.dt.day,
        'month': dates.dt.month,
        'year': dates.dt.year,
        'day_of_week': dates.dt.dayofweek
    })

    for column in date_features.columns:
        median_value = date_features[column].median()
        date_features[column] = date_features[column].fillna(median_value)

    return date_features.values


class OurLinearRegression(LinearRegression):
    """Linear Regression that ensures non-negative predictions.
    Inherits from the regular Linear regression class"""

    def predict(self, X):
        predictions = super().predict(X)
        return np.maximum(predictions, 0)  # Ensures all predictions are non-negative


# Load the dataset
file_path = 'filtered_with_review.csv'
df = pd.read_csv(file_path)

# Clean revenue data: remove negative values and convert to numeric
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df = df[df['revenue'].notna()]  # Remove NaN revenue entries

# Define feature groups
text_features = ['names', 'review', 'crew']
numerical_features = ['budget_x', 'score']
categorical_features = ['genre', 'status', 'orig_lang', 'country']
date_feature = 'date_x'

# Create preprocessing pipelines
text_transformer = Pipeline([
    ('text_combiner', FunctionTransformer(
        lambda X: pd.DataFrame(X)[text_features].apply(
            lambda row: combine_text_features(row, text_features), axis=1
        )
    )),
    ('vectorizer', TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    ))
])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', PowerTransformer(standardize=True))
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'))
])

date_transformer = FunctionTransformer(lambda X: process_dates(X, date_feature), validate=False)

# Combine all preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_features),
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('date', date_transformer, [date_feature])
    ],
    remainder='drop'
)

# Create the full pipeline with Non-Negative Linear Regression
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', OurLinearRegression())
])

# Prepare target variable and remove outliers
y = df['revenue']
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
mask = (y >= (Q1 - 1.5 * IQR)) & (y <= (Q3 + 1.5 * IQR))
df_cleaned = df[mask]
y = df_cleaned['revenue']
X = df_cleaned.drop('revenue', axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("\nTraining the model...")
model_pipeline.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE):        ${test_mae:,.2f}")
print(f"Mean Squared Error (MSE):         ${test_mse:,.2f}")
print(f"Root Mean Squared Error (RMSE):   ${np.sqrt(test_mse):,.2f}")

# Plotting the predictions vs actual revenue
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect '
                                                                                                    'Prediction Line')
plt.title("Actual vs Predicted Revenue - model 2")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.xlim(-20000000, 650000000)
plt.ylim(-20000000, 790000000)
plt.gca().set_aspect('auto')
plt.legend()
plt.show()


def predict_from_test_set(X_test, y_test, model_pipeline, index):
    """
    Predict revenue for a specific test set entry and show details
    """
    # Get the test point and actual value
    test_point = X_test.iloc[[index]]  # Using double brackets to keep DataFrame structure
    actual_value = y_test.iloc[index]

    # Get the movie details before preprocessing
    movie_name = test_point['names'].iloc[0]

    # Make prediction using the pipeline
    prediction = model_pipeline.predict(test_point)[0]

    # Print the movie details and prediction
    print(f"\nMovie Details:")
    print(f"Name: {movie_name}")
    print(f"\nRevenue Comparison:")
    print(f"Actual Revenue:    ${actual_value:,.2f}")
    print(f"Predicted Revenue: ${prediction:,.2f}")
    print(f"Prediction Error:  ${abs(actual_value - prediction):,.2f}")
    print(f"Error Percentage:  {abs(actual_value - prediction) / actual_value * 100:.1f}%")


# Choose a random index from the test set to predict a point
print("\n\n\nRandom Movie Prediction Example:")
random_index = np.random.randint(0, len(X_test))

# Predict and print for the selected test point
predict_from_test_set(X_test, y_test, model_pipeline, random_index)


# Add option to predict multiple random movies
def predict_multiple_random_movies(X_test, y_test, model_pipeline, n=3):
    print(f"\nPredicting {n} Random Movies:")
    for _ in range(n):
        random_index = np.random.randint(0, len(X_test))
        predict_from_test_set(X_test, y_test, model_pipeline, random_index)
        print("\n" + "=" * 50)


# Predict 3 random movies
predict_multiple_random_movies(X_test, y_test, model_pipeline, n=3)
