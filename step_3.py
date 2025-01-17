import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load and preprocess the data
file_path = 'filtered_with_review.csv'
data = pd.read_csv(file_path)

# there is 2 feature of the movie name, so we drop one of them
data = data.drop(columns=['orig_title'])

# Convert the 'date' column to datetime format
data['date_x'] = pd.to_datetime(data['date_x'], errors='coerce')
data['year'] = data['date_x'].dt.year
data['month'] = data['date_x'].dt.month
data['day'] = data['date_x'].dt.day
data['weekday'] = data['date_x'].dt.weekday
data = data.drop(columns=['date_x'])

# Fill missing values in new date columns
for col in ['year', 'month', 'day', 'weekday']:
    data[col] = data[col].fillna(data[col].median())

# Handle rare categories in 'country' and 'orig_lang'
threshold = 0.01  # Consider categories with less than 1% frequency as rare
for col in ['country', 'orig_lang']:
    counts = data[col].value_counts(normalize=True)
    rare_categories = counts[counts < threshold].index
    data[col] = data[col].apply(lambda x: x if x not in rare_categories else 'Other')

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['country', 'orig_lang'], drop_first=True)

# Handle missing values in other columns
data['budget_x'] = data['budget_x'].fillna(data['budget_x'].median())
data['score'] = data['score'].fillna(data['score'].median())
data['names'] = data['names'].fillna('Unknown')
data['genre'] = data['genre'].fillna('Unknown')
data['crew'] = data['crew'].fillna('')
data['review'] = data['review'].fillna('')

# Apply Multi-Hot Encoding to the 'genre' column
data['genre'] = data['genre'].str.split(',')
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(data['genre'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
data = pd.concat([data, genre_df], axis=1)
data = data.drop(columns=['genre'])

# Label encode 'names' (movie titles)
encoder = LabelEncoder()
data['movie_title_encoded'] = encoder.fit_transform(data['names'])
data = data.drop(columns=['names'])

# Label encode the 'status' column
status_encoder = LabelEncoder()
data['status_encoded'] = status_encoder.fit_transform(data['status'])
data = data.drop(columns=['status'])

# Process text features 'crew' and 'overview' using Bag of Words
vectorizer_crew = TfidfVectorizer(max_features=500,
                                  ngram_range=(1, 2),
                                  min_df=2,
                                  max_df=0.95, stop_words='english')
vectorizer_review = TfidfVectorizer(max_features=500,
                                    ngram_range=(1, 2),
                                    min_df=2,
                                    max_df=0.95, stop_words='english')

X_crew = vectorizer_crew.fit_transform(data['crew']).toarray()
X_review = vectorizer_review.fit_transform(data['review']).toarray()

# print the 500 words from review
selected_words = vectorizer_review.get_feature_names_out()
print(selected_words)

crew_df = pd.DataFrame(X_crew, columns=[f"crew_{i}" for i in range(X_crew.shape[1])])
review_df = pd.DataFrame(X_review, columns=[f"review_{i}" for i in range(X_review.shape[1])])

data = pd.concat([data, crew_df, review_df], axis=1)
data = data.drop(columns=['crew', 'overview', 'review'])

print(data.shape)

# Remove outliers using IQR
Q1 = data['revenue'].quantile(0.25)
Q3 = data['revenue'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['revenue'] < (Q1 - 1.5 * IQR)) | (data['revenue'] > (Q3 + 1.5 * IQR)))]

print(data.shape)

# Separate features (X) and target (y)
X = data.drop(columns=['revenue'])
y = data['revenue']

# Normalize features and target
scaler = StandardScaler()
X = scaler.fit_transform(X)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.values.reshape(-1, 1)).squeeze()

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed_value)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed_value)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# Define the neural network
class SimpleRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleRegressionNN, self).__init__()

        # First layer - reduced size from 256 to 128
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=True, momentum=0.1)  # Modified BatchNorm
        self.dropout1 = nn.Dropout(p=0.6)

        # Second layer - reduced size from 128 to 64
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64, track_running_stats=True, momentum=0.1)  # Modified BatchNorm
        self.dropout2 = nn.Dropout(p=0.5)

        # Output layer - directly from 64 to 1
        self.output = nn.Linear(64, 1)

        # Activation function
        self.relu = nn.ReLU()

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        # Ensure batch size > 1 for BatchNorm during training
        if self.training and x.size(0) == 1:
            # If batch size is 1 during training, repeat the sample
            x = x.repeat(2, 1)

        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.output(x)

        # If we duplicated the sample, take only the first prediction
        if self.training and x.size(0) == 2:
            x = x[0].unsqueeze(0)

        return x


input_size = X_train.shape[1]
model = SimpleRegressionNN(input_size)

# Set up optimizer, loss function, and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# Train the model with early stopping
epochs = 150
best_val_loss = float('inf')
patience = 20
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    scheduler.step(val_loss / len(val_loader))
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!\n")
            break

# Evaluate the model
model.eval()
y_pred_list = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs).squeeze()
        y_pred_list.append(outputs)

y_pred = torch.cat(y_pred_list).numpy()
y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze()

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE):        ${mae:,.2f}")
print(f"Mean Squared Error (MSE):         ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE):   ${np.sqrt(mse):,.2f}")

# Model performance scatter plot (actual vs predicted)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect '
                                                                                                    'Prediction Line')
plt.title("Actual vs Predicted Revenue - model 3")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
# Set axis limits
plt.xlim(-20000000, 650000000)
plt.ylim(-20000000, 790000000)
# Make sure the intervals are consistent
plt.gca().set_aspect('auto')  # Allows the axes to scale automatically
plt.legend()
plt.show()

# Training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
plt.title("Training and Validation Loss Over Epochs - model 3")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# Set axis limits
plt.xlim(-2, 152)
plt.ylim(-0.1, 2.2)
# Make sure the intervals are consistent
plt.gca().set_aspect('auto')  # Allows the axes to scale automatically
plt.legend()
plt.show()