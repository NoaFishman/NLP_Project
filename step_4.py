import pandas as pd
import numpy as np
import random
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

current_time = datetime.datetime.now()
print(f"\033[35mCurrent time: {current_time}\033[0m")


def get_bert_embeddings(texts, tokenizer, model, max_length=512):
    """
    Generate BERT embeddings for a list of texts
    """
    embeddings_list = []
    batch_size = 32

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
                model = model.cuda()

            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(batch_embeddings)

    return np.vstack(embeddings_list)


# Load the data
file_path = 'filtered_with_review.csv'
data = pd.read_csv(file_path)

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
threshold = 0.01
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

# Initialize BERT for text processing
print("Initializing BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Label encode the 'status' column
status_encoder = LabelEncoder()
data['status_encoded'] = status_encoder.fit_transform(data['status'])
data = data.drop(columns=['status'])


# return only the first 10 actors and their characters from the crew list
def get_first_ten(text):
    if pd.isna(text):  # Handle NaN/None values
        return text
    names = [name.strip() for name in str(text).split(',')]
    return ', '.join(names[:10])


print("Processing movie crew with BERT...")
data['crew'] = data['crew'].fillna('')
data['crew'].str.split(',')
# Apply the function to the specified column
data['crew'] = data['crew'].apply(get_first_ten)
crew_embeddings = get_bert_embeddings(data['crew'].tolist(), tokenizer, model)
crew_df = pd.DataFrame(
    crew_embeddings,
    columns=[f"crew_bert_{i}" for i in range(crew_embeddings.shape[1])]
)

# Process overview using BERT
print("Processing review text data with BERT...")
data['review'] = data['review'].fillna('')
review_embeddings = get_bert_embeddings(data['review'].tolist(), tokenizer, model)
review_df = pd.DataFrame(
    review_embeddings,
    columns=[f"review_bert_{i}" for i in range(review_embeddings.shape[1])]
)

# Process movie names using BERT
print("Processing movie names with BERT...")
names_embeddings = get_bert_embeddings(data['names'].tolist(), tokenizer, model)
names_df = pd.DataFrame(
    names_embeddings,
    columns=[f"name_bert_{i}" for i in range(names_embeddings.shape[1])]
)
# Remove the original 'names' column as we now have embeddings
data = data.drop(columns=['names'])

print("\nProcessing movie genre with BERT...")
genre_embeddings = get_bert_embeddings(data['genre'].tolist(), tokenizer, model)
genre_df = pd.DataFrame(
    genre_embeddings,
    columns=[f"genre_bert_{i}" for i in range(genre_embeddings.shape[1])]
)
data = data.drop(columns=['genre'])

# Combine all features
data = pd.concat([data, names_df, crew_df, review_df, genre_df], axis=1)
data = data.drop(columns=['crew', 'overview', 'review'])

print("Data shape after processing:", data.shape)

# Remove outliers using IQR
Q1 = data['revenue'].quantile(0.25)
Q3 = data['revenue'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['revenue'] < (Q1 - 1.5 * IQR)) | (data['revenue'] > (Q3 + 1.5 * IQR)))]

print("Data shape after outlier removal:", data.shape)

data.to_csv("output_just_review.csv", index=False)
print("saved the data frame to output.csv")

file_path = 'output_just_review.csv'
data = pd.read_csv(file_path)

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


# Define the neural network
class EnhancedRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(EnhancedRegressionNN, self).__init__()

        self.dropout = nn.Dropout(0.3)

        # Deeper architecture with skip connections
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(1024)
        self.ln3 = nn.LayerNorm(512)
        self.ln4 = nn.LayerNorm(256)
        self.ln5 = nn.LayerNorm(128)

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)

        # Skip connections
        self.skip1 = nn.Linear(input_size, 512)
        self.skip2 = nn.Linear(512, 256)
        self.skip3 = nn.Linear(256, 128)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Input processing
        x = self.ln1(x)

        # First block
        skip = self.skip1(x)
        x = self.fc1(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + skip

        # Second block
        skip = self.skip2(x)
        x = self.ln3(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x + skip

        # Third block
        skip = self.skip3(x)
        x = self.ln4(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = x + skip

        # Output
        x = self.ln5(x)
        x = F.gelu(x)
        x = self.fc5(x)
        return x


def train_model(model, train_loader, val_loader, epochs=150):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.02,
        betas=(0.9, 0.999)
    )

    # Combine MSE and Huber loss
    mse_criterion = nn.MSELoss()
    huber_criterion = nn.HuberLoss(delta=1.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            # Combined loss
            loss = 0.7 * mse_criterion(outputs, labels) + 0.3 * huber_criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = mse_criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return train_losses, val_losses


# Rest of the code remains the same
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

input_size = X_train.shape[1]
model = EnhancedRegressionNN(input_size)

# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader)

# Load best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate the model
print("\nEvaluating model...")
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
rmse = np.sqrt(mse)

print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE):        ${mae:,.2f}")
print(f"Mean Squared Error (MSE):         ${mse:,.2f}")
print(f"Root Mean Squared Error (RMSE):   ${np.sqrt(mse):,.2f}")

# Visualizations
print("\nGenerating visualizations...")

# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         color='red', linestyle='--', label='Perfect Prediction Line')
plt.title("Actual vs Predicted Revenue - model 4")
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
# Set axis limits
plt.xlim(-20000000, 650000000)
plt.ylim(-20000000, 790000000)
# Make sure the intervals are consistent
plt.gca().set_aspect('auto')  # Allows the axes to scale automatically
plt.legend()
plt.show()

# Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
plt.title("Training and Validation Loss Over Epochs - model 4")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# Set axis limits
plt.xlim(-2, 152)
plt.ylim(-0.1, 2.2)
# Make sure the intervals are consistent
plt.gca().set_aspect('auto')  # Allows the axes to scale automatically
plt.legend()
plt.show()

end_time = datetime.datetime.now()
elapsed_time = end_time - current_time
print(f"\033[35mTotal time: {elapsed_time}\033[0m")