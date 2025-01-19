# Movie Revenue Prediction using NLP Techniques

This project explores the application of Natural Language Processing (NLP) and machine learning techniques to predict movie revenues using various features including reviews, crew information, and genres. The project progresses from a baseline model to an advanced neural network utilizing BERT embeddings.

## Project Overview

The project implements four increasingly complex models to predict movie revenues:
1. Baseline Model (Average Revenue Prediction)
2. Linear Regression with TF-IDF
3. Basic Neural Network
4. Advanced Neural Network with BERT Embeddings

Each model demonstrates improvements in prediction accuracy, with the final BERT-based model achieving the best performance.

## Dataset

The project combines two datasets from Kaggle to data set call filtered_with_review:
- [IMDB Movies Dataset](https://www.kaggle.com/datasets/ashpalsingh1525/imdb-movies-dataset/data)
- [48,000+ Movies Dataset](https://www.kaggle.com/datasets/yashgupta24/48000-movies-dataset)

### Features Include:
- Movie names
- Release dates
- Scores
- Genres
- Reviews
- Crew information
- Budget
- Revenue
- Country
- Language
- Status

## Results

| Model | MAE ($) | MSE ($) | RMSE ($) |
|-------|----------|---------|-----------|
| Baseline | 121,433,652.96 | 23,676,641,432,614,836.00 | 153,872,159.38 |
| Linear Regression | 92,968,624.58 | 16,097,876,478,672,158.00 | 126,877,407.28 |
| Simple Neural Network | 81,014,284.61 | 13,285,544,772,123,966.00 | 115,262,937.55 |
| Advanced Neural Network | 73,535,891.10 | 11,611,688,000,320,616.00 | 107,757,542.66 |

Overall improvements from baseline to final model:
- 39.44% reduction in Mean Absolute Error
- 50.96% reduction in Mean Squared Error
- 29.97% reduction in Root Mean Squared Error

## Model Architectures

### Model 1: Baseline
- Simple average revenue prediction
- Used as a benchmark for comparison

### Model 2: Linear Regression
- TF-IDF vectorization for text features
- One-hot encoding for categorical features
- Power transformation for numerical features
- Date component extraction

### Model 3: Basic Neural Network
- Two hidden layers (128 and 64 neurons)
- Batch normalization and dropout
- ReLU activation
- Xavier initialization
- Adam optimizer with learning rate scheduling

### Model 4: Advanced Neural Network with BERT
- Five fully connected layers (1024 → 512 → 256 → 128 → 1)
- BERT embeddings for text processing
- Layer normalization
- GELU activation
- Orthogonal initialization
- Hybrid loss function (MSE + Huber)
- Cosine annealing scheduler

## Requirements

```python
# Core requirements
numpy
pandas
sklearn
tensorflow
torch
transformers
```

## Future Work

Potential improvements and extensions:
- Incorporate temporal trends and seasonal patterns
- Explore alternative transformer-based language models
- Develop ensemble methods combining multiple architectures
- Add more features related to movie marketing and distribution
