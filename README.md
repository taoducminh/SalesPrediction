
# Sales Prediction Project

This project aims to predict the sales performance of different marketing campaigns based on various features. The data includes factors like TV, Radio, Social Media spending, and influencer types. This project involves data preprocessing, feature engineering, model training, and evaluation using polynomial regression.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Requirements](#requirements)
5. [Setup and Usage](#setup-and-usage)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Acknowledgements](#acknowledgements)

## Project Overview

The goal of this project is to develop a machine learning model that can accurately predict sales figures for different marketing campaigns. The model takes into account various advertisement spending channels (TV, Radio, Social Media) and influencer categories, leveraging polynomial regression to capture non-linear relationships between input features and sales.

## Dataset

- **File:** `SalesPrediction.csv`
- **Attributes**:
  - `TV`: Advertising spend on TV.
  - `Radio`: Advertising spend on Radio.
  - `Social Media`: Advertising spend on Social Media.
  - `Influencer`: Type of influencer involved in the campaign (e.g., Macro, Mega, Micro, Nano).
  - `Sales`: Sales figures associated with each campaign.

## Project Workflow

1. **Data Loading**: Load the dataset into a pandas DataFrame.
2. **Data Preprocessing**:
   - One-hot encode the categorical `Influencer` column.
   - Fill missing values with column means.
3. **Feature Engineering**:
   - Scale features using `StandardScaler`.
   - Generate polynomial features to capture non-linear relationships.
4. **Model Training and Evaluation**:
   - Split the dataset into training and test sets.
   - Train a linear regression model on the polynomial features.
   - Evaluate the model using \( R^2 \) score.

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- numpy

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Setup and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sales-prediction
   cd sales-prediction
   ```

2. **Run the script**:
   Ensure the dataset `SalesPrediction.csv` is in the same directory, then execute the main script.
   ```bash
   python main.py
   ```

## Methodology

### Preprocessing
- **One-Hot Encoding**: Convert the `Influencer` column into binary columns to enable the model to understand categorical data.
- **Handling Missing Values**: Replace any null values in the dataset with the mean of the respective column.

### Feature Engineering
- **Feature Scaling**: Use `StandardScaler` to standardize features, improving model performance and stability.
- **Polynomial Features**: Generate polynomial features of degree 2 to allow the model to learn more complex relationships between input features and sales.

### Model Training and Evaluation
- Split the dataset into training (70%) and testing (30%) sets.
- Train a `LinearRegression` model on polynomial features and evaluate it using the \( R^2 \) score, which measures the model’s accuracy.

## Results

The model is evaluated using the \( R^2 \) score on the test set, which reflects the accuracy of the model's predictions.

## Acknowledgements

- This project is inspired by real-world applications in marketing analytics.
- Special thanks to the contributors of the scikit-learn library for their tools, which make machine learning accessible and efficient.
