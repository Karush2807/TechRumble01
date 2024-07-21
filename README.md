# House Price Prediction Model

## Overview
This project involves predicting house prices using two machine learning models: Linear Regression and Random Forest Regressor. The dataset includes various features such as area, number of bedrooms, number of bathrooms, etc., and the target variable is the house price.

## Data Preprocessing
The dataset was cleaned and preprocessed before being used to train the models. The following steps were taken:
1. **Handling Missing Values**: Missing values were filled or dropped as necessary.
2. **Encoding Categorical Variables**: Categorical variables like `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, and `furnishingstatus` were encoded into numerical values.
3. **Scaling Features**: Features were scaled using `StandardScaler` to normalize the data.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
dataset = pd.read_csv('house_data.csv')

# Data preprocessing
def preprocess_data(df):
    df['mainroad'] = df['mainroad'].apply(lambda x: 1 if x == 'yes' else 0)
    df['guestroom'] = df['guestroom'].apply(lambda x: 1 if x == 'yes' else 0)
    df['basement'] = df['basement'].apply(lambda x: 1 if x == 'yes' else 0)
    df['hotwaterheating'] = df['hotwaterheating'].apply(lambda x: 1 if x == 'yes' else 0)
    df['airconditioning'] = df['airconditioning'].apply(lambda x: 1 if x == 'yes' else 0)
    df['prefarea'] = df['prefarea'].apply(lambda x: 1 if x == 'yes' else 0)
    df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})
    return df

dataset = preprocess_data(dataset)

# Features and target
X = dataset.drop('price', axis=1)
y = dataset['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

