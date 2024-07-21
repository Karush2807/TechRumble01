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
from sklearn.preprocessing import StandardScaler
std_sc=StandardScaler()
x_train_final=std_sc.fit_transform(x_train)
x_test_final=std_sc.transform(x_test)
```
## Random Forest Regressor
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#performing hyperparameter tuning 
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search=GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)
```

## Linear Regresion Model
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error, mean_squared_error
lin_reg=LinearRegression()
lin_reg.fit(x_train_final,y_train)
prediction_of_lr=lin_reg.predict(x_test_final)
mean_absolute_percentage_error(y_test,prediction_of_lr)
mean_squared_error(y_test, prediction_of_lr)

# Save model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)
```