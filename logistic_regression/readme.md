# Understanding Logistic Regression

This README explains the key concepts of logistic regression using the implementation in the provided Python script. The script demonstrates logistic regression for heart disease prediction.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Logistic Regression Functions](#logistic-regression-functions)
4. [Visualization](#visualization)
5. [Model Training and Evaluation](#model-training-and-evaluation)

## Introduction

Logistic regression is a statistical method for predicting a binary outcome based on one or more independent variables. In this script, we use logistic regression to predict the presence of heart disease based on various health indicators.

## Data Preprocessing

Before applying logistic regression, we need to prepare our data:

```python
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['AHD'] = (data['AHD'] == 'Yes').astype(int)
    
    cat_columns = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Thal']
    for col in cat_columns:
        data[col] = pd.Categorical(data[col]).codes
    
    data = data.dropna()
    X = data.drop(['AHD'], axis=1)
    y = data['AHD']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
```

Key steps:
1. Convert the target variable to binary (0 or 1).
2. Encode categorical variables as numeric.
3. Handle missing data.
4. Split data into training and testing sets.
5. Standardize feature values.

## Logistic Regression Functions

### Sigmoid Function

The sigmoid function maps any input to a value between 0 and 1, representing a probability:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### Cost Function

The cost function measures how well the model's predictions match the actual data:

```python
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost
```

### Gradient Function

The gradient function calculates the direction of steepest increase in the cost:

```python
def gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1 / m) * X.T @ (h - y)
    return grad
```

### Logistic Regression

The logistic regression function uses gradient descent to optimize the model parameters:

```python
def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    costs = []
    
    for _ in range(num_iterations):
        theta = theta - learning_rate * gradient(X, y, theta)
        costs.append(cost_function(X, y, theta))
    
    return theta, costs
```

## Visualization

The script includes functions to visualize the cost surface and the optimization process:

```python
def create_cost_surface(X, y, param_index, theta_range, num_points=100):
    
def visualize_all_params_vs_ahd(X, y, feature_names, theta_range=(-20, 20), num_points=50):


def plot_cost_vs_iterations(costs):

```

These functions help in understanding how the cost changes with different parameter values and how the optimization progresses over iterations.

## Model Training and Evaluation

The main analysis function ties everything together:

```python
def perform_logistic_regression_analysis(file_path):
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    visualize_all_params_vs_ahd(X_train_b, y_train, feature_names)
    
    theta, costs = logistic_regression(X_train_b, y_train, learning_rate=0.1, num_iterations=1000)
    
    plot_cost_vs_iterations(costs)
    
    y_pred = sigmoid(X_test_b @ theta) >= 0.5
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")
    
    print("\nModel Coefficients:")
    print(f"Bias: {theta[0]:.4f}")
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {theta[i + 1]:.4f}")
```

This function:
1. Loads and preprocesses the data.
2. Visualizes the cost surfaces.
3. Trains the logistic regression model.
4. Plots the cost vs. iterations.
5. Makes predictions and calculates accuracy.
6. Prints the model coefficients.

By running this script and examining its output, you can gain a deeper understanding of how logistic regression works, from data preprocessing to model evaluation.