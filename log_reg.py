import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Convert the 'AHD' (Angiographic Heart Disease) column to binary
    # 'Yes' becomes 1, 'No' becomes 0
    data['AHD'] = (data['AHD'] == 'Yes').astype(int)

    # Convert categorical variables to numeric codes
    # This is necessary because logistic regression requires numeric inputs
    cat_columns = ['Sex', 'ChestPain', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Thal']
    for col in cat_columns:
        # pd.Categorical(data[col]).codes assigns a unique integer to each category
        data[col] = pd.Categorical(data[col]).codes

    # Remove any rows with missing values
    # This is a simple way to handle missing data, though it may not always be the best approach
    data = data.dropna()

    # Separate features (X) and target variable (y)
    # X contains all columns except 'AHD', y contains only 'AHD'
    X = data.drop(['AHD'], axis=1)
    y = data['AHD']

    # Split the data into training and testing sets
    # 80% of the data is used for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature values
    # This ensures all features are on the same scale, which is important for many ML algorithms
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


# 2. Logistic Regression Functions
def sigmoid(z):
    # The sigmoid function maps any input to a value between 0 and 1
    # It's used in logistic regression to model probabilities
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    # This function calculates the cost (error) of our current model parameters
    m = len(y)  # Number of training examples
    h = sigmoid(X @ theta)  # Model predictions

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-5

    # Calculate the cost using the logistic regression cost function formula
    # This is the average log-likelihood over all training examples
    cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost


def gradient(X, y, theta):
    # This function calculates the gradient of the cost function
    # The gradient points in the direction of steepest increase in the cost
    m = len(y)  # Number of training examples
    h = sigmoid(X @ theta)  # Model predictions

    # Calculate the gradient using the logistic regression gradient formula
    grad = (1 / m) * X.T @ (h - y)
    return grad


def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    # This function performs logistic regression using gradient descent
    m, n = X.shape  # m: number of examples, n: number of features
    theta = np.zeros(n)  # Initialize model parameters to zero
    costs = []  # To store cost at each iteration for visualization

    for _ in range(num_iterations):
        # Update theta using the gradient descent update rule
        theta = theta - learning_rate * gradient(X, y, theta)
        # Calculate and store the cost for this iteration
        costs.append(cost_function(X, y, theta))

    return theta, costs


# 3. Visualization Functions
def create_cost_surface(X, y, param_index, theta_range, num_points=100):
    # This function creates a cost surface for visualizing the optimization landscape
    # It varies two parameters (bias and one feature) while keeping others constant

    # Create a range of values for theta0 (bias) and theta1 (chosen feature)
    theta0_range = np.linspace(theta_range[0], theta_range[1], num_points)
    theta1_range = np.linspace(theta_range[0], theta_range[1], num_points)
    cost_surface = np.zeros((num_points, num_points))

    # Calculate the cost for each combination of theta0 and theta1
    for i, t0 in enumerate(theta0_range):
        for j, t1 in enumerate(theta1_range):
            theta = np.zeros(X.shape[1])
            theta[0] = t0
            theta[param_index] = t1
            cost_surface[i, j] = cost_function(X, y, theta)

    return theta0_range, theta1_range, cost_surface


def visualize_all_params_vs_ahd(X, y, feature_names, theta_range=(-20, 20), num_points=50):
    # This function creates visualizations for each feature vs. the bias term
    n_params = X.shape[1] - 1  # Number of features (excluding bias)

    for i in range(n_params):
        # Create cost surface for current feature
        theta0_range, theta1_range, cost_surface = create_cost_surface(X, y, i + 1, theta_range, num_points)
        theta0, theta1 = np.meshgrid(theta0_range, theta1_range)

        # Create 3D surface plot
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        surf = ax1.plot_surface(theta0, theta1, cost_surface, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Bias')
        ax1.set_ylabel(feature_names[i])
        ax1.set_zlabel('Cost')
        ax1.set_title(f'Cost Surface: AHD vs {feature_names[i]}')
        fig1.tight_layout()

        # Create contour plot
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)
        contour = ax2.contour(theta0, theta1, cost_surface, levels=20)
        ax2.clabel(contour, inline=1, fontsize=8)
        ax2.set_xlabel('Bias')
        ax2.set_ylabel(feature_names[i])
        ax2.set_title(f'Contour Plot: AHD vs {feature_names[i]}')
        fig2.tight_layout()

    plt.show()


def plot_cost_vs_iterations(costs):
    # This function plots how the cost changes over iterations during training
    plt.figure(figsize=(10, 5))
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations')
    plt.show()


# 4. Main Analysis Function
def perform_logistic_regression_analysis(file_path):
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)

    # Add bias term to the feature matrices
    # This is equivalent to adding a column of 1's as the first column
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Visualize cost surfaces for each feature
    visualize_all_params_vs_ahd(X_train_b, y_train, feature_names)

    # Perform logistic regression
    theta, costs = logistic_regression(X_train_b, y_train, learning_rate=0.1, num_iterations=1000)

    # Plot how the cost changes over iterations
    plot_cost_vs_iterations(costs)

    # Make predictions on the test set
    # Classify as 1 if probability > 0.5, else 0
    y_pred = sigmoid(X_test_b @ theta) >= 0.5

    # Calculate and print the accuracy of the model
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Print the model coefficients
    # These indicate the importance and direction of influence for each feature
    print("\nModel Coefficients:")
    print(f"Bias: {theta[0]:.4f}")
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {theta[i + 1]:.4f}")


# 5. Run the analysis
if __name__ == "__main__":
    file_path = 'Heart.csv'  # Specify the path to your dataset
    perform_logistic_regression_analysis(file_path)