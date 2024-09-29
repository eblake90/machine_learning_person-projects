import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from matplotlib.animation import FuncAnimation


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        # Backward propagation
        m = X.shape[0]
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0)

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X_train, y_train, X_test, y_test, epochs, learning_rate, monitor_every=10):
        self.learning_rate = learning_rate
        history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(epochs):
            output = self.forward(X_train)
            self.backward(X_train, y_train, output)

            if epoch % monitor_every == 0:
                train_loss = self.binary_cross_entropy(y_train, output)
                train_acc = self.accuracy(y_train, self.predict(X_train))
                test_acc = self.accuracy(y_test, self.predict(X_test))

                history['epoch'].append(epoch)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)

                print(f"Epoch {epoch}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        return history

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def preprocess_data(data):
    # Drop the unnamed column and any rows with missing values
    data = data.drop('Unnamed: 0', axis=1).dropna()

    # Separate features and target
    X = data.drop('AHD', axis=1)
    y = data['AHD']

    # Define categorical and numerical columns
    categorical_features = ['ChestPain', 'Thal', 'Sex', 'Fbs', 'RestECG', 'ExAng', 'Slope']
    numerical_features = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca']

    # Create preprocessing steps for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Convert target to numerical: 'Yes' -> 1, 'No' -> 0
    y = (y == 'Yes').astype(int)

    return X_processed, y, preprocessor


def plot_performance(history):
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
    plt.plot(history['epoch'], history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss over Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()


def animate_weight_matrices(nn, epochs, X_train, y_train, X_test, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    w1_im = ax1.imshow(nn.W1, cmap='viridis', aspect='auto', animated=True)
    w2_im = ax2.imshow(nn.W2, cmap='viridis', aspect='auto', animated=True)

    ax1.set_title('W1')
    ax2.set_title('W2')

    fig.colorbar(w1_im, ax=ax1)
    fig.colorbar(w2_im, ax=ax2)

    def update(frame):
        nn.train(X_train, y_train, X_test, y_test, epochs=1, learning_rate=0.01)
        w1_im.set_array(nn.W1)
        w2_im.set_array(nn.W2)
        return w1_im, w2_im

    anim = FuncAnimation(fig, update, frames=epochs, interval=50, blit=True)
    anim.save('weight_matrices_evolution.gif', writer='pillow')
    plt.close(fig)


def animate_decision_boundary(nn, X, y, epochs):
    fig, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', edgecolors='black')
    contourf = ax.contourf(xx, yy, np.zeros_like(xx), cmap='viridis', alpha=0.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('Decision Boundary Evolution')

    def init():
        return scatter, contourf

    def update(frame):
        nn.train(X, y, X, y, epochs=1, learning_rate=0.01)
        Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        for c in ax.collections:
            c.remove()

        contourf = ax.contourf(xx, yy, Z, cmap='viridis', alpha=0.4)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', edgecolors='black')

        return scatter, contourf

    anim = FuncAnimation(fig, update, frames=epochs, init_func=init, blit=False, repeat=False)

    try:
        anim.save('decision_boundary_evolution.gif', writer='pillow', fps=10)
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.close(fig)

def main():
    # Load the data
    data = pd.read_csv('Heart.csv')

    # Preprocess the data
    X, y, preprocessor = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert y_train and y_test to numpy arrays and reshape
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    # Initialize and train the neural network
    input_size = X_train.shape[1]
    hidden_size = 10
    output_size = 1

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Create animations
    print("Creating weight matrices animation...")
    animate_weight_matrices(nn, epochs=100, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    print("Creating decision boundary animation...")
    # Use only the first two numerical features for visualization
    numerical_features = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca']

    # Create a new preprocessor for 2D visualization
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor_2d = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features[:2])  # Use only the first two numerical features
        ])

    # Preprocess the data for 2D visualization
    X_2d = data[numerical_features[:2]]
    y_2d = data['AHD']

    # Remove any rows with NaN values
    mask = ~X_2d.isna().any(axis=1)
    X_2d = X_2d[mask]
    y_2d = y_2d[mask]

    # Now preprocess X_2d
    X_2d = preprocessor_2d.fit_transform(X_2d)

    # Convert y_2d to numeric values
    y_2d = (y_2d == 'Yes').astype(int).to_numpy().reshape(-1, 1)

    nn_2d = NeuralNetwork(2, hidden_size, output_size)
    animate_decision_boundary(nn_2d, X_2d, y_2d, epochs=100)

    # Train the network
    print("Training the neural network...")
    history = nn.train(X_train, y_train, X_test, y_test, epochs=1000, learning_rate=0.01, monitor_every=10)

    # Save history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv('nn_performance_history.csv', index=False)
    print("Performance history saved to 'nn_performance_history.csv'")

    # Plot performance
    plot_performance(history)

    # Final evaluation
    final_train_acc = nn.accuracy(y_train, nn.predict(X_train))
    final_test_acc = nn.accuracy(y_test, nn.predict(X_test))
    print(f"Final Train Accuracy: {final_train_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")


if __name__ == "__main__":
    main()