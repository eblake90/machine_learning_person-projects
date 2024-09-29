# Neural Network Project README

This project provides a comprehensive exploration of neural networks, from implementation to visualization. It consists of several Python scripts that work together to create, train, and visualize a neural network for heart disease prediction. The scripts are designed to provide an in-depth understanding of the entire process of working with neural networks.

## Table of Contents

1. [Neural Network Implementation (nn.py)](#neural-network-implementation-nnpy)
2. [Data Exploration and Visualization (heart_described.py)](#data-exploration-and-visualization-heart_describedpy)
3. [Weight and Bias Animation (animations.py)](#weight-and-bias-animation-animationspy)
4. [Graph Animation (graph_animation.py)](#graph-animation-graph_animationpy)

## Neural Network Implementation (nn.py)

The `nn.py` script is the core of this project. It implements a neural network from scratch using NumPy, trains it on the heart disease dataset, and provides various visualizations of the training process.

### Key Components:

1. **NeuralNetwork Class**: This class encapsulates the entire neural network.

```python
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
```

This constructor initializes the network with random weights and zero biases. The weights are scaled by the square root of the number of inputs to each layer, which is a common initialization technique to prevent vanishing or exploding gradients.

2. **Forward Propagation**: The `forward` method implements the forward pass of the network.

```python
def forward(self, X):
    # Forward propagation
    self.z1 = np.dot(X, self.W1) + self.b1
    self.a1 = self.relu(self.z1)
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    self.a2 = self.sigmoid(self.z2)
    return self.a2
```

This method applies the weights and biases to the input, using ReLU activation for the hidden layer and sigmoid activation for the output layer.

3. **Backward Propagation**: The `backward` method implements the backpropagation algorithm.

```python
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
```

This method calculates the gradients of the loss with respect to each parameter and updates the weights and biases accordingly.

4. **Training**: The `train` method orchestrates the training process.

```python
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
```

This method runs the training loop for a specified number of epochs, periodically calculating and storing performance metrics.

5. **Data Preprocessing**: The script includes a `preprocess_data` function that prepares the heart disease dataset for training.

```python
def preprocess_data(data):
    # ... (code omitted for brevity)

    # Create preprocessing steps for numerical and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ... (more preprocessing steps)

    return X_processed, y, preprocessor
```

This function handles missing values, encodes categorical variables, and scales numerical variables.

6. **Visualization**: The script includes several visualization functions, such as `plot_performance`, `animate_weight_matrices`, and `animate_decision_boundary`.

```python
def animate_weight_matrices(nn, epochs, X_train, y_train, X_test, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    w1_im = ax1.imshow(nn.W1, cmap='viridis', aspect='auto', animated=True)
    w2_im = ax2.imshow(nn.W2, cmap='viridis', aspect='auto', animated=True)

    # ... (animation setup and creation)

    anim.save('weight_matrices_evolution.gif', writer='pillow')
```

This function creates an animation showing how the weight matrices evolve during training.

The `nn.py` script provides a comprehensive implementation of a neural network, including training, evaluation, and visualization. It serves as an excellent tool for understanding the inner workings of neural networks.


---

---
## Data Exploration and Visualization (heart_described.py)

The `heart_described.py` script is a crucial component of our neural network project, focusing on exploratory data analysis (EDA) and visualization of the heart disease dataset. This script helps us understand the characteristics of our data before feeding it into the neural network, potentially informing preprocessing decisions and model architecture choices.

### Purpose

The main purposes of this script are:
1. To provide a comprehensive overview of the dataset's structure and content.
2. To visualize relationships between variables and the target (presence of heart disease).
3. To identify potential patterns, correlations, or anomalies in the data.
4. To generate static visualizations for further analysis and reporting.

### Key Components

#### 1. Data Loading and Initial Preprocessing

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('Heart.csv')

# Remove the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Separate numerical and categorical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
```

This section loads the data and separates numerical and categorical columns. Removing the 'Unnamed: 0' column (if it exists) is a common preprocessing step when working with CSV files that have been saved with an index.

#### 2. Pairplot Visualization

```python
plt.figure(figsize=(20, 15))
sns.pairplot(data, vars=numerical_columns, hue='AHD', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.tight_layout()
plt.savefig('heart_disease_numerical_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
```

This creates a pairplot of all numerical variables, colored by the presence or absence of heart disease (AHD). 

- **Purpose**: To visualize relationships between pairs of numerical variables and how they relate to the target variable (AHD).
- **Insights**: This plot can reveal correlations between variables, potential clusters in the data, and how well individual features separate the two classes (with/without heart disease).

#### 3. Correlation Heatmap

```python
plt.figure(figsize=(12, 10))
sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.savefig('heart_disease_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
```

This generates a heatmap showing the correlation between numerical variables.

- **Purpose**: To identify strong correlations between features, which could influence the neural network's performance or indicate potential for dimensionality reduction.
- **Insights**: Strong correlations (positive or negative) might indicate redundant information, while correlations with the target variable might suggest predictive power.

#### 4. Box Plots for Numerical Variables

```python
fig, axes = plt.subplots(3, 2, figsize=(20, 25))
axes = axes.flatten()
for i, col in enumerate(numerical_columns):
    if i < len(axes):
        sns.boxplot(x='AHD', y=col, data=data, ax=axes[i])
        axes[i].set_title(f'{col} by AHD')
plt.tight_layout()
plt.savefig('heart_disease_numerical_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
```

This creates box plots for each numerical variable, grouped by the presence or absence of heart disease.

- **Purpose**: To visualize the distribution of each numerical variable and how it differs between the two groups (with/without heart disease).
- **Insights**: These plots can reveal differences in central tendency, spread, and potential outliers between the two groups for each variable.

#### 5. Bar Plots for Categorical Variables

```python
fig, axes = plt.subplots(3, 2, figsize=(20, 25))
axes = axes.flatten()
for i, col in enumerate(categorical_columns):
    if i < len(axes):
        sns.countplot(x=col, hue='AHD', data=data, ax=axes[i])
        axes[i].set_title(f'{col} Distribution by AHD')
        axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('heart_disease_categorical_barplots.png', dpi=300, bbox_inches='tight')
plt.close()
```

This creates bar plots for each categorical variable, showing the distribution of values and how they relate to the presence or absence of heart disease.

- **Purpose**: To visualize the distribution of categorical variables and their relationship with the target variable.
- **Insights**: These plots can reveal which categories are more associated with the presence or absence of heart disease, potentially identifying important predictive features.

#### 6. Summary Statistics

```python
print("\nSummary Statistics for Numerical Variables:")
print(data[numerical_columns].describe())

print("\nSummary for Categorical Variables:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(data[col].value_counts(normalize=True))

print("\nCorrelation Matrix for Numerical Variables:")
correlation_matrix = data[numerical_columns].corr()
print(correlation_matrix)
```

This section prints summary statistics for both numerical and categorical variables, as well as the correlation matrix for numerical variables.

- **Purpose**: To provide a quantitative overview of the dataset, including measures of central tendency, dispersion, and correlations.
- **Insights**: These statistics can reveal the scale of different variables, potential class imbalances, and correlations that might not be immediately apparent in visualizations.

### How to Use

To run this script:

1. Ensure you have the required libraries installed (pandas, seaborn, matplotlib, numpy).
2. Place the 'Heart.csv' file in the same directory as the script.
3. Run the script using Python: `python heart_described.py`

The script will generate several PNG files with visualizations and print summary statistics to the console.

### Interpretation and Next Steps

After running this script, you should:

1. Examine the pairplot and correlation heatmap to identify strongly correlated features. This might inform feature selection or engineering steps.
2. Look at the box plots and bar plots to see which features show clear separation between the two classes. These might be particularly important for the neural network.
3. Check the summary statistics for any unusual values or distributions that might require additional preprocessing (e.g., scaling, normalization, or handling of outliers).
4. Consider the distribution of categorical variables. If some categories are very rare, you might need to consider strategies for handling imbalanced data.

The insights gained from this exploratory data analysis can guide decisions about data preprocessing, feature selection, and potentially even the architecture of the neural network (e.g., input size, handling of categorical variables).

By thoroughly understanding your data before modeling, you can make more informed decisions throughout the machine learning pipeline, potentially leading to better model performance and more meaningful results.


---

---

## Weight and Bias Animation (animations.py)

The `animations.py` script is a crucial component of our neural network project, focusing on visualizing the evolution of the network's weights and biases during the training process. This script provides a dynamic and intuitive way to understand how the neural network learns and adapts to the data over time.

### Purpose

The main purposes of this script are:
1. To visualize how weights and biases change during the training process.
2. To provide insights into the learning dynamics of different layers in the network.
3. To help identify potential issues like vanishing or exploding gradients.
4. To create an engaging visual representation of the learning process for educational purposes.

### Key Components

#### 1. Data Loading and Initialization

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Load the matrices
biases_files = sorted([f for f in os.listdir('biases_matrices') if f.endswith('.npy')])
weights_files = sorted([f for f in os.listdir('weights_matrices') if f.endswith('.npy')])

# Set the frames per second (fps) for the animation
fps = 2  # This can be changed

# Number of iterations
iterations = len(biases_files) // 2  # Assuming two matrices per iteration (hidden and output)
```

This section loads the saved weight and bias matrices from each iteration of training. The `iterations` variable is calculated based on the number of files, assuming two matrices (hidden and output layer) per iteration.

#### 2. Setting Up the Animation

```python
# Create a figure and axes for the animations
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Titles for the subplots (Flipping the order: Weights first, then Biases)
titles = [
    ("Hidden Layer Weights", "Hidden Layer Bias"),
    ("Output Layer Weights", "Output Layer Bias")
]
```

This sets up a 2x2 grid of subplots, one for each matrix (hidden layer weights, hidden layer bias, output layer weights, output layer bias). The `titles` list defines the labels for each subplot.

#### 3. Matrix Display Function

```python
def display_matrix(ax, matrix, title):
    ax.clear()
    ax.set_title(title)
    ax.imshow(matrix, cmap='viridis', aspect='auto', interpolation='none')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.4f}', ha='center', va='center', color='white')
```

This function is responsible for displaying a single matrix:
- It clears the previous content of the axis.
- Sets the title of the subplot.
- Displays the matrix as a color-coded image using a viridis colormap.
- Overlays the actual numerical values on each cell of the matrix.

#### 4. Update Function for Animation

```python
def update(frame):
    # Load the current matrices for the hidden layer
    weights_matrix_hidden = np.load(os.path.join('weights_matrices', weights_files[frame]))
    bias_matrix_hidden = np.load(os.path.join('biases_matrices', biases_files[frame]))

    # Load the current matrices for the output layer
    weights_matrix_output = np.load(os.path.join('weights_matrices', weights_files[frame + iterations]))
    bias_matrix_output = np.load(os.path.join('biases_matrices', biases_files[frame + iterations]))

    # Update the plots with the new matrices and their values
    display_matrix(axes[0, 0], weights_matrix_hidden, titles[0][0])
    display_matrix(axes[0, 1], bias_matrix_hidden, titles[0][1])
    display_matrix(axes[1, 0], weights_matrix_output, titles[1][0])
    display_matrix(axes[1, 1], bias_matrix_output, titles[1][1])
```

This function is called for each frame of the animation:
- It loads the current weight and bias matrices for both the hidden and output layers.
- It then calls `display_matrix` for each of the four matrices, updating the corresponding subplot.

#### 5. Creating and Saving the Animation

```python
# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=iterations, interval=1000/fps, blit=False
)

# Save the animation as a video file (optional)
ani.save('corrected_matrices_animation.mp4', writer='ffmpeg', fps=fps)

# Display the animation
plt.show()
```

This section creates the animation using matplotlib's `FuncAnimation`, saves it as an MP4 file, and displays it.

### How to Use

To run this script:

1. Ensure you have the required libraries installed (numpy, matplotlib).
2. Make sure you have run the neural network training script (`nn.py`) first to generate the weight and bias matrices.
3. Ensure the 'weights_matrices' and 'biases_matrices' directories are in the same location as this script.
4. Run the script using Python: `python animations.py`

The script will generate an MP4 file with the animation and display it on screen.

### Interpretation and Insights

When analyzing the animation produced by this script, look for:

1. **Initial state**: Observe the initial random initialization of weights and biases.

2. **Rate of change**: Notice how quickly or slowly different weights and biases change. Fast changes might indicate high learning rates or important features, while slow changes could suggest less important connections or potential vanishing gradients.

3. **Convergence**: See if weights and biases stabilize over time, which could indicate the network reaching a local optimum.

4. **Patterns**: Look for any emerging patterns in the weights, such as certain connections consistently having larger absolute values.

5. **Layer differences**: Compare the behavior of weights and biases in the hidden layer versus the output layer. The output layer might show more dramatic changes initially.

6. **Potential issues**: Watch for signs of potential problems:
   - Exploding gradients: Rapidly increasing absolute values of weights or biases.
   - Vanishing gradients: Weights or biases that barely change, especially in earlier layers.
   - Dead neurons: Rows or columns of weights that remain near zero.

7. **Feature importance**: In the input-to-hidden layer weights, columns with larger absolute values might correspond to more important input features.

This animation provides a unique window into the learning process of your neural network. By visualizing how weights and biases evolve over time, you can gain intuition about the network's behavior, identify potential issues, and potentially make informed decisions about hyperparameter tuning or network architecture adjustments.

Remember that this is a simplified 2D representation of high-dimensional data, so some nuances of the learning process may not be fully captured. Nonetheless, it serves as a powerful tool for understanding and debugging neural networks.



---

---


## Graph Animation (graph_animation.py)

The `graph_animation.py` script creates an animation of how the weights and biases change over time, represented as line graphs. This provides a different perspective on the learning process compared to the matrix visualization in `animations.py`.

### Key Components:

1. **Data Loading and Initialization**:

```python
# Load the dataset to get the variable names
df = pd.read_csv('Heart.csv')
df = df.drop(columns=['Unnamed: 0'])
df = df.dropna()

# Extract the variable names (excluding the target 'AHD')
variable_names = df.drop(columns=['AHD']).columns.tolist()

# Load the matrices
biases_files = sorted([f for f in os.listdir('biases_matrices') if f.endswith('.npy')])
weights_files = sorted([f for f in os.listdir('weights_matrices') if f.endswith('.npy')])

# Number of iterations
iterations = len(biases_files) // 2  # Assuming two matrices per iteration (hidden and output)
```

This section loads the dataset to get the variable names, which will be used for labeling the weights. It also loads the saved weight and bias matrices.

2. **Initializing Data Structures for Tracking Values**:

```python
# Initialize lists to store values over time for each row in the matrices
hidden_weights_values = [[] for _ in range(first_weights_matrix_hidden.shape[0])]
hidden_bias_values = [[] for _ in range(first_bias_matrix_hidden.shape[1])]
output_weights_values = [[] for _ in range(first_weights_matrix_output.shape[0])]
output_bias_values = [[] for _ in range(first_bias_matrix_output.shape[1])]
```

These lists will store the values of weights and biases over time, which will be used to create the line graphs.

3. **Setting Up the Animation**:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Titles for the subplots
titles = [
    ("Hidden Layer Weights", "Hidden Layer Bias"),
    ("Output Layer Weights", "Output Layer Bias")
]
```

This sets up a 2x2 grid of subplots, one for each set of parameters (hidden layer weights, hidden layer biases, output layer weights, output layer biases).

4. **Update Function for Animation**:

```python
def update(frame):
    # Load the current matrices for the hidden layer
    weights_matrix_hidden = np.load(os.path.join('weights_matrices', weights_files[frame]))
    bias_matrix_hidden = np.load(os.path.join('biases_matrices', biases_files[frame]))

    # Load the current matrices for the output layer
    weights_matrix_output = np.load(os.path.join('weights_matrices', weights_files[frame + iterations]))
    bias_matrix_output = np.load(os.path.join('biases_matrices', biases_files[frame + iterations]))

    # Append the values of all rows for weights and biases
    for i in range(first_weights_matrix_hidden.shape[0]):
        hidden_weights_values[i].append(weights_matrix_hidden[i, 0])

    for i in range(first_bias_matrix_hidden.shape[1]):
        hidden_bias_values[i].append(bias_matrix_hidden[0, i])

    for i in range(first_weights_matrix_output.shape[0]):
        output_weights_values[i].append(weights_matrix_output[i, 0])

    for i in range(first_bias_matrix_output.shape[1]):
        output_bias_values[i].append(bias_matrix_output[0, i])

    # Update the line plots for each set of parameters
    # ... (code for updating plots omitted for brevity)
```

This function updates the line plots for each frame of the animation. It loads the current weights and biases, appends their values to the tracking lists, and updates the plots.

5. **Creating and Saving the Animation**:

```python
ani = animation.FuncAnimation(
    fig, update, frames=iterations, interval=1000 / fps, blit=False
)

# Save the animation as a video file
ani.save('line_graphs_animation.mp4', writer='ffmpeg', fps=fps)
```

This creates the animation and saves it as an MP4 file.

The `graph_animation.py` script provides a dynamic visualization of how individual weights and biases change over the course of training. This can be particularly useful for identifying which parameters are changing the most (or least) during training, and how quickly they converge.

## Conclusion

This project provides a comprehensive exploration of neural networks, from implementation to visualization. The scripts work together to:

1. Implement a neural network from scratch (`nn.py`)
2. Explore and visualize the dataset (`heart_described.py`)
3. Animate the changes in weights and biases as matrices (`animations.py`)
4. Animate the changes in weights and biases as line graphs (`graph_animation.py`)

By studying these scripts and their outputs, you can gain a deep understanding of how neural networks work, how they learn from data, and how their internal parameters evolve during training. This project serves as an excellent educational tool for anyone looking to understand the inner workings of neural networks beyond just using pre-built libraries.

## Running the Project

To run this project:

1. Ensure you have all required libraries installed (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).
2. Place the 'Heart.csv' dataset in the same directory as the scripts.
3. Run `nn.py` first to train the neural network and generate the weight and bias matrices.
4. Run `heart_described.py` to generate visualizations of the dataset.
5. Run `animations.py` and `graph_animation.py` to create the animations of weights and biases.

Remember to adjust file paths if necessary, and ensure you have sufficient computational resources, especially for generating the animations.

This project demonstrates the power of visualizing the learning process of neural networks, providing insights that go beyond mere performance metrics. By understanding how the network's parameters change during training, you can gain intuition about the learning process and potentially identify ways to improve the network's architecture or training process.