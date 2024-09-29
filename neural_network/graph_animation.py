import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Load the dataset to get the variable names
df = pd.read_csv('Heart.csv')
df = df.drop(columns=['Unnamed: 0'])
df = df.dropna()

# Extract the variable names (excluding the target 'AHD')
variable_names = df.drop(columns=['AHD']).columns.tolist()

# Load the matrices
biases_files = sorted([f for f in os.listdir('biases_matrices') if f.endswith('.npy')])
weights_files = sorted([f for f in os.listdir('weights_matrices') if f.endswith('.npy')])

# Set the frames per second (fps) for the animation
fps = 2  # This can be changed

# Number of iterations
iterations = len(biases_files) // 2  # Assuming two matrices per iteration (hidden and output)

# Load the first matrices to determine the shapes
first_bias_matrix_hidden = np.load(os.path.join('biases_matrices', biases_files[0]))
first_weights_matrix_hidden = np.load(os.path.join('weights_matrices', weights_files[0]))
first_bias_matrix_output = np.load(os.path.join('biases_matrices', biases_files[iterations]))
first_weights_matrix_output = np.load(os.path.join('weights_matrices', weights_files[iterations]))

# Initialize lists to store values over time for each row in the matrices
hidden_weights_values = [[] for _ in range(first_weights_matrix_hidden.shape[0])]
hidden_bias_values = [[] for _ in range(first_bias_matrix_hidden.shape[1])]
output_weights_values = [[] for _ in range(first_weights_matrix_output.shape[0])]
output_bias_values = [[] for _ in range(first_bias_matrix_output.shape[1])]  # Adjusted for output bias

# Create a figure and axes for the animations
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Titles for the subplots (Flipping the order: Weights first, then Biases)
titles = [
    ("Hidden Layer Weights", "Hidden Layer Bias"),
    ("Output Layer Weights", "Output Layer Bias")
]


# Function to update the line plots for each frame
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

    # Update the line plots for hidden layer weights
    axes[0, 0].clear()
    for i in range(len(variable_names)):
        axes[0, 0].plot(hidden_weights_values[i], label=variable_names[i])
    axes[0, 0].set_title(titles[0][0])
    axes[0, 0].legend()

    # Update the line plots for hidden layer biases
    axes[0, 1].clear()
    for i in range(len(hidden_bias_values)):
        axes[0, 1].plot(hidden_bias_values[i], label=f'Bias {i + 1}')
    axes[0, 1].set_title(titles[0][1])
    axes[0, 1].legend()

    # Update the line plots for output layer weights
    axes[1, 0].clear()
    for i in range(len(output_weights_values)):
        axes[1, 0].plot(output_weights_values[i], label=f'Weight {i + 1}')
    axes[1, 0].set_title(titles[1][0])
    axes[1, 0].legend()

    # Update the line plot for output layer bias (only one bias value)
    axes[1, 1].clear()
    for i in range(len(output_bias_values)):
        axes[1, 1].plot(output_bias_values[i], label=f'Output Bias {i + 1}')
    axes[1, 1].set_title(titles[1][1])
    axes[1, 1].legend()


# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=iterations, interval=1000 / fps, blit=False
)

# Save the animation as a video file (optional)
ani.save('line_graphs_animation.mp4', writer='ffmpeg', fps=fps)

# Display the animation
plt.show()
