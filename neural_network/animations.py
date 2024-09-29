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

# Load the first matrices to determine the shapes
first_bias_matrix_hidden = np.load(os.path.join('biases_matrices', biases_files[0]))
first_weights_matrix_hidden = np.load(os.path.join('weights_matrices', weights_files[0]))
first_bias_matrix_output = np.load(os.path.join('biases_matrices', biases_files[iterations]))
first_weights_matrix_output = np.load(os.path.join('weights_matrices', weights_files[iterations]))

# Create a figure and axes for the animations
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Titles for the subplots (Flipping the order: Weights first, then Biases)
titles = [
    ("Hidden Layer Weights", "Hidden Layer Bias"),
    ("Output Layer Weights", "Output Layer Bias")
]

# Function to display a matrix with text annotations
def display_matrix(ax, matrix, title):
    ax.clear()
    ax.set_title(title)
    ax.imshow(matrix, cmap='viridis', aspect='auto', interpolation='none')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.4f}', ha='center', va='center', color='white')

# Function to update the images for each frame
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

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=iterations, interval=1000/fps, blit=False
)

# Save the animation as a video file (optional)
ani.save('corrected_matrices_animation.mp4', writer='ffmpeg', fps=fps)

# Display the animation
plt.show()
