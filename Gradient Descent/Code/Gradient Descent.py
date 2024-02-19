import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Function to compute the sum of squares
f_sqr = lambda x: np.sum(np.power(x, 2), axis=1)

# Generating the grid
x = np.linspace(-2., 2., 100)
x0, x1 = np.meshgrid(x, x)
X = np.column_stack((x0.flatten(), x1.flatten()))

# Compute function values
y = f_sqr(X)
y = y.reshape((x0.shape))

# Gradient Descent Function
def gradient_descent(start_point, learning_rate, num_iterations, threshold):
    path = [start_point]
    current_point = start_point

    for _ in range(num_iterations):
        grad = np.array([2 * current_point[0], 2 * current_point[1]])  # Gradient calculation
        next_point = current_point - learning_rate * grad  # Gradient descent step

        if np.linalg.norm(f_sqr([next_point]) - f_sqr([current_point])) < threshold:
            break  # Convergence criterion

        current_point = next_point
        path.append(current_point)

    return np.array(path)

# Example usage
start_point = np.array([-2, -2])  # Starting point for the descent
learning_rate = 0.1  # Learning rate
num_iterations = 100  # Maximum number of iterations
threshold = 1e-6  # Convergence threshold

gd_path = gradient_descent(start_point, learning_rate, num_iterations, threshold)

# Creating a figure for the animation
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Function for updating the animation
def update(frame):
    ax.clear()
    ax.plot_surface(x0, x1, y, cmap='viridis', edgecolor='none', alpha=0.5)
    f_val = f_sqr([frame])[0]
    ax.scatter(*frame, f_val, color='r')  # Red point
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$f(x_0, x_1)$')

# Initialization function
def init():
    ax.plot_surface(x0, x1, y, cmap='viridis', edgecolor='none', alpha=0.5)
    return fig,

# Creating an animation
ani = animation.FuncAnimation(fig, update, frames=gd_path, init_func=init, blit=False, repeat=True)

plt.show()

# Save the animation
fps = 10  # Frames per second
ani.save('/Users/rudrasondhi/Desktop/Animations/gradient_descent_simple.gif', writer='imagemagick', fps=20)  # Increased fps from 10 to 20
