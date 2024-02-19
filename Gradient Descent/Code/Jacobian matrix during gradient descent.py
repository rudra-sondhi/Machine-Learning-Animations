import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def func(x, y):
    # Return a scalar value suitable for 3D surface plot
    return x**2 + y**2

def jacobian(x, y):
    # Jacobian for the scalar function
    return np.array([2 * x, 2 * y])

def gradient_descent(start_point, learning_rate, num_iterations):
    path = [start_point]
    current_point = start_point

    for _ in range(num_iterations):
        grad = jacobian(*current_point)  # Use the simplified gradient
        next_point = current_point - learning_rate * grad

        current_point = next_point
        path.append(current_point)

    return np.array(path)

# Example usage
start_point = np.array([-2, -2])  # Adjusted starting point
learning_rate = 0.045  # Adjusted learning rate
num_iterations = 50  # Number of iterations

path = gradient_descent(start_point, learning_rate, num_iterations)

# Create a 3D plot for the animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate a grid to plot the function
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)



def update(frame):
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    x_pos, y_pos = path[frame]
    z_pos = func(x_pos, y_pos)  # Correct usage of func

    # Plotting the path up to the current frame
    path_x, path_y = path[:frame + 1, 0], path[:frame + 1, 1]
    path_z = [func(px, py) for px, py in zip(path_x, path_y)]  # List comprehension for z values
    ax.plot(path_x, path_y, path_z, color='r', marker='o', markersize=5)

    # Updating the arrow for the gradient
    grad = jacobian(x_pos, y_pos)
    ax.quiver(x_pos, y_pos, z_pos, -grad[0], -grad[1], 0, color='blue', length=0.5, normalize=True)

    # Adding a text label for the mathematical operation
    ax.text2D(0.05, 0.95, f"Jacobian at ({x_pos:.2f}, {y_pos:.2f}): ({-grad[0]:.2f}, {-grad[1]:.2f})", transform=ax.transAxes)

# Initialize the animation
def init():
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    return fig,

# Creating the animation
ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init, repeat=True, interval = 100)

plt.show()
ani.save('/Users/rudrasondhi/Desktop/Animations/jacobian.gif', writer='imagemagick', fps=20)  # Increased fps from 10 to 20
