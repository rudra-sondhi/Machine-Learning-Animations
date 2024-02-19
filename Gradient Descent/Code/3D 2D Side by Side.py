
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# The complex function
def complex_func(x, y):
    return -np.sin(np.sqrt(x**2 + y**2))

# The gradient of the function
def gradient(x, y):
    r = np.sqrt(x**2 + y**2)
    return -np.array([x/r * np.cos(r), y/r * np.cos(r)])

# Gradient descent path function with precomputed gradients
def gradient_descent_path(start_point, learning_rate, num_steps, momentum=0.9, gravity=0.02, height_threshold=1):
    path = []
    velocity = np.array([0, 0])
    point = start_point
    for _ in range(num_steps):
        path.append(point)
        grad = gradient(*point)
        velocity = momentum * velocity - learning_rate * grad
        point = point + velocity
        func_height = complex_func(*point)
        if func_height > height_threshold:
            velocity -= gravity * (func_height - height_threshold) * velocity
        point = point + velocity
        path.append(point)
    return np.array(path)

# Setup for the dual plot
fig = plt.figure(figsize=(14, 7))

# 3D subplot
ax3d = fig.add_subplot(121, projection='3d')
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
x0, x1 = np.meshgrid(x, y)
z = complex_func(x0, x1)
ax3d.plot_surface(x0, x1, z, alpha=0.6, cmap='viridis')

# Adjusting the view angle for better visibility
ax3d.view_init(elev=30, azim=45)

# 2D subplot with filled contour
ax2d = fig.add_subplot(122)
contour = ax2d.contourf(x0, x1, z, levels=20, cmap='viridis')
ax2d.set_xlim(-3, 3)
ax2d.set_ylim(-3, 3)

# Initialize paths and quivers for both plots
points = np.array([[-1, -2], [2, 1.5], [0, 3], [1, -1.5], [2, 1]])
paths = [gradient_descent_path(point, 0.1, 180) for point in points]
quivers_3d = [ax3d.quiver(0, 0, 0, 0, 0, 0, length=0.4, arrow_length_ratio=0.5, color='red', normalize=True, linewidth = 1.5) for _ in paths]
quivers_2d = [ax2d.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1.25, color='red', headwidth=5, headlength=5, headaxislength=5) for _ in paths]
path_histories = [deque(maxlen=10) for _ in paths]  # For fading trails in 2D plot

# Initialize text annotations for Jacobian values on both plots
text_3d = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes)
text_2d = ax2d.text(0.02, 0.92, "", transform=ax2d.transAxes)

def update(frame):
    artists = []

    for i, path in enumerate(paths):
        idx = frame % len(path)
        x, y = path[idx]
        u, v = gradient(x, y)
        z = complex_func(x, y)

        # Recreating 3D quivers for each frame
        quivers_3d[i].remove()
        quivers_3d[i] = ax3d.quiver(x, y, z, u, v, 0, length=0.4, arrow_length_ratio=0.5, color='red', normalize=True, linewidth=1.5)
        artists.append(quivers_3d[i])

        # Efficiently update 2D quiver properties
        quivers_2d[i].set_UVC(u, v)
        quivers_2d[i].set_offsets([x, y])
        artists.append(quivers_2d[i])

        # Efficiently update path history for fading trails in 2D plot
        path_histories[i].append((x, y))
        while len(path_histories[i]) > 10:
            path_histories[i].popleft()

        # Draw paths with fading effect in 2D plot
        for j, pos in enumerate(path_histories[i]):
            alpha = (j + 1) / len(path_histories[i])
            point, = ax2d.plot(pos[0], pos[1], marker='o', color='blue', alpha=alpha, markersize=3)
            artists.append(point)

    # Update text annotations for Jacobian values on both plots
    x, y = paths[0][frame % len(paths[0])]
    jacobian_values = gradient(x, y)
    text_2d.set_text(f"Position (x, y): ({x:.2f}, {y:.2f})\nJacobian: ({jacobian_values[0]:.2f}, {jacobian_values[1]:.2f})")
    text_3d.set_text(f"Position (x, y): ({x:.2f}, {y:.2f})\nJacobian: ({jacobian_values[0]:.2f}, {jacobian_values[1]:.2f})")
    artists.append(text_2d)
    artists.append(text_3d)

    return artists


# Create the animation with blitting for better performance
ani = animation.FuncAnimation(fig, update, frames=60, interval=5, blit=False, repeat=True)
plt.show()

# Uncomment this line to save the animation
ani.save('/Users/rudrasondhi/Desktop/Animations/gradient_descent_side_by_side_animation.gif', writer='imagemagick', fps=20)  # Increased fps from 10 to 20

