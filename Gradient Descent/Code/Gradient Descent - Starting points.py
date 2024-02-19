import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def complex_func(x, y):
    # Inverting the function to be concave up
    return -np.sin(np.sqrt(x**2 + y**2))

def gradient(x, y):
    # Gradient of the inverted function
    r = np.sqrt(x**2 + y**2)
    return -np.array([x/r * np.cos(r), y/r * np.cos(r)])  # Negating the original gradient

def gradient_descent_path(start_point, learning_rate, num_steps, momentum=0.75, gravity=0.04, height_threshold=1):
    path = []
    velocity = np.array([0, 0])
    point = start_point

    for _ in range(num_steps):
        path.append(point)
        grad = gradient(*point)
        velocity = momentum * velocity - learning_rate * grad
        point = point + velocity

        # Apply gravity-like effect if above threshold height
        func_height = complex_func(*point)
        if func_height > height_threshold:
            # Reduce the velocity, 'pulling' the point back towards lower function values
            velocity -= gravity * (func_height - height_threshold) * velocity

        point = point + velocity
        path.append(point)

    return np.array(path)

# Plot setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-3, 3, 100)
x0, x1 = np.meshgrid(x, x)
z = complex_func(x0, x1)
ax.plot_surface(x0, x1, z, alpha=0.6, cmap='viridis')

# Paths and quivers
points = np.array([[-1, -2], [2, 1.5], [0, 3], [1, -1.5], [2, 1]])
paths = [gradient_descent_path(point, 0.1, 180) for point in points]
quivers = []
for path in paths:
    x, y = path[0]
    u, v = gradient(x, y)
    z = complex_func(x, y)
    quiver = ax.quiver(x, y, z, u, v, 0, length=0.3, color='red', normalize=True)
    quivers.append(quiver)

def update(frame):
    global quivers
    new_quivers = []
    for quiver, path in zip(quivers, paths):
        quiver.remove()
        idx = frame % len(path)
        x, y = path[idx]
        u, v = gradient(x, y)
        z = complex_func(x, y)
        new_quiver = ax.quiver(x, y, z, u, v, 0, length=0.3 + 0.1 * np.linalg.norm([u, v]), color='red', normalize=True)
        new_quivers.append(new_quiver)
    quivers = new_quivers
    return quivers

ani = animation.FuncAnimation(fig, update, frames=54, repeat=True, blit = False, interval = 25)
plt.show()

fps = 10  # Frames per second
ani.save('/Users/rudrasondhi/Desktop/Animations/gradient_descent_animation.gif', writer='imagemagick', fps=20)  # Increased fps from 10 to 20
