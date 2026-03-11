import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline

def get_tube_surface(curve, radius, n_segments=15):
    """Generates X, Y, Z mesh grids for a tube around a 3D curve."""
    n_points = len(curve)
    theta = np.linspace(0, 2 * np.pi, n_segments)

    # Storage for mesh vertices
    X, Y, Z = np.zeros((n_points, n_segments)), np.zeros((n_points, n_segments)), np.zeros((n_points, n_segments))

    for i in range(n_points):
        p = curve[i]
        # Calculate tangent vector for the orientation of the circle
        if i < n_points - 1:
            tangent = curve[i+1] - curve[i]
        else:
            tangent = curve[i] - curve[i-1]

        tangent /= np.linalg.norm(tangent)

        # Create an orthogonal basis (Normal and Binormal)
        helper = np.array([1, 0, 0]) if abs(tangent[0]) < 0.9 else np.array([0, 1, 0])
        normal = np.cross(tangent, helper)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)

        # Create the ring of points around the centerline
        for j, t in enumerate(theta):
            ring_point = p + radius * (np.cos(t) * normal + np.sin(t) * binormal)
            X[i, j], Y[i, j], Z[i, j] = ring_point

    return X, Y, Z

# --- CONFIGURATION ---
radius = 1.0  # Physical thickness of the spline
bounds = {'x': (0, 15), 'y': (0, 15), 'z': (0, 20)}

# Start points and one End point (inside the bounds)
# starts = np.array([
#     [0, 12.5, 20],
#     [0, 7.5, 20],
#     [0, 5, 20],
#     [0, 0, 20],
# ])
starts = np.array([
    [0, 12.5, 20],
    [0, 10, 20],
    [0, 7.5, 20],
    [0, 5, 20],
    [0, 2.5, 20],
    [0, 0, 20],
])
end = np.array([2.5, 15, 0])

# Directions (Tangents) - Multiplier controls the "bend" radius
start_dir = np.array([1, 0, 0]) * 50
end_dir = np.array([0, 1, 0]) * 20

# --- PLOTTING ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#600505", "#27d3d6"]

for i, p_start in enumerate(starts):
    # Create the Spline
    spline = CubicHermiteSpline([0, 1], [p_start, end], [start_dir, end_dir])
    t_steps = np.linspace(0, 1, 40)
    curve = spline(t_steps)

    # Generate the 3D Pipe Surface
    X, Y, Z = get_tube_surface(curve, radius)

    # Plot the surface
    ax.plot_surface(X, Y, Z, color=colors[i], alpha=0.6, linewidth=0, antialiased=True)
    # Plot the centerline for clarity
    ax.plot(curve[:,0], curve[:,1], curve[:,2], color='black', lw=1, alpha=0.5)

# Draw Bounding Box (Wireframe)
for x in bounds['x']:
    for y in bounds['y']:
        ax.plot([x, x], [y, y], bounds['z'], color='gray', linestyle='--', alpha=0.3)
for x in bounds['x']:
    for z in bounds['z']:
        ax.plot([x, x], bounds['y'], [z, z], color='gray', linestyle='--', alpha=0.3)
for y in bounds['y']:
    for z in bounds['z']:
        ax.plot(bounds['x'], [y, y], [z, z], color='gray', linestyle='--', alpha=0.3)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f"3D Manifold: 4 Starts to 1 End (Radius r={radius})")
ax.view_init(elev=20, azim=45)
plt.show()